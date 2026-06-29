"""
Template Matching — Flask Web Application
==========================================
REST API + modern web UI for template matching.
"""

import io
import os
import time
import base64
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, jsonify, send_file, session,
)

from .template_matcher import (
    TemplateMatcher, MatchMethod, MatchResult, DetectionReport,
    draw_matches, draw_match_comparison, draw_match_with_differences,
)
from .multi_scale_matcher import MultiScaleMatcher
from .forgery_detector import (
    ForgeryDetector, ForgeryReport, ForgeryRisk,
    draw_difference_boxes, draw_forgery_report, create_full_analysis_canvas,
)
from .document_forensics import (
    DocumentForensicsEngine, DocumentForensicsReport, TamperSeverity,
)
from .document_fingerprint import FingerprintVault, get_vault
from .signature_verifier import SignatureVerifier, SignatureReport, SignatureVerdict
from .metadata_analyzer import MetadataAnalyzer, MetadataReport
from .forensic_reporter import ForensicReporter
from .copy_move_detector import CopyMoveDetector, CopyMoveReport
from .font_analyzer import FontAnalyzer, FontReport
from .authenticity_scorer import AuthenticityScorer, AuthenticityReport
from .advanced_detectors import AdvancedDetectors, AdvancedReport

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    import os
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    app = Flask(__name__, template_folder=template_dir)
    app.secret_key = "tm-secret-" + str(uuid.uuid4())
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

    # In-memory storage per session
    app.config["IMAGES"] = {}

    def _get_image(sid: Optional[str], kind: str) -> Optional[np.ndarray]:
        """Retrieve a stored image from the in-memory cache."""
        if not sid:
            return None
        return app.config["IMAGES"].get(f"{sid}_{kind}")

    # ------------------------------------------------------------------
    # Routes — Pages
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------------
    # Routes — API
    # ------------------------------------------------------------------

    @app.route("/api/upload-source", methods=["POST"])
    def upload_source():
        """Upload the source (main) image."""
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image provided"}), 400

        img = _read_image(file)
        sid = session.get("session_id") or str(uuid.uuid4())
        session["session_id"] = sid
        app.config["IMAGES"][f"{sid}_source"] = img

        return jsonify({
            "ok": True,
            "width": img.shape[1],
            "height": img.shape[0],
            "preview": _img_to_b64(img, max_dim=500),
        })

    @app.route("/api/upload-template", methods=["POST"])
    def upload_template():
        """Upload the template image."""
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image provided"}), 400

        img = _read_image(file)
        sid = session.get("session_id") or str(uuid.uuid4())
        session["session_id"] = sid
        app.config["IMAGES"][f"{sid}_template"] = img

        return jsonify({
            "ok": True,
            "width": img.shape[1],
            "height": img.shape[0],
            "preview": _img_to_b64(img, max_dim=300),
        })

    @app.route("/api/crop-template", methods=["POST"])
    def crop_template():
        """Crop a region from the source image to use as template."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")
        if source is None:
            return jsonify({"error": "Upload source image first"}), 400

        data = request.get_json() or {}
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        w = int(data.get("w", 50))
        h = int(data.get("h", 50))

        # Clamp to image bounds
        x = max(0, min(x, source.shape[1] - 1))
        y = max(0, min(y, source.shape[0] - 1))
        w = max(5, min(w, source.shape[1] - x))
        h = max(5, min(h, source.shape[0] - y))

        template = source[y:y + h, x:x + w].copy()
        app.config["IMAGES"][f"{sid}_template"] = template

        return jsonify({
            "ok": True,
            "width": template.shape[1],
            "height": template.shape[0],
            "preview": _img_to_b64(template, max_dim=300),
        })

    @app.route("/api/match", methods=["POST"])
    def run_match():
        """Execute template matching."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")
        template = _get_image(sid, "template")

        if source is None or template is None:
            return jsonify({"error": "Upload both source and template images first"}), 400

        data = request.get_json() or {}
        method_name = data.get("method", "TM_CCOEFF_NORMED")
        threshold = float(data.get("threshold", 0.80))
        multi_scale = bool(data.get("multi_scale", False))
        scale_min = float(data.get("scale_min", 0.3))
        scale_max = float(data.get("scale_max", 2.5))
        scale_steps = int(data.get("scale_steps", 25))
        nms_threshold = float(data.get("nms_threshold", 0.3))
        max_matches = int(data.get("max_matches", 50))

        try:
            method = MatchMethod[method_name]
        except KeyError:
            method = MatchMethod.TM_CCOEFF_NORMED

        # Run matching
        if multi_scale:
            matcher = MultiScaleMatcher(
                method=method, threshold=threshold,
                scale_range=(scale_min, scale_max),
                scale_steps=scale_steps,
                nms_threshold=nms_threshold,
            )
        else:
            matcher = TemplateMatcher(
                method=method, threshold=threshold,
                use_nms=True, nms_threshold=nms_threshold,
            )

        report = matcher.match(source, template, max_matches=max_matches)

        # Draw result with validation indicators
        result_img = draw_matches(source, report, show_validation=True)

        # Build response
        matches_json = []
        for i, m in enumerate(report.matches):
            matches_json.append({
                "id": i + 1,
                "x": m.x,
                "y": m.y,
                "width": m.width,
                "height": m.height,
                "confidence": round(m.confidence, 4),
                "scale": round(m.scale, 2),
            })

        validation_info = {}
        if report.validation is not None:
            validation_info = {
                "is_valid": report.validation.is_valid,
                "confidence": round(report.validation.confidence, 4),
                "ssim_score": round(report.validation.ssim_score, 4),
                "histogram_correlation": round(report.validation.histogram_correlation, 4),
                "edge_overlap": round(report.validation.edge_overlap, 4),
                "pixel_similarity": round(report.validation.pixel_similarity, 4),
                "reasons": report.validation.reasons,
            }

        return jsonify({
            "ok": True,
            "match_count": report.match_count,
            "method": report.method.label,
            "threshold": report.threshold,
            "elapsed_ms": round(report.elapsed_ms, 1),
            "matches": matches_json,
            "result_preview": _img_to_b64(result_img, max_dim=900),
            "source_shape": list(report.source_shape),
            "template_shape": list(report.template_shape),
            "validation": validation_info,
        })

    @app.route("/api/compare", methods=["POST"])
    def run_compare():
        """Compare all 6 matching methods."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")
        template = _get_image(sid, "template")

        if source is None or template is None:
            return jsonify({"error": "Upload both source and template images first"}), 400

        data = request.get_json() or {}
        threshold = float(data.get("threshold", 0.80))

        matcher = TemplateMatcher(threshold=threshold)
        reports = matcher.match_all_methods(source, template, threshold=threshold)

        comparison_img = draw_match_comparison(source, reports)

        methods_result = []
        for r in reports:
            methods_result.append({
                "method": r.method.label,
                "method_key": r.method.name,
                "match_count": r.match_count,
                "elapsed_ms": round(r.elapsed_ms, 1),
                "threshold": r.threshold,
            })

        return jsonify({
            "ok": True,
            "methods": methods_result,
            "comparison_preview": _img_to_b64(comparison_img, max_dim=1200),
        })

    @app.route("/api/forgery-analysis", methods=["POST"])
    def forgery_analysis():
        """Run comprehensive forgery/tampering detection."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")
        template = _get_image(sid, "template")

        if source is None or template is None:
            return jsonify({"error": "Upload both source and template images first"}), 400

        data = request.get_json() or {}
        threshold = float(data.get("threshold", 0.80))
        method_name = data.get("method", "TM_CCOEFF_NORMED")
        multi_scale = bool(data.get("multi_scale", False))

        try:
            method = MatchMethod[method_name]
        except KeyError:
            method = MatchMethod.TM_CCOEFF_NORMED

        # Step 1: Run matching with validation
        if multi_scale:
            matcher = MultiScaleMatcher(
                method=method, threshold=threshold,
                scale_range=(0.3, 2.5), scale_steps=25,
            )
        else:
            matcher = TemplateMatcher(
                method=method, threshold=threshold,
                use_nms=True, validate_matches=True, strict_validation=True,
            )

        report = matcher.match(source, template)

        # Step 2: Run forgery detection
        detector = ForgeryDetector()
        matched_regions = [(m.x, m.y, m.width, m.height) for m in report.matches]
        forgery_report = detector.analyze(source, template, matched_regions)

        # Step 3: Run quick validation on matched regions
        if matched_regions:
            is_valid, valid_conf, valid_reasons = detector.quick_validation(
                source, template, matched_regions
            )
        else:
            is_valid, valid_conf, valid_reasons = False, 0.0, ["No regions matched"]

        # Step 4: Generate visualizations
        # Main result with green difference boxes
        result_img = draw_match_with_differences(source, template, report)

        # Full analysis canvas
        analysis_canvas = create_full_analysis_canvas(
            source, template, forgery_report, matched_regions
        )

        # Build response
        matches_json = []
        for i, m in enumerate(report.matches):
            matches_json.append({
                "id": i + 1,
                "x": m.x,
                "y": m.y,
                "width": m.width,
                "height": m.height,
                "confidence": round(m.confidence, 4),
                "scale": round(m.scale, 2),
            })

        return jsonify({
            "ok": True,
            "match_count": report.match_count,
            "method": report.method.label,
            "threshold": report.threshold,
            "elapsed_ms": round(report.elapsed_ms, 1),
            "matches": matches_json,
            "result_preview": _img_to_b64(result_img, max_dim=900),
            "analysis_preview": _img_to_b64(analysis_canvas, max_dim=1200),

            # Validation results
            "is_genuine": is_valid,
            "validation_confidence": round(valid_conf, 4),
            "validation_reasons": valid_reasons,

            # Forgery analysis
            "forgery_risk": forgery_report.risk_level.name,
            "forgery_risk_label": forgery_report.risk_level.label,
            "forgery_risk_score": round(forgery_report.risk_score, 4),
            "ssim_score": round(forgery_report.ssim_score, 4),
            "ela_score": round(forgery_report.ela_score, 4),
            "noise_consistency": round(forgery_report.noise_consistency, 4),
            "histogram_correlation": round(forgery_report.histogram_correlation, 4),
            "edge_anomaly_score": round(forgery_report.edge_anomaly_score, 4),
            "diff_regions_count": len(forgery_report.diff_regions),
            "analysis_details": forgery_report.details,
            "analysis_elapsed_ms": round(forgery_report.elapsed_ms, 1),

            "source_shape": list(report.source_shape),
            "template_shape": list(report.template_shape),
        })

    @app.route("/api/validate-match", methods=["POST"])
    def validate_match():
        """Run ONLY validation on a match without full forgery analysis."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")
        template = _get_image(sid, "template")

        if source is None or template is None:
            return jsonify({"error": "Upload both source and template images first"}), 400

        data = request.get_json() or {}
        threshold = float(data.get("threshold", 0.80))
        strict = bool(data.get("strict", True))

        # Run matching
        matcher = TemplateMatcher(
            threshold=threshold, validate_matches=True, strict_validation=strict,
        )
        report = matcher.match(source, template)

        matched_regions = [(m.x, m.y, m.width, m.height) for m in report.matches]

        # Quick validation
        detector = ForgeryDetector()
        is_valid, conf, reasons = detector.quick_validation(
            source, template, matched_regions, strict=strict
        )

        # Generate result image
        result_img = draw_match_with_differences(source, template, report)

        return jsonify({
            "ok": True,
            "match_count": report.match_count,
            "is_genuine": is_valid,
            "confidence": round(conf, 4),
            "reasons": reasons,
            "result_preview": _img_to_b64(result_img, max_dim=900),
            "elapsed_ms": round(report.elapsed_ms, 1),
        })

    @app.route("/api/document-forensics", methods=["POST"])
    def document_forensics():
        """
        Run 5-method document forensics analysis.
        Compares the uploaded source (original) vs template (suspect).
        Detects: ink additions, character insertions, spacing changes, erasures.
        Returns RED-box annotations on all differences.
        """
        sid = session.get("session_id")
        original = _get_image(sid, "source")
        suspect = _get_image(sid, "template")

        if original is None or suspect is None:
            return jsonify({"error": "Upload both original and suspect document images"}), 400

        data = request.get_json() or {}

        # Run document forensics
        engine = DocumentForensicsEngine(
            pixel_diff_threshold=float(data.get("pixel_diff_threshold", 25.0)),
            min_region_area=int(data.get("min_region_area", 50)),
            morph_close_kernel=int(data.get("morph_close_kernel", 5)),
        )
        report = engine.analyze(original, suspect, align=True)

        # Convert images to base64
        annotated_b64 = _img_to_b64(report.annotated_image, max_dim=900) if report.annotated_image is not None else ""
        canvas_b64 = _img_to_b64(report.forensics_canvas, max_dim=1400) if report.forensics_canvas is not None else ""
        mask_b64 = _img_to_b64(report.difference_mask, max_dim=900) if report.difference_mask is not None else ""

        heatmap_vis = None
        if report.tamper_heatmap is not None:
            heatmap_vis = cv2.applyColorMap(
                (report.tamper_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
        heatmap_b64 = _img_to_b64(heatmap_vis, max_dim=900) if heatmap_vis is not None else ""

        # Build regions JSON
        regions_json = []
        for i, r in enumerate(report.tamper_regions):
            regions_json.append({
                "id": i + 1,
                "x": r.x, "y": r.y,
                "width": r.width, "height": r.height,
                "area_px": r.area_px,
                "tamper_confidence": round(r.tamper_confidence, 4),
                "severity": r.severity.name,
                "severity_label": r.severity.label,
                "likely_addition": r.likely_addition,
                "likely_erasure": r.likely_erasure,
                "likely_alteration": r.likely_alteration,
                "description": r.description,
                "method_scores": {
                    k: round(v, 4) for k, v in r.method_scores.items()
                },
            })

        # Method results
        methods_json = {}
        for method_name, result in report.method_results.items():
            methods_json[method_name] = {
                "confidence": round(result.get("confidence", 0), 4),
                "density": round(result.get("density", 0), 4),
            }

        return jsonify({
            "ok": True,
            "is_tampered": report.is_tampered,
            "overall_severity": report.overall_severity.name,
            "overall_severity_label": report.overall_severity.label,
            "tamper_score": round(report.tamper_score, 4),
            "tamper_percentage": round(report.tamper_percentage, 2),
            "region_count": report.region_count,
            "elapsed_ms": round(report.elapsed_ms, 1),

            # Images
            "annotated_preview": annotated_b64,
            "canvas_preview": canvas_b64,
            "mask_preview": mask_b64,
            "heatmap_preview": heatmap_b64,

            # Data
            "regions": regions_json,
            "method_results": methods_json,
            "summary": report.summary_lines,

            "source_shape": list(report.source_shape),
            "template_shape": list(report.template_shape),
        })

    @app.route("/api/signature-verify", methods=["POST"])
    def signature_verify():
        """Verify a signature against original."""
        sid = session.get("session_id")
        original = _get_image(sid, "source")
        suspect = _get_image(sid, "template")

        if original is None or suspect is None:
            return jsonify({"error": "Upload original signature as Source and suspect as Template"}), 400

        verifier = SignatureVerifier()
        report = verifier.verify(original, suspect)

        return jsonify({
            "ok": True,
            "verdict": report.verdict.name,
            "verdict_label": report.verdict.label,
            "confidence": round(report.confidence, 4),
            "geometry_similarity": round(report.geometry_similarity, 4),
            "density_similarity": round(report.density_similarity, 4),
            "topology_similarity": round(report.topology_similarity, 4),
            "hu_moments_similarity": round(report.hu_moments_similarity, 4),
            "orb_match_score": round(report.orb_match_score, 4),
            "details": report.details,
            "annotated_preview": _img_to_b64(report.annotated_image, max_dim=600) if report.annotated_image is not None else "",
            "elapsed_ms": round(report.elapsed_ms, 1),
        })

    @app.route("/api/vault-register", methods=["POST"])
    def vault_register():
        """Register the current source image in the fingerprint vault."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")

        if source is None:
            return jsonify({"error": "Upload source image first"}), 400

        data = request.get_json() or {}
        label = data.get("label", f"Document-{sid[:8]}")

        vault = get_vault()
        fp = vault.register(source, label=label, tags=data.get("tags", ""))

        return jsonify({
            "ok": True,
            "fingerprint_id": fp.fingerprint_id,
            "label": fp.label,
            "phash": fp.phash,
            "sha256": fp.sha256[:16] + "...",
            "size": f"{fp.width}x{fp.height}",
        })

    @app.route("/api/vault-search", methods=["POST"])
    def vault_search():
        """Search vault for matching fingerprint."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")

        if source is None:
            return jsonify({"error": "Upload source image first"}), 400

        vault = get_vault()
        results = vault.search(source, max_results=5)

        return jsonify({
            "ok": True,
            "results": [
                {
                    "id": r.fingerprint.fingerprint_id,
                    "label": r.fingerprint.label,
                    "similarity": round(r.similarity_score, 4),
                    "is_match": r.is_match,
                    "phash_distance": r.phash_distance,
                }
                for r in results
            ],
            "total_in_vault": vault.count(),
        })

    @app.route("/api/vault-list", methods=["GET"])
    def vault_list():
        """List all fingerprints in vault."""
        vault = get_vault()
        fingerprints = vault.list_all()
        return jsonify({
            "ok": True,
            "count": len(fingerprints),
            "fingerprints": [
                {
                    "id": fp.fingerprint_id,
                    "label": fp.label,
                    "size": f"{fp.width}x{fp.height}",
                    "stored_at": fp.stored_at,
                    "phash": fp.phash[:8] + "...",
                }
                for fp in fingerprints
            ],
        })

    @app.route("/api/metadata", methods=["POST"])
    def analyze_metadata():
        """Analyze metadata of an uploaded file."""
        # This needs the actual file path, so we save temp
        sid = session.get("session_id")
        source = _get_image(sid, "source")

        if source is None:
            return jsonify({"error": "Upload an image first"}), 400

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, source)
            tmp_path = tmp.name

        analyzer = MetadataAnalyzer()
        report = analyzer.analyze(tmp_path)

        os.unlink(tmp_path)

        return jsonify({
            "ok": True,
            "file_name": report.file_name,
            "file_size_bytes": report.file_size_bytes,
            "mime_type": report.mime_type,
            "has_exif": report.has_exif,
            "camera_make": report.camera_make,
            "camera_model": report.camera_model,
            "software_used": report.software_used,
            "date_original": report.date_original,
            "has_gps": report.has_gps,
            "gps_latitude": report.gps_latitude,
            "gps_longitude": report.gps_longitude,
            "image_width": report.image_width,
            "image_height": report.image_height,
            "anomalies": [a.name for a in report.anomalies],
            "anomaly_score": round(report.anomaly_score, 4),
            "is_suspicious": report.is_suspicious,
            "details": report.details,
        })

    @app.route("/api/generate-report", methods=["POST"])
    def generate_report():
        """Generate comprehensive forensic HTML report with ALL analysis results."""
        sid = session.get("session_id")
        original = _get_image(sid, "source")
        suspect = _get_image(sid, "template")

        if original is None or suspect is None:
            return jsonify({"error": "Upload both images first"}), 400

        data = request.get_json() or {}

        # Run ALL analyses
        f_report = DocumentForensicsEngine().analyze(original, suspect)
        fg = ForgeryDetector()
        regions = []
        if f_report and hasattr(f_report, 'tamper_regions'):
            regions = [(r.x, r.y, r.width, r.height) for r in f_report.tamper_regions]
        forgery_report = fg.analyze(original, suspect, regions)
        cm_report = CopyMoveDetector().detect(suspect)
        font_report = FontAnalyzer().analyze(suspect)
        sig_report = SignatureVerifier().verify(original, suspect)
        # Save suspect to temp file for metadata analysis
        import tempfile
        meta_report = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, suspect)
                meta_report = MetadataAnalyzer().analyze(tmp.name)
            os.unlink(tmp.name)
        except Exception: pass
        try: adv_report = AdvancedDetectors().analyze(original, suspect)
        except Exception: adv_report = None
        scorer = AuthenticityScorer()
        vault = get_vault()
        fp_res = vault.search(suspect, max_results=1)
        fp_match = {"matched": False, "best_label": "", "similarity": 0}
        if fp_res and fp_res[0].is_match:
            r = fp_res[0]
            fp_match = {"matched": True, "best_label": r.fingerprint.label,
                        "similarity": r.similarity_score, "phash_dist": r.phash_distance}
        das = scorer.compute(forensics_report=f_report, forgery_report=forgery_report,
                            font_report=font_report, copy_move_report=cm_report,
                            signature_report=sig_report, metadata_report=meta_report,
                            fingerprint_match=fp_match)

        # Build embedded images — include ALL available visuals
        embedded = {}
        # Original & suspect thumbnails
        thumb_src = cv2.resize(original, (300, int(300*original.shape[0]/original.shape[1])))
        _, buf = cv2.imencode('.png', thumb_src)
        embedded['original'] = base64.b64encode(buf).decode()
        thumb_sus = cv2.resize(suspect, (300, int(300*suspect.shape[0]/suspect.shape[1])))
        _, buf = cv2.imencode('.png', thumb_sus)
        embedded['suspect'] = base64.b64encode(buf).decode()
        # Forensics
        if f_report.annotated_image is not None:
            _, buf = cv2.imencode('.png', f_report.annotated_image)
            embedded['Forensics_Annotated'] = base64.b64encode(buf).decode()
        if hasattr(f_report, 'forensics_canvas') and f_report.forensics_canvas is not None:
            _, buf = cv2.imencode('.png', f_report.forensics_canvas)
            embedded['5-Method_Canvas'] = base64.b64encode(buf).decode()
        if f_report.difference_mask is not None:
            _, buf = cv2.imencode('.png', f_report.difference_mask)
            embedded['Difference_Mask'] = base64.b64encode(buf).decode()
        # Copy-Move
        if cm_report.annotated_image is not None:
            _, buf = cv2.imencode('.png', cm_report.annotated_image)
            embedded['CopyMove_Detection'] = base64.b64encode(buf).decode()
        # Font
        if font_report.annotated_image is not None:
            _, buf = cv2.imencode('.png', font_report.annotated_image)
            embedded['Font_Analysis'] = base64.b64encode(buf).decode()
        # Signature
        if sig_report.annotated_image is not None:
            _, buf = cv2.imencode('.png', sig_report.annotated_image)
            embedded['Signature_Verification'] = base64.b64encode(buf).decode()
        # Advanced Detectors Heatmap
        if adv_report is not None and adv_report.heatmap is not None and adv_report.heatmap.size > 0:
            try:
                hm = np.nan_to_num(adv_report.heatmap, nan=0.0)
                hm_vis = cv2.applyColorMap((np.clip(hm,0,1)*255).astype(np.uint8), cv2.COLORMAP_HOT)
                _, buf = cv2.imencode('.png', hm_vis)
                embedded['Advanced_Heatmap'] = base64.b64encode(buf).decode()
            except Exception: pass
        # Forgery ELA
        if forgery_report is not None and forgery_report.ela_image is not None:
            _, buf = cv2.imencode('.png', forgery_report.ela_image)
            embedded['ELA_Analysis'] = base64.b64encode(buf).decode()

        reporter = ForensicReporter(
            case_id=data.get("case_id", ""),
            examiner=data.get("examiner", "AI Forensic System"),
        )
        html = reporter.generate(
            forensics_report=f_report,
            forgery_report=forgery_report,
            fingerprint_result=fp_match,
            signature_report=sig_report,
            metadata_report=meta_report,
            copy_move_report=cm_report,
            font_report=font_report,
            advanced_report=adv_report,
            das_report=das,
            original_path="original",
            suspect_path="suspect",
            embedded_images=embedded,
        )

        return jsonify({
            "ok": True,
            "html_report": html,
            "tamper_score": round(f_report.tamper_score, 4),
            "severity": f_report.overall_severity.label,
            "region_count": f_report.region_count,
            "das_score": das.overall_score,
            "das_verdict": das.verdict,
        })

    @app.route("/api/advanced-detect", methods=["POST"])
    def advanced_detect():
        """Run 5 advanced forensic detectors."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")
        template = _get_image(sid, "template")
        if source is None:
            return jsonify({"error": "Upload source image first"}), 400
        try:
            engine = AdvancedDetectors()
            report = engine.analyze(source, template)
            heatmap_b64 = ""
            if report.heatmap is not None and report.heatmap.size > 0:
                try:
                    hm = np.nan_to_num(report.heatmap, nan=0.0, posinf=1.0, neginf=0.0)
                    hm_uint8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
                    heatmap_vis = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_HOT)
                    heatmap_b64 = _img_to_b64(heatmap_vis, max_dim=900)
                except Exception: heatmap_b64 = ""
            return jsonify({
                "ok": True,
                "verdict": str(report.verdict),
                "overall_risk": round(float(report.overall_risk), 4),
                "is_suspicious": bool(report.is_suspicious),
                "jpeg_ghost_score": round(float(report.jpeg_ghost_score), 4),
                "jpeg_ghost_detected": bool(report.jpeg_ghost_detected),
                "jpeg_ghost_details": str(report.jpeg_ghost_details),
                "double_jpeg_score": round(float(report.double_jpeg_score), 4),
                "double_jpeg_detected": bool(report.double_jpeg_detected),
                "double_jpeg_details": str(report.double_jpeg_details),
                "luminance_score": round(float(report.luminance_score), 4),
                "luminance_anomaly": bool(report.luminance_anomaly),
                "luminance_details": str(report.luminance_details),
                "cfa_score": round(float(report.cfa_score), 4),
                "cfa_anomaly": bool(report.cfa_anomaly),
                "cfa_details": str(report.cfa_details),
                "prnu_available": bool(report.prnu_available),
                "prnu_similarity": round(float(report.prnu_similarity), 4) if report.prnu_available else 0,
                "prnu_details": str(report.prnu_details),
                "details": [str(d) for d in report.details],
                "heatmap_preview": heatmap_b64,
                "elapsed_ms": round(report.elapsed_ms, 1),
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/copy-move", methods=["POST"])
    def detect_copy_move():
        """Detect copy-move forgery in the source document."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")
        if source is None:
            return jsonify({"error": "Upload source image first"}), 400
        detector = CopyMoveDetector()
        report = detector.detect(source)
        heatmap_b64 = ""
        if report.heatmap is not None:
            heatmap_vis = cv2.applyColorMap((report.heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_b64 = _img_to_b64(heatmap_vis, max_dim=900)
        return jsonify({
            "ok": True,
            "has_clones": report.has_clones,
            "clone_count": report.clone_count,
            "confidence": round(report.confidence, 4),
            "details": report.details,
            "annotated_preview": _img_to_b64(report.annotated_image, max_dim=900) if report.annotated_image is not None else "",
            "heatmap_preview": heatmap_b64,
            "elapsed_ms": round(report.elapsed_ms, 1),
        })

    @app.route("/api/font-analysis", methods=["POST"])
    def font_analysis():
        """Analyze font/text consistency in the source document."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")
        if source is None:
            return jsonify({"error": "Upload source image first"}), 400
        analyzer = FontAnalyzer()
        report = analyzer.analyze(source)
        regions_json = []
        for i, r in enumerate(report.inconsistent_regions):
            regions_json.append({
                "id": i+1, "x": r.x, "y": r.y, "width": r.width, "height": r.height,
                "estimated_height": round(r.estimated_height, 1),
                "estimated_weight": round(r.estimated_weight, 2),
                "consistency_score": round(r.consistency_score, 4),
            })
        return jsonify({
            "ok": True,
            "is_consistent": report.is_consistent,
            "inconsistent_count": len(report.inconsistent_regions),
            "avg_height": round(report.avg_height, 1),
            "avg_weight": round(report.avg_weight, 2),
            "height_variation": round(report.height_variation, 4),
            "weight_variation": round(report.weight_variation, 4),
            "confidence": round(report.confidence, 4),
            "details": report.details,
            "regions": regions_json,
            "annotated_preview": _img_to_b64(report.annotated_image, max_dim=900) if report.annotated_image is not None else "",
            "elapsed_ms": round(report.elapsed_ms, 1),
        })

    @app.route("/api/authenticity-score", methods=["POST"])
    def authenticity_score():
        """Compute combined document authenticity score from ALL analyses."""
        sid = session.get("session_id")
        source = _get_image(sid, "source")
        template = _get_image(sid, "template")
        if source is None:
            return jsonify({"error": "Upload at least the source image"}), 400
        scorer = AuthenticityScorer()
        f_report = None
        if template is not None:
            try: f_report = DocumentForensicsEngine().analyze(source, template)
            except Exception: pass
        forgery_report = None
        if template is not None:
            try:
                fg = ForgeryDetector()
                regions = []
                if f_report and hasattr(f_report, 'tamper_regions'):
                    regions = [(r.x, r.y, r.width, r.height) for r in f_report.tamper_regions]
                forgery_report = fg.analyze(source, template, regions)
            except Exception: pass
        cm_report = CopyMoveDetector().detect(source)
        font_report = FontAnalyzer().analyze(source)
        vault = get_vault()
        fp_res = vault.search(source if template is None else template, max_results=1)
        fp_match = None
        if fp_res and fp_res[0].is_match:
            fp_match = {"matched": True, "similarity": fp_res[0].similarity_score}
        result = scorer.compute(
            forensics_report=f_report,
            forgery_report=forgery_report,
            font_report=font_report,
            copy_move_report=cm_report,
            fingerprint_match=fp_match,
        )
        return jsonify({
            "ok": True,
            "overall_score": result.overall_score,
            "verdict": result.verdict,
            "category_scores": result.category_scores,
            "weighted_breakdown": result.weighted_breakdown,
            "risk_factors": result.risk_factors,
            "recommendations": result.recommendations,
        })

    @app.route("/api/download-result", methods=["GET"])
    def download_result():
        """Download the last result image as PNG."""
        # Result is regenerated on demand from session
        sid = session.get("session_id")
        source = _get_image(sid, "source")
        template = _get_image(sid, "template")
        if source is None:
            return jsonify({"error": "No session data"}), 404

        # Quick re-run to get result
        matcher = TemplateMatcher(validate_matches=True)
        report = matcher.match(source, template)
        result_img = draw_match_with_differences(source, template, report)

        _, buf = cv2.imencode(".png", result_img)
        return send_file(
            io.BytesIO(buf),
            mimetype="image/png",
            as_attachment=True,
            download_name="template_matching_result.png",
        )

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_image(file) -> np.ndarray:
    """Read an uploaded file into an OpenCV image (numpy array)."""
    raw = file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _img_to_b64(img: np.ndarray, max_dim: int = 500) -> str:
    """Convert an OpenCV image to a base64 PNG data-URI, resizing if needed."""
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode()
    return f"data:image/png;base64,{b64}"


def run_web(host: str = None, port: int = None, debug: bool = False):
    """Launch the Flask web server.
    
    Uses PORT env var for Render compatibility (default: 5000 locally, 10000 on Render).
    Binds to 0.0.0.0 on Render, 127.0.0.1 locally.
    """
    app = create_app()
    
    # Render sets PORT env var — always prefer it over defaults
    env_port = os.environ.get("PORT")
    if env_port:
        host = "0.0.0.0"
        port = int(env_port)
    else:
        if host is None:
            host = "127.0.0.1"
        if port is None:
            port = 5000
    
    print(f"\n🌐 Template Matching Web App")
    print(f"   Open in browser: http://{host}:{port}")
    print(f"   Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=debug)
