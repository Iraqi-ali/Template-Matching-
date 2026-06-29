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
    draw_matches, draw_match_comparison,
)
from .multi_scale_matcher import MultiScaleMatcher

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

        # Draw result
        result_img = draw_matches(source, report)

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
            "source_shape": list(report.source_shape),
            "template_shape": list(report.template_shape),
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
        matcher = TemplateMatcher()
        report = matcher.match(source, template)
        result_img = draw_matches(source, report)

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
