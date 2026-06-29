#!/usr/bin/env python3
"""
Template Matching & Forgery Detection — Main Entry Point
=========================================================

Usage:
    python main.py gui              Launch the GUI application
    python main.py web              Launch the Web interface
    python main.py match <source> <template> [--method METHOD] [--threshold THRESHOLD] [--multi-scale]
    python main.py compare <source> <template>   Compare all matching methods
    python main.py detect <source> <template>    Forgery detection & difference analysis
    python main.py validate <source> <template>  Quick validation only
"""

import argparse
import sys
from pathlib import Path

from src.template_matcher import (
    TemplateMatcher, MatchMethod, DetectionReport,
    draw_matches, draw_match_comparison, draw_match_with_differences,
)
from src.multi_scale_matcher import MultiScaleMatcher
from src.forgery_detector import (
    ForgeryDetector, draw_difference_boxes, draw_forgery_report,
    create_full_analysis_canvas,
)
from src.document_forensics import (
    DocumentForensicsEngine, DocumentForensicsReport, TamperSeverity,
    run_document_forensics,
)
from src.document_fingerprint import (
    FingerprintVault, get_vault, compute_phash, compute_ahash,
    compute_dhash, hamming_distance,
)
from src.metadata_analyzer import MetadataAnalyzer, MetadataReport
from src.signature_verifier import SignatureVerifier, SignatureReport, SignatureVerdict
from src.forensic_reporter import (
    ForensicReporter, generate_forensic_charts, generate_histogram_comparison,
)
from src.utils import load_image, save_image, resize_to_max_dim


# ---------------------------------------------------------------------------
# CLI: single match
# ---------------------------------------------------------------------------

def cmd_match(args):
    """Run single template matching with validation and save the result."""
    source = load_image(args.source)
    template = load_image(args.template)

    method = MatchMethod[args.method] if args.method else MatchMethod.TM_CCOEFF_NORMED

    if args.multi_scale:
        matcher = MultiScaleMatcher(
            method=method, threshold=args.threshold,
            scale_range=tuple(args.scale_range), scale_steps=args.scale_steps,
        )
    else:
        matcher = TemplateMatcher(
            method=method, threshold=args.threshold,
            validate_matches=True, strict_validation=True,
        )

    report = matcher.match(source, template)

    # Generate result with difference boxes
    result_img = draw_match_with_differences(source, template, report)

    out_path = args.output or "result.png"
    save_image(result_img, out_path)

    # Display results
    if report.match_count == 0:
        print(f"❌ No matches found (threshold={report.threshold:.2f})")
        print(f"   Method:  {report.method.label}")
        if report.validation is not None:
            print(f"   Validation: {report.validation.reasons}")
        return

    validation_status = "⚠ UNVERIFIED"
    if report.validation is not None:
        validation_status = "✓ GENUINE" if report.validation.is_valid else "⚠ UNVERIFIED"

    print(f"✅ {report.match_count} match(es) found [{validation_status}] in {report.elapsed_ms:.1f} ms")
    print(f"   Method:  {report.method.label}")
    print(f"   Output:  {out_path}")
    for i, m in enumerate(report.matches):
        print(f"   [{i+1}] x={m.x}, y={m.y}, w={m.width}, h={m.height}, "
              f"confidence={m.confidence:.4f}, scale={m.scale:.2f}")

    if report.validation is not None:
        print(f"\n📋 Validation Report:")
        for reason in report.validation.reasons:
            print(f"   {reason}")


# ---------------------------------------------------------------------------
# CLI: compare all methods
# ---------------------------------------------------------------------------

def cmd_compare(args):
    """Run matching with all methods and display a comparison grid."""
    source = load_image(args.source)
    template = load_image(args.template)

    matcher = TemplateMatcher(threshold=args.threshold)
    reports = matcher.match_all_methods(source, template, threshold=args.threshold)

    comparison_img = draw_match_comparison(source, reports)

    out_path = args.output or "comparison.png"
    save_image(comparison_img, out_path)

    print(f"✅ Comparison saved to: {out_path}")
    for r in reports:
        print(f"   {r.method.label:<45s} → {r.match_count} match(es)  ({r.elapsed_ms:.1f} ms)")


# ---------------------------------------------------------------------------
# CLI: GUI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI: forgery detection
# ---------------------------------------------------------------------------

def cmd_detect(args):
    """Run comprehensive forgery/tampering detection."""
    source = load_image(args.source)
    template = load_image(args.template)

    print("🔍 Running comprehensive forgery analysis...")
    print(f"   Source:   {args.source} ({source.shape[1]}×{source.shape[0]})")
    print(f"   Template: {args.template} ({template.shape[1]}×{template.shape[0]})")

    # Step 1: Matching with validation
    matcher = TemplateMatcher(
        method=MatchMethod.TM_CCOEFF_NORMED,
        threshold=args.threshold,
        validate_matches=True,
        strict_validation=True,
    )
    report = matcher.match(source, template)
    matched_regions = [(m.x, m.y, m.width, m.height) for m in report.matches]

    print(f"\n📊 Matching Results:")
    print(f"   Matches found: {report.match_count}")
    if report.validation:
        print(f"   Validated:     {'✓ YES' if report.validation.is_valid else '⚠ NO'}")
        for reason in report.validation.reasons:
            print(f"     • {reason}")

    # Step 2: Forgery analysis
    print(f"\n🔬 Forgery Analysis:")
    detector = ForgeryDetector()
    forgery_report = detector.analyze(source, template, matched_regions)

    print(f"   Risk Level:  {forgery_report.risk_level.label}")
    print(f"   Risk Score:  {forgery_report.risk_score:.3f}")
    print(f"   SSIM:        {forgery_report.ssim_score:.4f}")
    print(f"   ELA Score:   {forgery_report.ela_score:.4f}")
    print(f"   Noise Cons:  {forgery_report.noise_consistency:.4f}")
    print(f"   Hist Correlation: {forgery_report.histogram_correlation:.4f}")
    print(f"   Edge Anomaly:     {forgery_report.edge_anomaly_score:.4f}")
    print(f"   Diff Regions:     {len(forgery_report.diff_regions)}")
    print(f"   Analysis Time:    {forgery_report.elapsed_ms:.1f} ms")

    print(f"\n📋 Details:")
    for detail in forgery_report.details:
        print(f"   {detail}")

    # Step 3: Quick validation
    if matched_regions:
        is_valid, conf, reasons = detector.quick_validation(
            source, template, matched_regions,
        )
        print(f"\n✅ Quick Validation: {'PASSED' if is_valid else 'FAILED'} (confidence: {conf:.4f})")
        for reason in reasons:
            print(f"   {reason}")

    # Generate visualizations
    result_img = draw_match_with_differences(source, template, report)
    out_path = args.output or "forgery_detection_result.png"
    save_image(result_img, out_path)
    print(f"\n🖼️  Result image saved: {out_path}")

    # Generate full analysis canvas
    analysis_canvas = create_full_analysis_canvas(
        source, template, forgery_report, matched_regions,
    )
    analysis_path = args.analysis_output or "forgery_analysis.png"
    save_image(analysis_canvas, analysis_path)
    print(f"🖼️  Analysis canvas saved: {analysis_path}")


def cmd_validate(args):
    """Quick validation only — is this match genuine?"""
    source = load_image(args.source)
    template = load_image(args.template)

    print("🔍 Running quick validation...")

    matcher = TemplateMatcher(
        threshold=args.threshold,
        validate_matches=True,
        strict_validation=args.strict,
    )
    report = matcher.match(source, template)

    matched_regions = [(m.x, m.y, m.width, m.height) for m in report.matches]

    if not matched_regions:
        print("❌ No matches found — cannot validate.")
        return

    detector = ForgeryDetector()
    is_valid, conf, reasons = detector.quick_validation(
        source, template, matched_regions, strict=args.strict,
    )

    print(f"\n{'✅ GENUINE MATCH' if is_valid else '❌ NOT GENUINE / DIFFERENT IMAGES'}")
    print(f"   Confidence: {conf:.4f}")
    print(f"   Reasons:")
    for reason in reasons:
        print(f"     • {reason}")

    if report.validation:
        print(f"\n📊 Additional Metrics:")
        print(f"   SSIM: {report.validation.ssim_score:.4f}")
        print(f"   Histogram Correlation: {report.validation.histogram_correlation:.4f}")
        print(f"   Edge Overlap: {report.validation.edge_overlap:.4f}")
        print(f"   Pixel Similarity: {report.validation.pixel_similarity:.4f}")

    # Save result
    result_img = draw_match_with_differences(source, template, report)
    out_path = args.output or "validation_result.png"
    save_image(result_img, out_path)
    print(f"\n🖼️  Result saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI: Document Forensics (5 methods)
# ---------------------------------------------------------------------------

def cmd_forensics(args):
    """Run 5-method document forensics analysis."""
    print("🔬 DOCUMENT FORENSICS ANALYSIS")
    print("=" * 60)
    print(f"   Original: {args.original}")
    print(f"   Suspect:  {args.suspect}")
    print()

    # Run analysis
    report = run_document_forensics(
        original_path=args.original,
        suspect_path=args.suspect,
        output_dir=args.output_dir,
    )

    # Print summary
    for line in report.summary_lines:
        print(f"   {line}")

    # Print detailed method results
    print(f"\n📊 DETAILED METHOD RESULTS:")
    print(f"   {'Method':<25s} {'Confidence':>10s} {'Density':>10s}")
    print(f"   {'-'*45}")
    for method_name, result in report.method_results.items():
        conf = result.get("confidence", 0)
        density = result.get("density", 0)
        print(f"   {method_name:<25s} {conf:>10.4f} {density:>10.4f}")

    if report.tamper_regions:
        print(f"\n🔴 DETECTED TAMPER REGIONS ({report.region_count}):")
        for i, r in enumerate(report.tamper_regions):
            print(f"   [{i+1}] ({r.x},{r.y}) {r.width}×{r.height}px "
                  f"area={r.area_px}px² conf={r.tamper_confidence:.1%}")
            if r.description:
                print(f"       → {r.description}")

    # Print output files
    import os
    print(f"\n📁 Output files saved to: {os.path.abspath(args.output_dir)}/")
    print(f"   • forensics_annotated.png  — Suspect image with RED difference boxes")
    print(f"   • forensics_canvas.png     — 6-panel forensic evidence canvas")
    print(f"   • forensics_mask.png       — Binary difference mask")
    print(f"   • forensics_heatmap.png    — Tampering probability heatmap")
    print(f"   • forensics_report.txt     — Full text report")


# ---------------------------------------------------------------------------
# CLI: Fingerprint Vault
# ---------------------------------------------------------------------------

def cmd_vault(args):
    """Manage the document fingerprint vault."""
    vault = get_vault(args.db)

    if args.action == "register":
        img = load_image(args.image)
        fp = vault.register(img, label=args.label, file_path=args.image, tags=args.tags or "")
        print(f"✅ Fingerprint registered!")
        print(f"   ID:       {fp.fingerprint_id}")
        print(f"   Label:    {fp.label}")
        print(f"   pHash:    {fp.phash}")
        print(f"   SHA-256:  {fp.sha256[:32]}...")
        print(f"   Size:     {fp.width}×{fp.height}")

    elif args.action == "search":
        img = load_image(args.image)
        results = vault.search(img, max_results=args.top)
        print(f"\n🔍 Search results ({len(results)} found):")
        for i, r in enumerate(results):
            icon = "✅" if r.is_match else "❌"
            print(f"   [{i+1}] {icon} {r.fingerprint.label}")
            print(f"        Similarity: {r.similarity_score:.2%}")
            print(f"        pHash dist: {r.phash_distance}")

    elif args.action == "verify":
        img = load_image(args.image)
        result = vault.verify(img, args.id)
        print(f"\n🔍 Verification: {'✅ MATCH' if result.is_match else '❌ NO MATCH'}")
        print(f"   Label: {result.fingerprint.label}")
        print(f"   Similarity: {result.similarity_score:.2%}")

    elif args.action == "list":
        fps = vault.list_all()
        print(f"\n📋 Vault ({len(fps)} fingerprints):")
        for fp in fps:
            print(f"   [{fp.fingerprint_id[:8]}...] {fp.label:30s} {fp.width}×{fp.height}")

    elif args.action == "delete":
        vault.delete(args.id)
        print(f"✅ Deleted: {args.id}")

    elif args.action == "stats":
        print(f"\n📊 Total: {vault.count()} fingerprints")
        for entry in vault.get_audit_log(5):
            print(f"   [{entry['timestamp']}] {entry['action']}: {entry['details'][:50]}")


# ---------------------------------------------------------------------------
# CLI: Signature Verification
# ---------------------------------------------------------------------------

def cmd_signature(args):
    """Verify a signature against an original."""
    original = load_image(args.original)
    suspect = load_image(args.suspect)
    verifier = SignatureVerifier()
    report = verifier.verify(original, suspect)

    print(f"\n✍️ SIGNATURE: {report.verdict.label} ({report.confidence:.1%})")
    print(f"   Geometry={report.geometry_similarity:.3f} Density={report.density_similarity:.3f}")
    print(f"   Topology={report.topology_similarity:.3f} Hu={report.hu_moments_similarity:.3f}")
    print(f"   ORB={report.orb_match_score:.3f} Time={report.elapsed_ms:.1f}ms")
    for d in report.details:
        print(f"   {d}")
    if report.annotated_image is not None:
        save_image(report.annotated_image, args.output or "signature_result.png")
        print(f"   🖼️ Saved: {args.output or 'signature_result.png'}")


# ---------------------------------------------------------------------------
# CLI: Metadata Analysis
# ---------------------------------------------------------------------------

def cmd_metadata(args):
    """Analyze file metadata for forgery indicators."""
    analyzer = MetadataAnalyzer()
    report = analyzer.analyze(args.file)
    for line in report.details:
        print(f"   {line}")
    if args.compare:
        comp = analyzer.compare_metadata(args.file, args.compare)
        print(f"\n   Compare: {'✅ Consistent' if comp['consistent'] else '❌ Different'}")
        for d in comp['differences']:
            print(f"   • {d}")


# ---------------------------------------------------------------------------
# CLI: Full Forensic Report (HTML)
# ---------------------------------------------------------------------------

def cmd_report(args):
    """Generate comprehensive professional forensic HTML report."""
    import os, base64 as b64
    print("📝 Generating comprehensive forensic report...\n")

    original = load_image(args.original)
    suspect = load_image(args.suspect)

    # 1. Forensics
    print("[1/4] 5-method document forensics...")
    f_report = DocumentForensicsEngine().analyze(original, suspect)

    # 2. Fingerprint
    print("[2/4] Fingerprint vault search...")
    vault = get_vault(args.vault_db)
    fp_res = vault.search(suspect, max_results=1)
    fp_match = {"matched": False, "best_label": "", "similarity": 0}
    if fp_res and fp_res[0].is_match:
        r = fp_res[0]
        fp_match = {"matched": True, "best_label": r.fingerprint.label,
                    "similarity": r.similarity_score, "phash_dist": r.phash_distance,
                    "dhash_dist": r.dhash_distance}

    # 3. Metadata
    print("[3/4] Metadata analysis...")
    meta_report = MetadataAnalyzer().analyze(args.suspect)

    # 4. Build report
    print("[4/4] Generating HTML report + charts...")
    method_confs = {}
    if hasattr(f_report, 'method_results'):
        method_confs = {k: v.get('confidence', 0) for k, v in f_report.method_results.items()}
    if method_confs:
        generate_forensic_charts(method_confs, output_dir=args.output_dir)
    generate_histogram_comparison(original, suspect, output_dir=args.output_dir)

    embedded = {}
    if f_report.annotated_image is not None:
        _, buf = cv2.imencode('.png', f_report.annotated_image)
        embedded['annotated'] = b64.b64encode(buf).decode()
    if hasattr(f_report, 'forensics_canvas') and f_report.forensics_canvas is not None:
        _, buf = cv2.imencode('.png', f_report.forensics_canvas)
        embedded['canvas'] = b64.b64encode(buf).decode()

    reporter = ForensicReporter(case_id=args.case_id or "", examiner=args.examiner or "AI Forensic System")
    html = reporter.generate(
        forensics_report=f_report, fingerprint_result=fp_match,
        metadata_report=meta_report, original_path=args.original,
        suspect_path=args.suspect, embedded_images=embedded,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "forensic_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Report: {report_path}")
    print(f"   📊 Charts: {args.output_dir}/")
    print(f"   🔗 Open in browser!")


# ---------------------------------------------------------------------------
# CLI: GUI
# ---------------------------------------------------------------------------

def cmd_gui(_args):
    """Launch the interactive GUI."""
    from src.gui_app import run_gui
    run_gui()


# ---------------------------------------------------------------------------
# CLI: Web
# ---------------------------------------------------------------------------

def cmd_web(args):
    """Launch the web interface."""
    from src.web_app import run_web
    run_web(host=args.host, port=args.port, debug=args.debug)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Template Matching — Find a template image within a source image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py gui
  python main.py web
  python main.py web --port 8080
  python main.py match source.jpg template.png --threshold 0.75
  python main.py match source.jpg template.png --multi-scale --scale-range 0.5 2.0
  python main.py compare source.jpg template.png --output comparison.png
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- gui ----
    sub.add_parser("gui", help="Launch the interactive GUI application")

    # ---- web ----
    p_web = sub.add_parser("web", help="Launch the web interface")
    p_web.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    p_web.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    p_web.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    p_web.set_defaults(func=cmd_web)

    # ---- match ----
    p_match = sub.add_parser("match", help="Run template matching with validation (single method)")
    p_match.add_argument("source", help="Path to the source image")
    p_match.add_argument("template", help="Path to the template image")
    p_match.add_argument("--method", default="TM_CCOEFF_NORMED",
                         choices=[m.name for m in MatchMethod],
                         help="Matching method (default: TM_CCOEFF_NORMED)")
    p_match.add_argument("--threshold", type=float, default=0.80,
                         help="Confidence threshold 0.0-1.0 (default: 0.80)")
    p_match.add_argument("--output", "-o", help="Output image path (default: result.png)")
    p_match.add_argument("--multi-scale", action="store_true",
                         help="Enable multi-scale matching")
    p_match.add_argument("--scale-range", type=float, nargs=2, default=[0.3, 2.5],
                         help="Scale range for multi-scale (default: 0.3 2.5)")
    p_match.add_argument("--scale-steps", type=int, default=25,
                         help="Number of scale steps (default: 25)")
    p_match.set_defaults(func=cmd_match)

    # ---- detect (forgery) ----
    p_detect = sub.add_parser("detect", help="Comprehensive forgery/tampering detection + difference analysis")
    p_detect.add_argument("source", help="Path to the source image")
    p_detect.add_argument("template", help="Path to the template image")
    p_detect.add_argument("--threshold", type=float, default=0.70,
                          help="Matching threshold (default: 0.70)")
    p_detect.add_argument("--output", "-o", help="Output image path")
    p_detect.add_argument("--analysis-output", help="Analysis canvas output path")
    p_detect.set_defaults(func=cmd_detect)

    # ---- validate ----
    p_val = sub.add_parser("validate", help="Quick validation — check if match is genuine or false")
    p_val.add_argument("source", help="Path to the source image")
    p_val.add_argument("template", help="Path to the template image")
    p_val.add_argument("--threshold", type=float, default=0.70,
                       help="Matching threshold (default: 0.70)")
    p_val.add_argument("--strict", action="store_true", default=True,
                       help="Use strict validation (default: True)")
    p_val.add_argument("--output", "-o", help="Output image path")
    p_val.set_defaults(func=cmd_validate)

    # ---- forensics (document) ----
    p_forensics = sub.add_parser(
        "forensics",
        help="Document forensics — 5 methods to detect tampering (ink, strokes, spacing, etc.)",
    )
    p_forensics.add_argument("original", help="Path to the ORIGINAL/genuine document image")
    p_forensics.add_argument("suspect", help="Path to the SUSPECT document image to analyze")
    p_forensics.add_argument("--output-dir", "-o", default="forensics_output",
                             help="Output directory for results (default: forensics_output/)")
    p_forensics.set_defaults(func=cmd_forensics)

    # ---- vault (fingerprint) ----
    p_vault = sub.add_parser("vault", help="Manage document fingerprint vault (register, search, verify)")
    p_vault_sub = p_vault.add_subparsers(dest="action", required=True)
    p_vault_reg = p_vault_sub.add_parser("register", help="Register a document fingerprint")
    p_vault_reg.add_argument("image", help="Path to the document image")
    p_vault_reg.add_argument("--label", "-l", required=True, help="Label for this document")
    p_vault_reg.add_argument("--tags", help="Comma-separated tags")
    p_vault_reg.add_argument("--db", default="fingerprint_vault.db", help="Vault DB path")
    p_vault_search = p_vault_sub.add_parser("search", help="Search vault for matching fingerprint")
    p_vault_search.add_argument("image", help="Path to the suspect image")
    p_vault_search.add_argument("--db", default="fingerprint_vault.db", help="Vault DB path")
    p_vault_search.add_argument("--top", type=int, default=5, help="Max results")
    p_vault_verify = p_vault_sub.add_parser("verify", help="Verify against specific fingerprint")
    p_vault_verify.add_argument("image", help="Path to the suspect image")
    p_vault_verify.add_argument("--id", required=True, help="Fingerprint ID to verify against")
    p_vault_verify.add_argument("--db", default="fingerprint_vault.db", help="Vault DB path")
    p_vault_list = p_vault_sub.add_parser("list", help="List all fingerprints in vault")
    p_vault_list.add_argument("--db", default="fingerprint_vault.db", help="Vault DB path")
    p_vault_del = p_vault_sub.add_parser("delete", help="Delete a fingerprint")
    p_vault_del.add_argument("--id", required=True, help="Fingerprint ID to delete")
    p_vault_del.add_argument("--db", default="fingerprint_vault.db", help="Vault DB path")
    p_vault_stats = p_vault_sub.add_parser("stats", help="Show vault statistics")
    p_vault_stats.add_argument("--db", default="fingerprint_vault.db", help="Vault DB path")
    p_vault.set_defaults(func=cmd_vault)

    # ---- signature ----
    p_sig = sub.add_parser("signature", help="Signature verification — detect forged signatures")
    p_sig.add_argument("original", help="Path to the original/genuine signature image")
    p_sig.add_argument("suspect", help="Path to the suspect signature image")
    p_sig.add_argument("--output", "-o", help="Output image path")
    p_sig.set_defaults(func=cmd_signature)

    # ---- metadata ----
    p_meta = sub.add_parser("metadata", help="Metadata & EXIF analysis for forgery detection")
    p_meta.add_argument("file", help="Path to the image file")
    p_meta.add_argument("--compare", help="Compare with another file's metadata")
    p_meta.set_defaults(func=cmd_metadata)

    # ---- report (full forensic) ----
    p_report = sub.add_parser("report", help="Generate comprehensive professional forensic HTML report")
    p_report.add_argument("original", help="Path to the ORIGINAL document")
    p_report.add_argument("suspect", help="Path to the SUSPECT document")
    p_report.add_argument("--output-dir", "-o", default="forensic_report",
                          help="Output directory (default: forensic_report/)")
    p_report.add_argument("--case-id", help="Case ID for the report")
    p_report.add_argument("--examiner", default="AI Forensic System", help="Examiner name")
    p_report.add_argument("--vault-db", default="fingerprint_vault.db", help="Vault DB path")
    p_report.set_defaults(func=cmd_report)

    # ---- compare ----
    p_comp = sub.add_parser("compare", help="Compare ALL matching methods side-by-side")
    p_comp.add_argument("source", help="Path to the source image")
    p_comp.add_argument("template", help="Path to the template image")
    p_comp.add_argument("--threshold", type=float, default=0.80,
                        help="Confidence threshold (default: 0.80)")
    p_comp.add_argument("--output", "-o", help="Output image path (default: comparison.png)")
    p_comp.set_defaults(func=cmd_compare)

    args = parser.parse_args()

    if args.command in ("gui", "web"):
        args.func(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
