#!/usr/bin/env python3
"""
Template Matching — Main Entry Point
=====================================

Usage:
    python main.py gui              Launch the GUI application
    python main.py web              Launch the Web interface
    python main.py match <source> <template> [--method METHOD] [--threshold THRESHOLD] [--multi-scale]
    python main.py compare <source> <template>   Compare all matching methods
"""

import argparse
import sys
from pathlib import Path

from src.template_matcher import (
    TemplateMatcher, MatchMethod, DetectionReport,
    draw_matches, draw_match_comparison,
)
from src.multi_scale_matcher import MultiScaleMatcher
from src.utils import load_image, save_image, resize_to_max_dim


# ---------------------------------------------------------------------------
# CLI: single match
# ---------------------------------------------------------------------------

def cmd_match(args):
    """Run single template matching and save the result."""
    source = load_image(args.source)
    template = load_image(args.template)

    method = MatchMethod[args.method] if args.method else MatchMethod.TM_CCOEFF_NORMED

    if args.multi_scale:
        matcher = MultiScaleMatcher(
            method=method, threshold=args.threshold,
            scale_range=tuple(args.scale_range), scale_steps=args.scale_steps,
        )
    else:
        matcher = TemplateMatcher(method=method, threshold=args.threshold)

    report = matcher.match(source, template)

    result_img = draw_matches(source, report)

    out_path = args.output or "result.png"
    save_image(result_img, out_path)

    print(f"✅ {report.match_count} match(es) found in {report.elapsed_ms:.1f} ms")
    print(f"   Method:  {report.method.label}")
    print(f"   Output:  {out_path}")
    for i, m in enumerate(report.matches):
        print(f"   [{i+1}] x={m.x}, y={m.y}, w={m.width}, h={m.height}, "
              f"confidence={m.confidence:.4f}, scale={m.scale:.2f}")


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
    p_match = sub.add_parser("match", help="Run template matching (single method)")
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
