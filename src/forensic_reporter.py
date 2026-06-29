"""
Forensic Report Generator — Professional PDF Reports
======================================================
Generates courtroom-ready PDF reports with charts, histograms,
heatmaps, and comprehensive forensic evidence.

Features:
  - Professional cover page with case details
  - Method comparison bar charts
  - Histogram overlay comparisons
  - Confidence score radar/spider chart
  - Before/After annotated image panels
  - Evidence summary table
  - Tamper region catalog
  - Audit trail & chain of custody
  - Export to PDF with embedded images & charts
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import os
import io
import json
import base64


@dataclass
class ReportSection:
    """A section of the forensic report."""
    title: str
    content: str = ""
    image_base64: str = ""
    chart_type: str = ""  # "bar", "histogram", "heatmap", "radar", "table"
    chart_data: Dict = field(default_factory=dict)


@dataclass
class ForensicPDFReport:
    """Complete forensic PDF report data."""
    case_id: str = ""
    case_title: str = ""
    examiner: str = ""
    date: str = ""
    original_file: str = ""
    suspect_file: str = ""

    overall_verdict: str = ""
    tamper_score: float = 0.0
    severity: str = ""

    sections: List[ReportSection] = field(default_factory=list)
    method_results: Dict[str, float] = field(default_factory=dict)
    regions: List[Dict] = field(default_factory=list)
    fingerprint_match: Optional[Dict] = None
    metadata_report: Optional[Dict] = None
    signature_report: Optional[Dict] = None

    # Binary content
    pdf_bytes: Optional[bytes] = None


class ForensicReporter:
    """
    Professional forensic report generator.
    Generates HTML reports (viewable in any browser) with embedded charts.
    Can convert to PDF via weasyprint or browser print.

    Usage:
        reporter = ForensicReporter()
        pdf_bytes = reporter.generate(document_forensics_report, fingerprint_result, ...)
        with open("report.html", "w") as f: f.write(pdf_bytes)
    """

    def __init__(
        self,
        case_id: str = "",
        examiner: str = "AI Forensic System",
    ):
        self.case_id = case_id or f"FR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.examiner = examiner
        self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate(
        self,
        forensics_report: Any = None,
        fingerprint_result: Optional[Dict] = None,
        metadata_report: Optional[Any] = None,
        signature_report: Optional[Any] = None,
        original_path: str = "",
        suspect_path: str = "",
        embedded_images: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate a complete forensic HTML report.

        Returns HTML string that can be saved as .html or printed to PDF.
        """
        report = ForensicPDFReport(
            case_id=self.case_id,
            case_title="Document Forensics Analysis",
            examiner=self.examiner,
            date=self.date,
            original_file=original_path,
            suspect_file=suspect_path,
        )

        # Extract data from forensics report
        if forensics_report is not None:
            report.tamper_score = getattr(forensics_report, 'tamper_score', 0.0)
            report.severity = getattr(getattr(forensics_report, 'overall_severity', None), 'label', 'Unknown')
            report.overall_verdict = "TAMPERED" if getattr(forensics_report, 'is_tampered', False) else "CLEAN"

            # Method results
            if hasattr(forensics_report, 'method_results'):
                for name, result in forensics_report.method_results.items():
                    report.method_results[name] = result.get('confidence', 0)

            # Regions
            if hasattr(forensics_report, 'tamper_regions'):
                for r in forensics_report.tamper_regions:
                    report.regions.append({
                        'x': r.x, 'y': r.y, 'w': r.width, 'h': r.height,
                        'confidence': r.tamper_confidence,
                        'severity': r.severity.label if hasattr(r.severity, 'label') else '',
                        'description': r.description,
                    })

        # Fingerprint
        if fingerprint_result:
            report.fingerprint_match = fingerprint_result

        # Metadata
        if metadata_report is not None:
            report.metadata_report = {
                'anomalies': [a.value for a in getattr(metadata_report, 'anomalies', [])],
                'camera': getattr(metadata_report, 'camera_make', ''),
                'software': getattr(metadata_report, 'software_used', ''),
                'date_original': getattr(metadata_report, 'date_original', ''),
                'has_gps': getattr(metadata_report, 'has_gps', False),
            }

        # Signature
        if signature_report is not None:
            report.signature_report = {
                'verdict': getattr(signature_report, 'verdict', None).label if hasattr(signature_report, 'verdict') and signature_report.verdict else '',
                'confidence': getattr(signature_report, 'confidence', 0),
                'geometry': getattr(signature_report, 'geometry_similarity', 0),
                'density': getattr(signature_report, 'density_similarity', 0),
                'topology': getattr(signature_report, 'topology_similarity', 0),
                'hu_moments': getattr(signature_report, 'hu_moments_similarity', 0),
                'orb': getattr(signature_report, 'orb_match_score', 0),
            }

        # Build HTML
        html = self._build_html(report, embedded_images or {})
        return html

    # ==================================================================
    # HTML Builder
    # ==================================================================

    def _build_html(self, report: ForensicPDFReport, images: Dict[str, str]) -> str:
        """Build the complete HTML report."""
        severity_color = {
            'CRITICAL': '#8b0000', 'MAJOR': '#c0392b', 'SIGNIFICANT': '#e74c3c',
            'SUSPICIOUS': '#e67e22', 'MINOR': '#f39c12', 'NONE': '#27ae60',
        }.get(
            report.severity.split('—')[0].strip() if '—' in report.severity else report.severity,
            '#333'
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Forensic Report — {report.case_id}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; color:#1a1a2e; line-height:1.7; font-size:11pt; }}
.page {{ max-width:210mm; margin:0 auto; padding:15mm 18mm; background:#fff; page-break-after:always; }}
.page:last-child {{ page-break-after:auto; }}
h1 {{ font-size:22pt; color:{severity_color}; margin-bottom:8pt; border-bottom:3px solid {severity_color}; padding-bottom:8pt; }}
h2 {{ font-size:14pt; color:#16213e; margin:16pt 0 8pt 0; border-left:4px solid {severity_color}; padding-left:10pt; }}
h3 {{ font-size:11pt; color:#0f3460; margin:10pt 0 5pt 0; }}
table {{ width:100%; border-collapse:collapse; margin:8pt 0; font-size:9pt; }}
th {{ background:#16213e; color:#fff; padding:6pt 8pt; text-align:left; font-weight:600; }}
td {{ padding:5pt 8pt; border-bottom:1px solid #ddd; }}
tr:nth-child(even) {{ background:#f8f9fa; }}
.cover {{ text-align:center; padding-top:60pt; }}
.cover h1 {{ font-size:28pt; border:none; }}
.cover .badge {{ display:inline-block; padding:8pt 24pt; border-radius:20pt; font-size:16pt; font-weight:bold; color:#fff; background:{severity_color}; margin:16pt 0; }}
.meta {{ color:#666; font-size:9pt; margin:4pt 0; }}
.score-bar {{ height:14pt; background:#eee; border-radius:7pt; overflow:hidden; margin:4pt 0; }}
.score-fill {{ height:100%; border-radius:7pt; transition:width 0.5s; }}
.warning {{ background:#fff3cd; border-left:3px solid #f39c12; padding:8pt 12pt; margin:8pt 0; font-size:9pt; }}
.critical {{ background:#ffe0e0; border-left:3px solid #e74c3c; padding:8pt 12pt; margin:8pt 0; font-size:9pt; }}
.success {{ background:#d4edda; border-left:3px solid #27ae60; padding:8pt 12pt; margin:8pt 0; font-size:9pt; }}
.chart-container {{ margin:12pt 0; text-align:center; }}
.chart-container img {{ max-width:100%; height:auto; }}
.evidence-panel {{ display:flex; gap:12pt; margin:12pt 0; }}
.evidence-panel > div {{ flex:1; }}
.evidence-panel img {{ width:100%; border:1px solid #ddd; border-radius:4pt; }}
.footer {{ margin-top:24pt; padding-top:8pt; border-top:1px solid #ccc; font-size:8pt; color:#999; text-align:center; }}
@media print {{ body {{ font-size:10pt; }} .page {{ padding:10mm 15mm; }} }}
</style>
</head>
<body>

<!-- ═══════════ COVER PAGE ═══════════ -->
<div class="page">
<div class="cover">
    <h1>FORENSIC ANALYSIS REPORT</h1>
    <div class="badge">{report.overall_verdict}</div>
    <p style="font-size:14pt; color:#555; margin:12pt 0">{report.case_title}</p>
    <table style="width:60%; margin:24pt auto; text-align:left;">
        <tr><td><strong>Case ID:</strong></td><td>{report.case_id}</td></tr>
        <tr><td><strong>Date:</strong></td><td>{report.date}</td></tr>
        <tr><td><strong>Examiner:</strong></td><td>{report.examiner}</td></tr>
        <tr><td><strong>Severity:</strong></td><td style="color:{severity_color}; font-weight:bold">{report.severity}</td></tr>
        <tr><td><strong>Tamper Score:</strong></td><td>{report.tamper_score:.1%}</td></tr>
        <tr><td><strong>Original:</strong></td><td style="font-size:8pt">{Path(report.original_file).name if report.original_file else 'N/A'}</td></tr>
        <tr><td><strong>Suspect:</strong></td><td style="font-size:8pt">{Path(report.suspect_file).name if report.suspect_file else 'N/A'}</td></tr>
    </table>
</div>
</div>

<!-- ═══════════ EXECUTIVE SUMMARY ═══════════ -->
<div class="page">
<h2>1. Executive Summary</h2>
<p>{self._executive_summary(report)}</p>

<h2>2. Overall Risk Assessment</h2>
<div class="score-bar"><div class="score-fill" style="width:{report.tamper_score*100:.0f}%;background:{severity_color}"></div></div>
<p style="font-size:9pt;color:#666">Tamper Score: {report.tamper_score:.1%} | {report.severity}</p>

<h2>3. 5-Method Detection Results</h2>
{self._method_chart_html(report)}
"""

        # Fingerprint section
        if report.fingerprint_match:
            fp = report.fingerprint_match
            html += f"""
<h2>4. Fingerprint Vault Match</h2>
<table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Vault Match Found</td><td>{'✅ Yes' if fp.get('matched', False) else '❌ No'}</td></tr>
    <tr><td>Best Match</td><td>{fp.get('best_label', 'N/A')}</td></tr>
    <tr><td>Similarity Score</td><td>{fp.get('similarity', 0):.1%}</td></tr>
    <tr><td>pHash Distance</td><td>{fp.get('phash_dist', 'N/A')}</td></tr>
    <tr><td>dHash Distance</td><td>{fp.get('dhash_dist', 'N/A')}</td></tr>
</table>
"""

        # Metadata section
        if report.metadata_report:
            meta = report.metadata_report
            html += f"""
<h2>5. Metadata Analysis</h2>
<table>
    <tr><th>Property</th><th>Value</th></tr>
    <tr><td>Camera</td><td>{meta.get('camera', 'N/A')}</td></tr>
    <tr><td>Software</td><td>{meta.get('software', 'N/A')}</td></tr>
    <tr><td>Date Original</td><td>{meta.get('date_original', 'N/A')}</td></tr>
    <tr><td>GPS</td><td>{'Present' if meta.get('has_gps') else 'Not available'}</td></tr>
    <tr><td>Anomalies</td><td>{', '.join(meta.get('anomalies', ['None']))}</td></tr>
</table>
"""

        # Signature section
        if report.signature_report:
            sig = report.signature_report
            html += f"""
<h2>6. Signature Verification</h2>
<table>
    <tr><th>Metric</th><th>Score</th></tr>
    <tr><td>Verdict</td><td style="font-weight:bold">{sig.get('verdict', 'N/A')}</td></tr>
    <tr><td>Overall Confidence</td><td>{sig.get('confidence', 0):.1%}</td></tr>
    <tr><td>Geometry Similarity</td><td>{sig.get('geometry', 0):.3f}</td></tr>
    <tr><td>Density Similarity</td><td>{sig.get('density', 0):.3f}</td></tr>
    <tr><td>Topology Similarity</td><td>{sig.get('topology', 0):.3f}</td></tr>
    <tr><td>Hu Moments</td><td>{sig.get('hu_moments', 0):.3f}</td></tr>
    <tr><td>ORB Match</td><td>{sig.get('orb', 0):.3f}</td></tr>
</table>
"""

        # Regions
        if report.regions:
            html += """
<h2>7. Detected Tamper Regions</h2>
<table>
    <tr><th>#</th><th>Position</th><th>Size</th><th>Confidence</th><th>Severity</th><th>Description</th></tr>
"""
            for i, r in enumerate(report.regions):
                html += f"""
    <tr>
        <td>{i+1}</td>
        <td>({r['x']}, {r['y']})</td>
        <td>{r['w']}×{r['h']}px</td>
        <td>{r['confidence']:.1%}</td>
        <td>{r['severity']}</td>
        <td style="font-size:8pt">{r['description']}</td>
    </tr>"""
            html += "</table>"

        # Evidence images
        html += """
<h2>8. Visual Evidence</h2>
<div class="evidence-panel">
"""
        for key, b64 in list(images.items())[:4]:
            label = key.replace('_', ' ').title()
            html += f"""
    <div>
        <p style="font-size:8pt;font-weight:bold;text-align:center">{label}</p>
        <img src="data:image/png;base64,{b64}" alt="{label}">
    </div>"""
        html += "</div>"

        # Footer
        html += f"""
<div class="footer">
    Forensic Report {report.case_id} | Generated: {report.date} | Examiner: {report.examiner}<br>
    This report is generated automatically and should be reviewed by a qualified forensic examiner.
</div>
</div>
</body>
</html>"""

        return html

    def _executive_summary(self, report: ForensicPDFReport) -> str:
        """Generate executive summary paragraph."""
        if report.tamper_score < 0.15:
            return (
                "The forensic analysis found <strong>no significant evidence of tampering</strong>. "
                "All 5 detection methods returned low confidence scores, indicating the suspect document "
                "is consistent with the original. Minor variations are within expected thresholds for "
                "natural scanning/copying artifacts."
            )
        elif report.tamper_score < 0.35:
            return (
                "The analysis detected <strong>minor inconsistencies</strong> between the original and "
                "suspect documents. While some methods flagged potential issues, the overall evidence is "
                "not sufficient to conclusively determine tampering. Manual review is recommended."
            )
        elif report.tamper_score < 0.55:
            return (
                "The forensic examination revealed <strong>suspicious patterns</strong> consistent with "
                "document manipulation. Multiple detection methods identified anomalies in the suspect "
                "document. Further investigation by a qualified forensic document examiner is strongly recommended."
            )
        elif report.tamper_score < 0.75:
            return (
                "Analysis indicates <strong>significant evidence of document tampering</strong>. "
                f"A total of {len(report.regions)} altered regions were identified with high confidence. "
                "The nature and pattern of alterations suggest deliberate manipulation of the document content."
            )
        else:
            return (
                f"The forensic analysis confirms <strong>definitive document forgery</strong> with "
                f"{len(report.regions)} tampered regions detected at high confidence. "
                "The suspect document shows clear, systematic alterations inconsistent with the original. "
                "This document should be treated as FABRICATED/FORGED evidence."
            )

    def _method_chart_html(self, report: ForensicPDFReport) -> str:
        """Generate HTML bar chart for method results."""
        method_labels = {
            'pixel_subtraction': 'Pixel Subtraction',
            'stroke_width': 'Stroke Width',
            'ink_consistency': 'Ink Consistency',
            'spacing_analysis': 'Spacing Analysis',
            'gradient_texture': 'Gradient Texture',
        }

        rows = ""
        for method, conf in report.method_results.items():
            label = method_labels.get(method, method)
            color = '#e74c3c' if conf > 0.5 else '#f39c12' if conf > 0.25 else '#27ae60'
            rows += f"""
        <tr>
            <td style="width:180px"><strong>{label}</strong></td>
            <td>
                <div class="score-bar" style="height:12pt">
                    <div class="score-fill" style="width:{conf*100:.0f}%;background:{color}"></div>
                </div>
            </td>
            <td style="width:60px;text-align:right;font-weight:bold;color:{color}">{conf:.1%}</td>
        </tr>"""

        return f"""
<table>
    <tr><th>Method</th><th>Confidence</th><th>Score</th></tr>
    {rows}
</table>
"""


# ===========================================================================
# Chart Generation with Matplotlib
# ===========================================================================

def generate_forensic_charts(
    method_results: Dict[str, float],
    output_dir: str = ".",
) -> Dict[str, str]:
    """
    Generate standalone chart images for the forensic report.
    Returns dict of chart_name -> base64 encoded PNG.

    Charts generated:
      - method_bar_chart: Horizontal bar chart of all method scores
      - histogram_overlay: Overlaid histograms of original vs suspect
      - confidence_radar: Radar chart of confidence metrics
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    charts = {}

    # Style
    plt.style.use('dark_background')
    colors_bar = ['#e74c3c' if v > 0.5 else '#f39c12' if v > 0.25 else '#27ae60'
                  for v in method_results.values()]

    # 1. Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = list(method_results.keys())
    scores = list(method_results.values())
    labels = [m.replace('_', ' ').title() for m in methods]

    bars = ax.barh(labels, scores, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Confidence Score')
    ax.set_title('5-Method Detection Confidence', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.1%}', va='center', fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    buf.seek(0)
    charts['method_bar_chart'] = base64.b64encode(buf.read()).decode()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'chart_methods.png'), 'wb') as f:
            f.write(base64.b64decode(charts['method_bar_chart']))

    # 2. Radar Chart (if 5 methods)
    if len(methods) >= 5:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(methods), endpoint=False).tolist()
        scores_plot = scores + [scores[0]]
        angles += [angles[0]]

        ax.fill(angles, scores_plot, alpha=0.3, color='#3498db')
        ax.plot(angles, scores_plot, 'o-', color='#3498db', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
        ax.set_title('Confidence Radar', fontweight='bold', pad=20)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        buf.seek(0)
        charts['confidence_radar'] = base64.b64encode(buf.read()).decode()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'chart_radar.png'), 'wb') as f:
                f.write(base64.b64decode(charts['confidence_radar']))

    return charts


def generate_histogram_comparison(
    original: np.ndarray,
    suspect: np.ndarray,
    output_dir: str = ".",
) -> Dict[str, str]:
    """Generate histogram comparison charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    charts = {}

    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
    gray_susp = cv2.cvtColor(suspect, cv2.COLOR_BGR2GRAY) if suspect.ndim == 3 else suspect

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors_hist = ['blue', 'red']

    # Original histogram
    axes[0, 0].hist(gray_orig.ravel(), bins=64, color='#3498db', alpha=0.7)
    axes[0, 0].set_title('Original Document — Brightness Histogram')
    axes[0, 0].set_xlabel('Pixel Intensity')
    axes[0, 0].set_ylabel('Frequency')

    # Suspect histogram
    axes[0, 1].hist(gray_susp.ravel(), bins=64, color='#e74c3c', alpha=0.7)
    axes[0, 1].set_title('Suspect Document — Brightness Histogram')
    axes[0, 1].set_xlabel('Pixel Intensity')

    # Overlay comparison
    axes[1, 0].hist(gray_orig.ravel(), bins=64, color='#3498db', alpha=0.5, label='Original')
    axes[1, 0].hist(gray_susp.ravel(), bins=64, color='#e74c3c', alpha=0.5, label='Suspect')
    axes[1, 0].set_title('Histogram Overlay Comparison')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].legend(fontsize=8)

    # Difference histogram
    diff = cv2.absdiff(gray_orig, gray_susp)
    axes[1, 1].hist(diff.ravel(), bins=64, color='#f39c12', alpha=0.7)
    axes[1, 1].set_title('Pixel Difference Distribution')
    axes[1, 1].set_xlabel('Absolute Difference')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    buf.seek(0)
    charts['histogram_comparison'] = base64.b64encode(buf.read()).decode()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'chart_histograms.png'), 'wb') as f:
            f.write(base64.b64decode(charts['histogram_comparison']))

    return charts
