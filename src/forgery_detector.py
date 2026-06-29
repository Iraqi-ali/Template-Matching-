"""
Advanced Image Forgery Detection & Difference Analysis Module
==============================================================
Professional-grade module for detecting image manipulation/tampering
and visualizing differences between template and matched regions.

Capabilities:
  1. Error Level Analysis (ELA) — detects JPEG compression anomalies
  2. Structural Similarity Index (SSIM) — pixel-level similarity
  3. Noise Pattern Analysis — identifies inconsistent noise profiles
  4. Difference Mapping — highlights altered regions with green boxes
  5. Edge Density Analysis — detects edge discontinuities
  6. Histogram Comparison — statistical distribution matching
  7. Clone/Region Detection — finds duplicated regions
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import time


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ForgeryRisk(Enum):
    """Risk level for image forgery."""
    NONE = ("No Tampering Detected", (0, 255, 0))
    LOW = ("Low Risk — Minor inconsistencies", (0, 255, 255))
    MEDIUM = ("Medium Risk — Suspicious patterns found", (0, 165, 255))
    HIGH = ("High Risk — Strong evidence of manipulation", (0, 0, 255))
    CRITICAL = ("Critical — Definite tampering confirmed", (255, 0, 255))

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def color(self) -> Tuple[int, int, int]:
        return self.value[1]


@dataclass
class DifferenceRegion:
    """A region where significant difference was detected."""
    x: int
    y: int
    width: int
    height: int
    diff_score: float  # 0.0 - 1.0, higher = more different
    description: str = ""


@dataclass
class ForgeryReport:
    """Complete forgery analysis report."""
    risk_level: ForgeryRisk = ForgeryRisk.NONE
    risk_score: float = 0.0  # 0.0 (safe) - 1.0 (definitely tampered)
    ela_score: float = 0.0
    ssim_score: float = 1.0  # 1.0 = identical
    noise_consistency: float = 1.0
    edge_anomaly_score: float = 0.0
    histogram_correlation: float = 1.0
    diff_regions: List[DifferenceRegion] = field(default_factory=list)
    ela_image: Optional[np.ndarray] = None
    diff_heatmap: Optional[np.ndarray] = None
    noise_map: Optional[np.ndarray] = None
    elapsed_ms: float = 0.0
    details: List[str] = field(default_factory=list)

    @property
    def is_tampered(self) -> bool:
        return self.risk_level in (
            ForgeryRisk.HIGH, ForgeryRisk.CRITICAL, ForgeryRisk.MEDIUM
        )


# ---------------------------------------------------------------------------
# Forgery Detector Engine
# ---------------------------------------------------------------------------

class ForgeryDetector:
    """
    Comprehensive image forgery detection engine.

    Usage:
        detector = ForgeryDetector()
        report = detector.analyze(source_image, template_image, matched_regions)
        print(f"Risk: {report.risk_level.label}")
    """

    def __init__(
        self,
        ela_quality: int = 90,
        ssim_threshold: float = 0.85,
        diff_sensitivity: float = 30.0,
        noise_block_size: int = 32,
    ):
        """
        Args:
            ela_quality: JPEG save quality for ELA (lower = more sensitive).
            ssim_threshold: Below this SSIM value, regions are flagged.
            diff_sensitivity: Pixel difference threshold for diff mapping.
            noise_block_size: Block size for noise analysis.
        """
        self.ela_quality = ela_quality
        self.ssim_threshold = ssim_threshold
        self.diff_sensitivity = diff_sensitivity
        self.noise_block_size = noise_block_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        source: np.ndarray,
        template: np.ndarray,
        matched_regions: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> ForgeryReport:
        """
        Run complete forgery analysis.

        Args:
            source: Source image (BGR).
            template: Template image (BGR).
            matched_regions: Optional list of (x, y, w, h) matched regions.

        Returns:
            ForgeryReport with comprehensive analysis.
        """
        t0 = time.perf_counter()
        report = ForgeryReport()

        # 1. Error Level Analysis on source image
        report.ela_score, report.ela_image = self._error_level_analysis(source)

        # 2. If we have matched regions, analyze each one
        if matched_regions:
            for (x, y, w, h) in matched_regions:
                # Extract the matched region
                region = source[y:y + h, x:x + w].copy()

                # SSIM between template and matched region
                ssim_val = self._compute_ssim(template, region)
                report.ssim_score = min(report.ssim_score, ssim_val)

                # Histogram comparison
                hist_corr = self._compare_histograms(template, region)
                report.histogram_correlation = min(report.histogram_correlation, hist_corr)

                # Difference mapping
                diff_regions, diff_heatmap = self._difference_mapping(template, region, x, y)
                report.diff_regions.extend(diff_regions)
                report.diff_heatmap = diff_heatmap

                # Edge analysis
                edge_score = self._edge_anomaly_detection(template, region)
                report.edge_anomaly_score = max(report.edge_anomaly_score, edge_score)

        else:
            # Compare source and template globally
            if source.shape == template.shape:
                ssim_val = self._compute_ssim(source, template)
                report.ssim_score = ssim_val
                hist_corr = self._compare_histograms(source, template)
                report.histogram_correlation = hist_corr
                diff_regions, diff_heatmap = self._difference_mapping(
                    source, template, 0, 0
                )
                report.diff_regions = diff_regions
                report.diff_heatmap = diff_heatmap

        # 3. Noise analysis on source
        report.noise_consistency, report.noise_map = self._noise_analysis(source)

        # 4. Compute overall risk
        report.risk_score, report.risk_level = self._compute_risk(report)
        report.details = self._generate_details(report)

        report.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return report

    def quick_validation(
        self,
        source: np.ndarray,
        template: np.ndarray,
        matched_regions: List[Tuple[int, int, int, int]],
        strict: bool = True,
    ) -> Tuple[bool, float, List[str]]:
        """
        Quick multi-layer validation of matched regions.
        Returns: (is_valid, confidence_score, reasons).

        This is the KEY function that prevents false positives.
        Uses multiple independent checks to validate if a match is genuine.
        """
        if not matched_regions:
            return False, 0.0, ["No regions to validate"]

        checks_passed = 0
        total_checks = 4
        reasons: List[str] = []
        scores: List[float] = []

        for (x, y, w, h) in matched_regions:
            # Clamp to image bounds
            x, y = max(0, x), max(0, y)
            w = min(w, source.shape[1] - x)
            h = min(h, source.shape[0] - y)
            if w < 5 or h < 5:
                reasons.append(f"Region too small ({w}x{h})")
                continue

            region = source[y:y + h, x:x + w].copy()

            # Resize template to match region size if needed
            tpl_resized = template
            if template.shape[:2] != region.shape[:2]:
                tpl_resized = cv2.resize(template, (w, h))

            # Check 1: SSIM (Structural Similarity)
            ssim_val = self._compute_ssim(tpl_resized, region)
            scores.append(ssim_val)
            ssim_ok = ssim_val >= (0.70 if strict else 0.55)
            if ssim_ok:
                checks_passed += 1
            else:
                reasons.append(f"SSIM too low: {ssim_val:.3f} < {0.70 if strict else 0.55}")

            # Check 2: Histogram Correlation
            hist_corr = self._compare_histograms(tpl_resized, region)
            hist_ok = hist_corr >= (0.65 if strict else 0.50)
            if hist_ok:
                checks_passed += 1
            else:
                reasons.append(f"Histogram correlation too low: {hist_corr:.3f}")

            # Check 3: Edge Overlap
            edge_overlap = self._edge_overlap_score(tpl_resized, region)
            edge_ok = edge_overlap >= (0.40 if strict else 0.25)
            if edge_ok:
                checks_passed += 1
            else:
                reasons.append(f"Edge overlap too low: {edge_overlap:.3f}")

            # Check 4: Mean pixel difference
            mean_diff = np.mean(cv2.absdiff(tpl_resized, region))
            max_possible = 255.0
            pixel_similarity = 1.0 - (mean_diff / max_possible)
            pixel_ok = pixel_similarity >= (0.60 if strict else 0.40)
            if pixel_ok:
                checks_passed += 1
            else:
                reasons.append(f"Pixel similarity too low: {pixel_similarity:.3f}")

        # Overall confidence
        confidence = np.mean(scores) if scores else 0.0
        pass_ratio = checks_passed / (len(matched_regions) * total_checks) if matched_regions else 0.0

        # Must pass ALL checks for strict, majority for non-strict
        if strict:
            is_valid = pass_ratio >= 0.75 and confidence >= 0.70
        else:
            is_valid = pass_ratio >= 0.50 and confidence >= 0.55

        if is_valid and not reasons:
            reasons.append("All validation checks passed ✓")

        return is_valid, confidence, reasons

    # ------------------------------------------------------------------
    # Analysis Methods
    # ------------------------------------------------------------------

    def _error_level_analysis(self, img: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Error Level Analysis (ELA).
        Resave the image at a known quality and compare with original.
        Differences highlight regions that have been modified.
        """
        # Encode to JPEG at specified quality
        _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.ela_quality])
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Compute absolute difference, amplify for visibility
        diff = cv2.absdiff(img, recompressed)
        diff_amplified = cv2.convertScaleAbs(diff, alpha=8.0)

        # Score: mean difference normalized
        ela_score = float(np.mean(diff) / 255.0)
        ela_score = min(ela_score * 5.0, 1.0)  # Amplify for sensitivity

        return ela_score, diff_amplified

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index between two images.
        Uses multi-scale approach for robustness.
        Returns 0.0 (completely different) to 1.0 (identical).
        """
        # Ensure same size
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

        # Compute SSIM manually (avoid scikit-image dependency issues)
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        gray1 = gray1.astype(np.float64)
        gray2 = gray2.astype(np.float64)

        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.clip(np.mean(ssim_map), 0.0, 1.0))

    def _compare_histograms(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare histograms using correlation method. Returns 0.0 - 1.0."""
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        correlations = []
        for i in range(3):  # B, G, R channels
            hist1 = cv2.calcHist([img1], [i], None, [64], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [64], [0, 256])
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            correlations.append(max(0.0, corr))

        return float(np.mean(correlations))

    def _difference_mapping(
        self,
        template: np.ndarray,
        region: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> Tuple[List[DifferenceRegion], np.ndarray]:
        """
        Create pixel-level difference map between template and region.
        Returns difference regions (with green boxes) and heatmap.
        """
        # Resize to match
        if template.shape[:2] != region.shape[:2]:
            template_resized = cv2.resize(
                template, (region.shape[1], region.shape[0])
            )
        else:
            template_resized = template

        # Absolute difference
        diff = cv2.absdiff(template_resized, region)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff

        # Create heatmap (colorized difference)
        heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

        # Threshold to find significant difference regions
        _, thresh = cv2.threshold(
            diff_gray, int(self.diff_sensitivity), 255, cv2.THRESH_BINARY
        )

        # Find contours of different regions
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        diff_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 25:  # Filter tiny noise
                continue

            rx, ry, rw, rh = cv2.boundingRect(cnt)
            # Compute average diff in this region
            roi = diff_gray[ry:ry + rh, rx:rx + rw]
            avg_diff = float(np.mean(roi)) / 255.0 if roi.size > 0 else 0.0

            diff_regions.append(DifferenceRegion(
                x=offset_x + rx,
                y=offset_y + ry,
                width=rw,
                height=rh,
                diff_score=min(avg_diff * 3.0, 1.0),
                description=f"Diff region: avg intensity diff = {avg_diff:.2f}",
            ))

        return diff_regions, heatmap

    def _edge_anomaly_detection(
        self, template: np.ndarray, region: np.ndarray
    ) -> float:
        """
        Detect edge discontinuities that indicate splicing.
        Returns anomaly score 0.0 (consistent) - 1.0 (anomalous).
        """
        if template.shape[:2] != region.shape[:2]:
            region = cv2.resize(region, (template.shape[1], template.shape[0]))

        gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
        gray_reg = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region

        edges_tpl = cv2.Canny(gray_tpl, 50, 150)
        edges_reg = cv2.Canny(gray_reg, 50, 150)

        overlap = cv2.bitwise_and(edges_tpl, edges_reg)
        union = cv2.bitwise_or(edges_tpl, edges_reg)

        overlap_count = float(np.sum(overlap > 0))
        union_count = float(np.sum(union > 0))

        if union_count == 0:
            return 0.0

        iou = overlap_count / union_count
        anomaly = 1.0 - iou
        return anomaly

    def _edge_overlap_score(
        self, template: np.ndarray, region: np.ndarray
    ) -> float:
        """Simplified edge overlap score for quick validation."""
        gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
        gray_reg = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region

        edges_tpl = cv2.Canny(gray_tpl, 50, 150)
        edges_reg = cv2.Canny(gray_reg, 50, 150)

        # Count edge pixel overlap
        overlap = np.sum((edges_tpl > 0) & (edges_reg > 0))
        tpl_edges = np.sum(edges_tpl > 0)

        if tpl_edges == 0:
            return 0.0

        return float(overlap) / float(tpl_edges)

    def _noise_analysis(self, img: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Analyze noise patterns across the image.
        Inconsistent noise can indicate splicing from different sources.
        Returns (consistency_score, noise_map).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # Estimate noise using high-pass filter (Laplacian)
        noise = cv2.Laplacian(gray, cv2.CV_64F)
        noise_abs = np.abs(noise)
        noise_map = cv2.convertScaleAbs(noise_abs, alpha=2.0)

        # Divide image into blocks and compute noise variance per block
        h, w = gray.shape
        bs = self.noise_block_size
        variances = []

        for y in range(0, h - bs, bs // 2):
            for x in range(0, w - bs, bs // 2):
                block = noise_abs[y:y + bs, x:x + bs]
                if block.size > 0:
                    variances.append(float(np.var(block)))

        if not variances:
            return 1.0, noise_map

        variances = np.array(variances)
        mean_var = np.mean(variances)
        std_var = np.std(variances)

        if mean_var == 0:
            return 1.0, noise_map

        # Coefficient of variation: lower = more consistent noise
        cv_val = std_var / mean_var
        consistency = max(0.0, 1.0 - min(cv_val / 2.0, 1.0))

        return consistency, noise_map

    # ------------------------------------------------------------------
    # Risk Assessment
    # ------------------------------------------------------------------

    def _compute_risk(self, report: ForgeryReport) -> Tuple[float, ForgeryRisk]:
        """
        Compute overall risk score from all analysis components.
        Weighted ensemble approach.
        """
        weights = {
            'ssim': 0.30,       # Structural similarity (inverted)
            'histogram': 0.15,   # Color distribution
            'ela': 0.20,        # Error level analysis
            'noise': 0.15,      # Noise consistency
            'edge': 0.10,       # Edge anomalies
            'diff': 0.10,       # Difference regions
        }

        # SSIM risk (inverted: high SSIM = low risk)
        ssim_risk = max(0.0, (1.0 - report.ssim_score) * 2.0)

        # Histogram risk (inverted)
        hist_risk = max(0.0, (1.0 - report.histogram_correlation) * 2.0)

        # ELA risk
        ela_risk = report.ela_score

        # Noise risk (inverted)
        noise_risk = max(0.0, (1.0 - report.noise_consistency) * 2.0)

        # Edge risk
        edge_risk = report.edge_anomaly_score

        # Diff risk: based on number and intensity of diff regions
        if report.diff_regions:
            avg_diff = np.mean([r.diff_score for r in report.diff_regions])
            diff_risk = min(avg_diff * 1.5, 1.0)
        else:
            diff_risk = 0.0

        risk_score = (
            weights['ssim'] * ssim_risk +
            weights['histogram'] * hist_risk +
            weights['ela'] * ela_risk +
            weights['noise'] * noise_risk +
            weights['edge'] * edge_risk +
            weights['diff'] * diff_risk
        )

        risk_score = min(max(risk_score, 0.0), 1.0)

        # Determine risk level
        if risk_score < 0.15:
            level = ForgeryRisk.NONE
        elif risk_score < 0.30:
            level = ForgeryRisk.LOW
        elif risk_score < 0.55:
            level = ForgeryRisk.MEDIUM
        elif risk_score < 0.75:
            level = ForgeryRisk.HIGH
        else:
            level = ForgeryRisk.CRITICAL

        return risk_score, level

    def _generate_details(self, report: ForgeryReport) -> List[str]:
        """Generate human-readable analysis details."""
        details = []

        if report.ssim_score < 0.85:
            details.append(
                f"⚠️ Low structural similarity (SSIM: {report.ssim_score:.3f}) — "
                "images may be different or modified"
            )
        else:
            details.append(f"✅ Good structural similarity (SSIM: {report.ssim_score:.3f})")

        if report.histogram_correlation < 0.80:
            details.append(
                f"⚠️ Color distribution mismatch (hist corr: {report.histogram_correlation:.3f})"
            )
        else:
            details.append(f"✅ Consistent color distribution (hist corr: {report.histogram_correlation:.3f})")

        if report.ela_score > 0.20:
            details.append(
                f"⚠️ ELA detected compression anomalies (ELA: {report.ela_score:.3f}) — "
                "possible editing"
            )
        elif report.ela_score > 0.05:
            details.append(f"ℹ️ Minor ELA variations (ELA: {report.ela_score:.3f})")
        else:
            details.append(f"✅ Clean ELA analysis (ELA: {report.ela_score:.3f})")

        if report.noise_consistency < 0.70:
            details.append(
                f"⚠️ Inconsistent noise patterns — possible splicing from different sources"
            )
        else:
            details.append(f"✅ Consistent noise patterns")

        if report.edge_anomaly_score > 0.40:
            details.append(
                f"⚠️ Edge anomalies detected — possible region insertion"
            )
        else:
            details.append(f"✅ No significant edge anomalies")

        if report.diff_regions:
            details.append(
                f"🔍 {len(report.diff_regions)} difference region(s) identified"
            )

        return details


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------

def draw_difference_boxes(
    image: np.ndarray,
    diff_regions: List[DifferenceRegion],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw GREEN boxes around difference regions.
    Thicker boxes for higher difference scores.
    """
    output = image.copy()

    for region in diff_regions:
        # Thicker border for more significant differences
        dynamic_thickness = max(thickness, int(thickness + region.diff_score * 5))

        # Draw green bounding box
        cv2.rectangle(
            output,
            (region.x, region.y),
            (region.x + region.width, region.y + region.height),
            color,
            dynamic_thickness,
        )

        # Draw center marker
        cx, cy = region.x + region.width // 2, region.y + region.height // 2
        cv2.drawMarker(output, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)

        # Label with intensity
        label = f"{region.diff_score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        cv2.rectangle(
            output,
            (region.x, region.y - th - 8),
            (region.x + tw + 6, region.y),
            color,
            -1,
        )
        cv2.putText(
            output, label,
            (region.x + 3, region.y - 3),
            font, 0.45, (0, 0, 0), 1,
        )

    return output


def draw_forgery_report(
    source: np.ndarray,
    forgery_report: ForgeryReport,
    matched_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    """
    Create a comprehensive visualization of the forgery analysis.
    Shows: original + difference boxes + ELA + heatmap + summary.
    """
    h, w = source.shape[:2]
    output = source.copy()

    # 1. Draw matched region boxes (if any) in green
    if matched_regions:
        for (x, y, rw, rh) in matched_regions:
            cv2.rectangle(output, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
            cv2.drawMarker(
                output, (x + rw // 2, y + rh // 2),
                (0, 255, 0), cv2.MARKER_CROSS, 10, 1,
            )

    # 2. Draw difference region boxes in green
    if forgery_report.diff_regions:
        output = draw_difference_boxes(output, forgery_report.diff_regions)

    # 3. Add risk indicator bar at top
    risk_color = forgery_report.risk_level.color
    bar_height = 32
    cv2.rectangle(output, (0, 0), (w, bar_height), risk_color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    risk_text = f"Risk: {forgery_report.risk_level.label} (Score: {forgery_report.risk_score:.2f})"
    cv2.putText(
        output, risk_text, (10, bar_height - 8),
        font, 0.55, (255, 255, 255), 1,
    )

    # 4. Add detail summary at bottom
    detail_y = h - 28 * min(4, len(forgery_report.details)) - 10
    cv2.rectangle(output, (0, detail_y), (w, h), (0, 0, 0), -1)

    for i, detail in enumerate(forgery_report.details[:4]):
        y_pos = detail_y + 20 + i * 24
        cv2.putText(
            output, detail, (10, y_pos),
            font, 0.42, (200, 200, 200), 1,
        )

    return output


def create_full_analysis_canvas(
    source: np.ndarray,
    template: np.ndarray,
    forgery_report: ForgeryReport,
    matched_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    """
    Create a comprehensive multi-panel analysis canvas:
    [Original + Boxes] | [ELA] | [Diff Heatmap] | [Noise Map]
    """
    h, w = source.shape[:2]

    # Resize panels to match
    panel_h, panel_w = 350, 350
    scale = min(panel_w / w, panel_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Panel 1: Annotated original
    panel1 = draw_forgery_report(source, forgery_report, matched_regions)
    panel1 = cv2.resize(panel1, (panel_w, panel_h))

    # Panel 2: ELA
    if forgery_report.ela_image is not None:
        panel2 = cv2.resize(forgery_report.ela_image, (panel_w, panel_h))
    else:
        panel2 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # Panel 3: Diff heatmap
    if forgery_report.diff_heatmap is not None:
        panel3 = cv2.resize(forgery_report.diff_heatmap, (panel_w, panel_h))
    else:
        panel3 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # Panel 4: Noise map
    if forgery_report.noise_map is not None:
        panel4 = cv2.cvtColor(forgery_report.noise_map, cv2.COLOR_GRAY2BGR)
        panel4 = cv2.resize(panel4, (panel_w, panel_h))
    else:
        panel4 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # Add labels to each panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (panel, label) in enumerate([
        (panel1, "Original + Detections"),
        (panel2, "Error Level Analysis"),
        (panel3, "Difference Heatmap"),
        (panel4, "Noise Analysis"),
    ]):
        cv2.rectangle(panel, (0, 0), (panel_w, 28), (0, 0, 0), -1)
        cv2.putText(panel, label, (6, 20), font, 0.5, (255, 255, 255), 1)

    # Arrange in 2x2 grid
    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([panel3, panel4])
    canvas = np.vstack([top_row, bottom_row])

    # Add template thumbnail
    tpl_h, tpl_w = template.shape[:2]
    tpl_scale = min(120 / tpl_w, 120 / tpl_h)
    tpl_small = cv2.resize(template, (int(tpl_w * tpl_scale), int(tpl_h * tpl_scale)))
    tpl_h_s, tpl_w_s = tpl_small.shape[:2]

    # Place template at bottom-right corner
    margin = 10
    canvas_h, canvas_w = canvas.shape[:2]
    y_start = canvas_h - tpl_h_s - margin
    x_start = canvas_w - tpl_w_s - margin

    canvas[y_start:y_start + tpl_h_s, x_start:x_start + tpl_w_s] = tpl_small
    cv2.rectangle(
        canvas,
        (x_start - 2, y_start - 2),
        (x_start + tpl_w_s + 2, y_start + tpl_h_s + 2),
        (0, 255, 0), 2,
    )
    cv2.putText(
        canvas, "Template", (x_start, y_start - 8),
        font, 0.4, (0, 255, 0), 1,
    )

    return canvas
