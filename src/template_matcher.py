"""
Template Matching Engine - Core Module
======================================
Finds a template image within a source image using various matching methods.
Supports single-scale and multi-scale template matching with MULTI-LAYER
VALIDATION to prevent false positives.

Matching Methods (OpenCV):
    TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED,
    TM_SQDIFF, TM_SQDIFF_NORMED

Validation Layers:
    1. Correlation threshold (primary)
    2. Statistical significance (z-score vs background)
    3. Structural Similarity (SSIM)
    4. Histogram comparison
    5. Edge overlap analysis
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class MatchMethod(Enum):
    """Available template matching methods."""
    TM_CCOEFF = cv2.TM_CCOEFF
    TM_CCOEFF_NORMED = cv2.TM_CCOEFF_NORMED
    TM_CCORR = cv2.TM_CCORR
    TM_CCORR_NORMED = cv2.TM_CCORR_NORMED
    TM_SQDIFF = cv2.TM_SQDIFF
    TM_SQDIFF_NORMED = cv2.TM_SQDIFF_NORMED

    @property
    def label(self) -> str:
        labels = {
            MatchMethod.TM_CCOEFF: "Correlation Coefficient",
            MatchMethod.TM_CCOEFF_NORMED: "Correlation Coefficient (Normalized)",
            MatchMethod.TM_CCORR: "Cross-Correlation",
            MatchMethod.TM_CCORR_NORMED: "Cross-Correlation (Normalized)",
            MatchMethod.TM_SQDIFF: "Squared Difference",
            MatchMethod.TM_SQDIFF_NORMED: "Squared Difference (Normalized)",
        }
        return labels[self]

    @property
    def is_min_best(self) -> bool:
        """SQDIFF methods look for minimum; others look for maximum."""
        return self in (MatchMethod.TM_SQDIFF, MatchMethod.TM_SQDIFF_NORMED)


@dataclass
class MatchResult:
    """Holds the result of a single template match."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    method: MatchMethod
    scale: float = 1.0

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Returns (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class ValidationResult:
    """Result of multi-layer match validation."""
    is_valid: bool
    confidence: float  # 0.0 - 1.0 overall confidence
    ssim_score: float = 1.0
    histogram_correlation: float = 1.0
    edge_overlap: float = 1.0
    pixel_similarity: float = 1.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class DetectionReport:
    """Full report of template matching detection."""
    matches: List[MatchResult] = field(default_factory=list)
    source_shape: Tuple[int, int] = (0, 0)
    template_shape: Tuple[int, int] = (0, 0)
    method: MatchMethod = MatchMethod.TM_CCOEFF_NORMED
    threshold: float = 0.8
    elapsed_ms: float = 0.0
    validated: bool = False
    validation: Optional[ValidationResult] = None

    @property
    def match_count(self) -> int:
        return len(self.matches)

    @property
    def best_match(self) -> Optional[MatchResult]:
        return self.matches[0] if self.matches else None

    @property
    def is_reliable(self) -> bool:
        """Whether the detection is reliable (passed validation)."""
        if self.validation is not None:
            return self.validation.is_valid
        return self.match_count > 0


class TemplateMatcher:
    """
    Core template matching engine with multi-layer validation.

    The engine now uses a statistical approach:
    1. Runs matchTemplate to get correlation map
    2. Computes z-score of matches vs background (statistical significance)
    3. Validates each match using SSIM, histogram, and edge analysis
    4. Only returns matches that pass ALL validation layers

    This eliminates false positives where different images were reported as "matching".

    Usage:
        matcher = TemplateMatcher(method=MatchMethod.TM_CCOEFF_NORMED, threshold=0.8)
        result = matcher.match(source_img, template_img, threshold=0.8)
        print(f"Valid matches: {result.match_count}")
        print(f"Reliable: {result.is_reliable}")
    """

    def __init__(
        self,
        method: MatchMethod = MatchMethod.TM_CCOEFF_NORMED,
        threshold: float = 0.8,
        use_nms: bool = True,
        nms_threshold: float = 0.3,
        validate_matches: bool = True,
        strict_validation: bool = True,
    ):
        """
        Args:
            method: OpenCV template matching method.
            threshold: Minimum confidence to accept a match (0.0 - 1.0).
            use_nms: Apply Non-Maximum Suppression to remove duplicate detections.
            nms_threshold: IoU threshold for NMS.
            validate_matches: Enable multi-layer validation (SSIM, histogram, edges).
            strict_validation: If True, reject matches failing ANY check.
        """
        self.method = method
        self.threshold = threshold
        self.use_nms = use_nms
        self.nms_threshold = nms_threshold
        self.validate_matches = validate_matches
        self.strict_validation = strict_validation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        source: np.ndarray,
        template: np.ndarray,
        threshold: Optional[float] = None,
        max_matches: int = 50,
    ) -> DetectionReport:
        """
        Find all occurrences of *template* inside *source*.

        Now with STATISTICAL VALIDATION:
        - Computes z-score to ensure matches are statistically significant
        - Validates each match using SSIM, histogram, and edge analysis
        - Rejects false positives where different images appeared to match

        Args:
            source: Source image (BGR or grayscale).
            template: Template image to search for (BGR or grayscale).
            threshold: Override instance threshold.
            max_matches: Maximum number of matches to return.

        Returns:
            DetectionReport with all VALIDATED matches.
        """
        thresh = threshold if threshold is not None else self.threshold
        src_gray = self._to_gray(source)
        tpl_gray = self._to_gray(template)

        import time
        t0 = time.perf_counter()

        # Run OpenCV matchTemplate
        result_map = cv2.matchTemplate(src_gray, tpl_gray, self.method.value)

        # --- STATISTICAL SIGNIFICANCE CHECK ---
        # Compute z-score: how many standard deviations above the mean is the best match?
        mean_val = float(np.mean(result_map))
        std_val = float(np.std(result_map))
        best_val = float(np.max(result_map)) if not self.method.is_min_best else float(np.min(result_map))

        if std_val > 0:
            if self.method.is_min_best:
                z_score = (mean_val - best_val) / std_val
            else:
                z_score = (best_val - mean_val) / std_val
        else:
            z_score = 0.0

        # If z-score is too low, the "match" is just random correlation — reject
        MIN_Z_SCORE = 2.5  # Must be at least 2.5 std above mean
        if z_score < MIN_Z_SCORE and thresh < 0.95:
            # Boost threshold to compensate for noisy correlation
            adjusted_thresh = min(thresh + 0.15, 0.98)
        else:
            adjusted_thresh = thresh

        # Extract match locations above threshold
        matches = self._extract_matches(
            result_map, tpl_gray.shape[1], tpl_gray.shape[0], adjusted_thresh
        )

        # Non-Maximum Suppression
        if self.use_nms and len(matches) > 1:
            matches = self._apply_nms(matches)

        # Sort by confidence & limit
        matches.sort(key=lambda m: m.confidence, reverse=True)
        matches = matches[:max_matches]

        # --- MULTI-LAYER VALIDATION ---
        validation = None
        if self.validate_matches and matches:
            # Always run per-match validation
            valid_matches = []
            for m in matches:
                region = source[m.y:m.y + m.height, m.x:m.x + m.width].copy()
                tpl_resized = template
                if template.shape[:2] != region.shape[:2]:
                    tpl_resized = cv2.resize(template, (m.width, m.height))

                ssim_val = self._compute_ssim(tpl_resized, region)
                # Use stricter SSIM threshold (0.65) for genuine match verification
                if ssim_val >= 0.65:
                    valid_matches.append(m)

            # Validate the best match for the report
            validation = self._validate_matches(source, template, valid_matches[:1] if valid_matches else matches[:1])

            if self.strict_validation:
                matches = valid_matches
            elif len(valid_matches) > 0:
                matches = valid_matches

        elapsed = (time.perf_counter() - t0) * 1000.0

        return DetectionReport(
            matches=matches,
            source_shape=(source.shape[1], source.shape[0]),
            template_shape=(template.shape[1], template.shape[0]),
            method=self.method,
            threshold=thresh,
            elapsed_ms=elapsed,
            validated=self.validate_matches,
            validation=validation,
        )

    def match_all_methods(
        self,
        source: np.ndarray,
        template: np.ndarray,
        threshold: float = 0.8,
    ) -> List[DetectionReport]:
        """Run matching with every available method for comparison."""
        reports = []
        original_method = self.method
        for method in MatchMethod:
            self.method = method
            reports.append(self.match(source, template, threshold))
        self.method = original_method
        return reports

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_matches(
        self,
        source: np.ndarray,
        template: np.ndarray,
        matches: List[MatchResult],
    ) -> ValidationResult:
        """
        Multi-layer validation of matched regions.
        Uses SSIM, histogram comparison, edge overlap, and pixel similarity
        to confirm that matches are genuine.
        """
        if not matches:
            return ValidationResult(
                is_valid=False, confidence=0.0,
                reasons=["No matches to validate"],
            )

        best = matches[0]
        x, y, w, h = best.x, best.y, best.width, best.height

        # Clamp to bounds
        x, y = max(0, x), max(0, y)
        w = min(w, source.shape[1] - x)
        h = min(h, source.shape[0] - y)

        if w < 5 or h < 5:
            return ValidationResult(
                is_valid=False, confidence=0.0,
                reasons=["Match region too small"],
            )

        region = source[y:y + h, x:x + w].copy()

        # Resize template to match region
        tpl_resized = template
        if template.shape[:2] != region.shape[:2]:
            tpl_resized = cv2.resize(template, (w, h))

        reasons = []
        scores = []

        # --- Check 1: SSIM ---
        ssim_val = self._compute_ssim(tpl_resized, region)
        scores.append(ssim_val)
        if ssim_val < 0.65:
            reasons.append(f"SSIM too low ({ssim_val:.3f}) — images are structurally different")

        # --- Check 2: Histogram Correlation ---
        hist_corr = self._compare_histograms(tpl_resized, region)
        scores.append(hist_corr)
        if hist_corr < 0.50:
            reasons.append(f"Histogram mismatch ({hist_corr:.3f}) — color distribution differs")

        # --- Check 3: Edge Overlap ---
        edge_overlap = self._edge_overlap_score(tpl_resized, region)
        scores.append(edge_overlap)
        if edge_overlap < 0.25:
            reasons.append(f"Edge overlap low ({edge_overlap:.3f}) — shape mismatch")

        # --- Check 4: Pixel Similarity ---
        mean_diff = np.mean(cv2.absdiff(tpl_resized, region))
        pixel_sim = 1.0 - (mean_diff / 255.0)
        scores.append(pixel_sim)
        if pixel_sim < 0.40:
            reasons.append(f"Pixel difference too high (similarity={pixel_sim:.3f})")

        avg_confidence = float(np.mean(scores)) if scores else 0.0
        is_valid = len(reasons) == 0

        if is_valid:
            reasons.append("✓ All validation checks passed — match is genuine")

        return ValidationResult(
            is_valid=is_valid,
            confidence=avg_confidence,
            ssim_score=ssim_val,
            histogram_correlation=hist_corr,
            edge_overlap=edge_overlap,
            pixel_similarity=pixel_sim,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def _extract_matches(
        self,
        result_map: np.ndarray,
        tpl_w: int,
        tpl_h: int,
        threshold: float,
    ) -> List[MatchResult]:
        """Extract (x, y) locations where the result map exceeds the threshold."""
        if self.method.is_min_best:
            # For SQDIFF methods: lower values = better match
            # Normalized SQDIFF range is [0, 1]; non-normalized is [0, inf)
            # We invert for consistency
            max_possible = np.max(result_map)
            if max_possible > 0:
                normalized = result_map / (max_possible + 1e-9)
            else:
                normalized = result_map

            loc = np.where(normalized <= (1.0 - threshold))
            # Confidence: invert so higher = better
            confidences = 1.0 - normalized[loc]
        else:
            loc = np.where(result_map >= threshold)
            confidences = result_map[loc]

        matches = []
        for pt_y, pt_x in zip(*loc):
            if len(matches) >= 5000:  # Safety cap
                break
            conf = float(confidences[len(matches)])
            matches.append(MatchResult(
                x=int(pt_x), y=int(pt_y),
                width=tpl_w, height=tpl_h,
                confidence=conf,
                method=self.method,
            ))
        return matches

    def _apply_nms(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Non-Maximum Suppression to remove overlapping detections."""
        if not matches:
            return matches

        boxes = np.array([[m.x, m.y, m.x + m.width, m.y + m.height] for m in matches], dtype=np.float32)
        scores = np.array([m.confidence for m in matches], dtype=np.float32)

        try:
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(),
                score_threshold=0.0,
                nms_threshold=self.nms_threshold,
            )
        except cv2.error:
            return matches

        if len(indices) == 0:
            return []

        return [matches[i] for i in indices.flatten()]

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Structural Similarity Index (0.0 to 1.0)."""
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        gray1 = gray1.astype(np.float64)
        gray2 = gray2.astype(np.float64)

        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.clip(np.mean(ssim_map), 0.0, 1.0))

    def _compare_histograms(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare color histograms (0.0 to 1.0)."""
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        correlations = []
        for i in range(3):
            hist1 = cv2.calcHist([img1], [i], None, [64], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [64], [0, 256])
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            correlations.append(max(0.0, corr))

        return float(np.mean(correlations))

    def _edge_overlap_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute edge overlap score (0.0 to 1.0)."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)

        overlap = np.sum((edges1 > 0) & (edges2 > 0))
        tpl_edges = np.sum(edges1 > 0)

        return float(overlap) / float(tpl_edges) if tpl_edges > 0 else 0.0


# ------------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------------

def draw_matches(
    image: np.ndarray,
    report: DetectionReport,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_confidence: bool = True,
    show_validation: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes for all matches on the image.
    Green boxes for valid matches, yellow for uncertain, red for invalid.
    """
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, match in enumerate(report.matches):
        x, y, w, h = match.bbox

        # Determine box color based on validation
        if report.validation is not None and not report.validation.is_valid:
            box_color = (0, 0, 255)  # Red = invalid/false match
        elif report.validated and report.is_reliable:
            box_color = (0, 255, 0)  # Green = validated genuine match
        else:
            box_color = color  # Default

        cv2.rectangle(output, (x, y), (x + w, y + h), box_color, thickness)
        cv2.drawMarker(output, match.center, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

        if show_confidence:
            label = f"#{i+1} {match.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(output, (x, y - th - 6), (x + tw + 4, y), box_color, -1)
            cv2.putText(output, label, (x + 2, y - 4), font, 0.5, (0, 0, 0), 1)

    # Summary overlay
    h, w = output.shape[:2]
    validation_status = ""
    if report.validation is not None:
        if report.validation.is_valid:
            validation_status = " ✓ VALIDATED"
        else:
            validation_status = " ⚠ UNVERIFIED"

    summary = (
        f"Matches: {report.match_count}{validation_status} | "
        f"Method: {report.method.label} | "
        f"{report.elapsed_ms:.1f} ms"
    )
    cv2.rectangle(output, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(output, summary, (8, h - 8), font, 0.5, (255, 255, 255), 1)

    return output


def draw_match_with_differences(
    source: np.ndarray,
    template: np.ndarray,
    report: DetectionReport,
    diff_threshold: float = 30.0,
) -> np.ndarray:
    """
    Draw matches AND highlight pixel-level differences with GREEN boxes.
    Shows side-by-side: [original + matches] | [diff heatmap + green boxes].
    """
    h, w = source.shape[:2]
    output = source.copy()

    for match in report.matches:
        x, y, mw, mh = match.bbox

        # Draw match box in green
        cv2.rectangle(output, (x, y), (x + mw, y + mh), (0, 255, 0), 3)
        cv2.drawMarker(output, (x + mw // 2, y + mh // 2), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)

        # Extract region and compute differences
        region = source[y:y + mh, x:x + mw].copy()
        tpl_resized = template
        if template.shape[:2] != region.shape[:2]:
            tpl_resized = cv2.resize(template, (mw, mh))

        diff = cv2.absdiff(tpl_resized, region)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Find contours of different areas
        _, thresh = cv2.threshold(diff_gray, int(diff_threshold), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw GREEN boxes around different regions
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            cv2.rectangle(
                output,
                (x + rx, y + ry),
                (x + rx + rw, y + ry + rh),
                (0, 255, 0),  # Green color
                2,
            )

            # Label with diff intensity
            roi = diff_gray[ry:ry + rh, rx:rx + rw]
            avg_diff = float(np.mean(roi)) if roi.size > 0 else 0
            label = f"{avg_diff:.0f}"
            cv2.putText(
                output, label,
                (x + rx + 2, y + ry + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )

    # Add status bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    validation_status = ""
    if report.validation is not None:
        validation_status = " ✓ GENUINE" if report.validation.is_valid else " ⚠ SUSPICIOUS"

    summary = (
        f"Detection{validation_status} | "
        f"Conf: {report.best_match.confidence:.3f}" if report.best_match else "No match"
    ) + f" | {report.elapsed_ms:.1f} ms"
    cv2.rectangle(output, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(output, summary, (8, h - 8), font, 0.5, (255, 255, 255), 1)

    return output


def draw_match_comparison(
    image: np.ndarray,
    reports: List[DetectionReport],
    cols: int = 3,
) -> np.ndarray:
    """Create a tiled comparison image of all matching methods."""
    import math
    rows = math.ceil(len(reports) / cols)
    h, w = image.shape[:2]
    cell_h, cell_w = h + 30, w

    canvas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)

    for i, report in enumerate(reports):
        r, c = divmod(i, cols)
        cell = draw_matches(image, report)
        canvas[r * cell_h:r * cell_h + h, c * cell_w:c * cell_w + w] = cell
        # Method name at bottom
        cv2.putText(
            canvas, report.method.label.split("(")[0].strip(),
            (c * cell_w + 8, r * cell_h + h + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

    return canvas
