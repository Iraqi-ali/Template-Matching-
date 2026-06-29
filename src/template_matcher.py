"""
Template Matching Engine - Core Module
======================================
Finds a template image within a source image using various matching methods.
Supports single-scale and multi-scale template matching.

Matching Methods (OpenCV):
    TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED,
    TM_SQDIFF, TM_SQDIFF_NORMED
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
class DetectionReport:
    """Full report of template matching detection."""
    matches: List[MatchResult] = field(default_factory=list)
    source_shape: Tuple[int, int] = (0, 0)
    template_shape: Tuple[int, int] = (0, 0)
    method: MatchMethod = MatchMethod.TM_CCOEFF_NORMED
    threshold: float = 0.8
    elapsed_ms: float = 0.0

    @property
    def match_count(self) -> int:
        return len(self.matches)

    @property
    def best_match(self) -> Optional[MatchResult]:
        return self.matches[0] if self.matches else None


class TemplateMatcher:
    """
    Core template matching engine.

    Usage:
        matcher = TemplateMatcher(method=MatchMethod.TM_CCOEFF_NORMED, threshold=0.8)
        result = matcher.match(source_img, template_img, threshold=0.8)
        for m in result.matches:
            print(f"Found at ({m.x}, {m.y}) with confidence {m.confidence:.2f}")
    """

    def __init__(
        self,
        method: MatchMethod = MatchMethod.TM_CCOEFF_NORMED,
        threshold: float = 0.8,
        use_nms: bool = True,
        nms_threshold: float = 0.3,
    ):
        """
        Args:
            method: OpenCV template matching method.
            threshold: Minimum confidence to accept a match (0.0 - 1.0).
            use_nms: Apply Non-Maximum Suppression to remove duplicate detections.
            nms_threshold: IoU threshold for NMS.
        """
        self.method = method
        self.threshold = threshold
        self.use_nms = use_nms
        self.nms_threshold = nms_threshold

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

        Args:
            source: Source image (BGR or grayscale).
            template: Template image to search for (BGR or grayscale).
            threshold: Override instance threshold.
            max_matches: Maximum number of matches to return.

        Returns:
            DetectionReport with all valid matches.
        """
        thresh = threshold if threshold is not None else self.threshold
        src_gray = self._to_gray(source)
        tpl_gray = self._to_gray(template)

        import time
        t0 = time.perf_counter()

        # Run OpenCV matchTemplate
        result_map = cv2.matchTemplate(src_gray, tpl_gray, self.method.value)

        # Extract match locations above threshold
        matches = self._extract_matches(
            result_map, tpl_gray.shape[1], tpl_gray.shape[0], thresh
        )

        # Non-Maximum Suppression
        if self.use_nms and len(matches) > 1:
            matches = self._apply_nms(matches)

        # Sort by confidence & limit
        matches.sort(key=lambda m: m.confidence, reverse=True)
        matches = matches[:max_matches]

        elapsed = (time.perf_counter() - t0) * 1000.0

        return DetectionReport(
            matches=matches,
            source_shape=(source.shape[1], source.shape[0]),
            template_shape=(template.shape[1], template.shape[0]),
            method=self.method,
            threshold=thresh,
            elapsed_ms=elapsed,
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
            # For SQDIFF, values below threshold are better matches
            loc = np.where(result_map <= (1.0 - threshold))
            # Invert for confidence scoring
            confidences = 1.0 - result_map[loc]
        else:
            loc = np.where(result_map >= threshold)
            confidences = result_map[loc]

        matches = []
        for pt_y, pt_x in zip(*loc):
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

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            score_threshold=0.0,
            nms_threshold=self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        return [matches[i] for i in indices.flatten()]


# ------------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------------

def draw_matches(
    image: np.ndarray,
    report: DetectionReport,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw bounding boxes for all matches on the image."""
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, match in enumerate(report.matches):
        x, y, w, h = match.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        cv2.drawMarker(output, match.center, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

        if show_confidence:
            label = f"#{i+1} {match.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(output, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(output, label, (x + 2, y - 4), font, 0.5, (0, 0, 0), 1)

    # Summary overlay
    summary = f"Matches: {report.match_count} | Method: {report.method.label} | {report.elapsed_ms:.1f} ms"
    h, w = output.shape[:2]
    cv2.rectangle(output, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(output, summary, (8, h - 8), font, 0.55, (255, 255, 255), 1)

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
