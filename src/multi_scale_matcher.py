"""
Multi-Scale Template Matching
==============================
Searches for a template at multiple scales to handle size variations.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import time

from .template_matcher import (
    TemplateMatcher, MatchMethod, MatchResult, DetectionReport
)


class MultiScaleMatcher:
    """
    Template matching across multiple scales.

    The template is resized over a range of scales and matched at each scale,
    so the template can be found even if it appears at a different size in the
    source image.

    Usage:
        ms = MultiScaleMatcher(scale_range=(0.5, 2.0), scale_steps=20)
        report = ms.match(source_img, template_img)
    """

    def __init__(
        self,
        method: MatchMethod = MatchMethod.TM_CCOEFF_NORMED,
        threshold: float = 0.8,
        scale_range: Tuple[float, float] = (0.3, 2.5),
        scale_steps: int = 30,
        use_nms: bool = True,
        nms_threshold: float = 0.3,
    ):
        """
        Args:
            method: Matching method.
            threshold: Minimum confidence (0-1).
            scale_range: (min_scale, max_scale) – e.g. (0.5, 2.0) searches
                         from half size to double size.
            scale_steps: Number of scale increments (more = finer, slower).
            use_nms: Apply NMS across all scales.
            nms_threshold: IoU threshold for NMS.
        """
        self.method = method
        self.threshold = threshold
        self.scale_range = scale_range
        self.scale_steps = scale_steps
        self.matcher = TemplateMatcher(
            method=method, threshold=threshold,
            use_nms=False,  # NMS applied globally after all scales
        )
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
        Multi-scale template matching.

        Returns:
            DetectionReport with matches consolidated across all scales.
        """
        thresh = threshold if threshold is not None else self.threshold
        src_gray = self._to_gray(source)
        tpl_gray = self._to_gray(template)

        t0 = time.perf_counter()

        scales = np.linspace(self.scale_range[0], self.scale_range[1], self.scale_steps)
        all_matches: List[MatchResult] = []

        for scale in scales:
            # Resize template
            new_w = int(tpl_gray.shape[1] * scale)
            new_h = int(tpl_gray.shape[0] * scale)

            # Skip invalid sizes
            if new_w < 10 or new_h < 10:
                continue
            if new_w > src_gray.shape[1] or new_h > src_gray.shape[0]:
                continue

            scaled_tpl = cv2.resize(tpl_gray, (new_w, new_h))

            # Run single-scale match
            result_map = cv2.matchTemplate(src_gray, scaled_tpl, self.method.value)

            matches = self.matcher._extract_matches(
                result_map, scaled_tpl.shape[1], scaled_tpl.shape[0], thresh
            )
            for m in matches:
                m.scale = scale
            all_matches.extend(matches)

        # Global NMS across all scales
        if self.use_nms and len(all_matches) > 1:
            all_matches = self.matcher._apply_nms(all_matches)

        # Sort and limit
        all_matches.sort(key=lambda m: m.confidence, reverse=True)
        all_matches = all_matches[:max_matches]

        elapsed = (time.perf_counter() - t0) * 1000.0

        return DetectionReport(
            matches=all_matches,
            source_shape=(source.shape[1], source.shape[0]),
            template_shape=(template.shape[1], template.shape[0]),
            method=self.method,
            threshold=thresh,
            elapsed_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
