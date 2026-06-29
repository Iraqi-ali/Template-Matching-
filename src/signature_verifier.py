"""
Signature Verification & Analysis Module
=========================================
Detects forged signatures vs original using geometric and texture analysis.

Methods:
  1. Contour Geometry — signature shape, aspect ratio, bounding area
  2. Stroke Density — ink distribution pattern
  3. Skeleton/Topology — centerline matching
  4. Hu Moments — rotation/scale invariant shape descriptors
  5. Feature Point Matching (ORB) — keypoint correspondence
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class SignatureVerdict(Enum):
    GENUINE = ("Genuine Signature", (0, 255, 0))
    LIKELY_GENUINE = ("Likely Genuine", (0, 200, 100))
    INCONCLUSIVE = ("Inconclusive — Needs Manual Review", (0, 165, 255))
    SUSPICIOUS = ("Suspicious — Possible Forgery", (0, 100, 255))
    FORGED = ("Forged Signature", (0, 0, 255))

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def color(self) -> Tuple[int, int, int]:
        return self.value[1]


@dataclass
class SignatureReport:
    """Complete signature analysis report."""
    verdict: SignatureVerdict = SignatureVerdict.INCONCLUSIVE
    confidence: float = 0.0
    geometry_similarity: float = 0.0
    density_similarity: float = 0.0
    topology_similarity: float = 0.0
    hu_moments_similarity: float = 0.0
    orb_match_score: float = 0.0
    details: List[str] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    elapsed_ms: float = 0.0


class SignatureVerifier:
    """
    Professional signature forgery detection.

    Usage:
        verifier = SignatureVerifier()
        report = verifier.verify(original_signature, suspect_signature)
        print(report.verdict.label)
    """

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500)

    def verify(
        self,
        original: np.ndarray,
        suspect: np.ndarray,
    ) -> SignatureReport:
        """Run complete signature verification."""
        import time
        t0 = time.perf_counter()

        report = SignatureReport()

        # Preprocess: extract signature region
        orig_sig = self._extract_signature_region(original)
        susp_sig = self._extract_signature_region(suspect)

        # Resize suspect to match original if needed
        if orig_sig.shape[:2] != susp_sig.shape[:2]:
            susp_sig = cv2.resize(susp_sig, (orig_sig.shape[1], orig_sig.shape[0]))

        # Binarize
        orig_bin = self._binarize(orig_sig)
        susp_bin = self._binarize(susp_sig)

        # 1. Geometry Analysis
        report.geometry_similarity = self._geometry_analysis(orig_bin, susp_bin)

        # 2. Stroke Density
        report.density_similarity = self._density_analysis(orig_bin, susp_bin)

        # 3. Topology (Skeleton)
        report.topology_similarity = self._topology_analysis(orig_bin, susp_bin)

        # 4. Hu Moments
        report.hu_moments_similarity = self._hu_moments_analysis(orig_bin, susp_bin)

        # 5. ORB Feature Matching
        report.orb_match_score = self._orb_analysis(orig_sig, susp_sig)

        # Ensemble verdict
        scores = [
            report.geometry_similarity * 0.25,
            report.density_similarity * 0.25,
            report.topology_similarity * 0.20,
            report.hu_moments_similarity * 0.15,
            report.orb_match_score * 0.15,
        ]
        report.confidence = sum(scores)

        # Classify
        if report.confidence >= 0.85:
            report.verdict = SignatureVerdict.GENUINE
        elif report.confidence >= 0.70:
            report.verdict = SignatureVerdict.LIKELY_GENUINE
        elif report.confidence >= 0.50:
            report.verdict = SignatureVerdict.INCONCLUSIVE
        elif report.confidence >= 0.30:
            report.verdict = SignatureVerdict.SUSPICIOUS
        else:
            report.verdict = SignatureVerdict.FORGED

        # Generate details
        report.details = self._generate_details(report)

        # Annotate
        report.annotated_image = self._annotate_signature(suspect, orig_sig, report)

        report.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return report

    # ==================================================================
    # Preprocessing
    # ==================================================================

    def _extract_signature_region(self, img: np.ndarray) -> np.ndarray:
        """Extract the main signature area from the image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find largest contour (likely signature)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img

        # Get bounding box of all contours combined
        all_points = np.vstack([c for c in contours if cv2.contourArea(c) > 20])
        if len(all_points) == 0:
            return img

        x, y, w, h = cv2.boundingRect(all_points)
        x, y = max(0, x - 10), max(0, y - 10)
        w, h = min(w + 20, img.shape[1] - x), min(h + 20, img.shape[0] - y)

        return img[y:y + h, x:x + w].copy()

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """Convert image to binary."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    # ==================================================================
    # Analysis Methods
    # ==================================================================

    def _geometry_analysis(self, orig_bin: np.ndarray, susp_bin: np.ndarray) -> float:
        """Compare geometric properties of signatures."""
        # Find contours
        cnt_orig, _ = cv2.findContours(orig_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_susp, _ = cv2.findContours(susp_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnt_orig or not cnt_susp:
            return 0.0

        # Combine all contours
        all_orig = np.vstack(cnt_orig)
        all_susp = np.vstack(cnt_susp)

        # Bounding rectangles
        ox, oy, ow, oh = cv2.boundingRect(all_orig)
        sx, sy, sw, sh = cv2.boundingRect(all_susp)

        # Aspect ratio similarity
        orig_ratio = ow / max(oh, 1)
        susp_ratio = sw / max(sh, 1)
        ratio_sim = 1.0 - min(abs(orig_ratio - susp_ratio) / max(orig_ratio, 0.01), 1.0)

        # Area similarity
        orig_area = cv2.contourArea(all_orig) if len(all_orig) > 0 else 0
        susp_area = cv2.contourArea(all_susp) if len(all_susp) > 0 else 0
        area_sim = 1.0 - min(
            abs(orig_area - susp_area) / max(orig_area, 1), 1.0
        )

        # Convex hull similarity
        hull_orig = cv2.convexHull(all_orig)
        hull_susp = cv2.convexHull(all_susp)
        hull_area_orig = cv2.contourArea(hull_orig)
        hull_area_susp = cv2.contourArea(hull_susp)
        hull_sim = 1.0 - min(
            abs(hull_area_orig - hull_area_susp) / max(hull_area_orig, 1), 1.0
        )

        return float(np.mean([ratio_sim, area_sim, hull_sim]))

    def _density_analysis(self, orig_bin: np.ndarray, susp_bin: np.ndarray) -> float:
        """Compare ink/stroke density distribution."""
        # Horizontal projection
        h_proj_orig = np.sum(orig_bin, axis=0) / 255.0
        h_proj_susp = np.sum(susp_bin, axis=0) / 255.0

        # Normalize
        if np.max(h_proj_orig) > 0:
            h_proj_orig = h_proj_orig / np.max(h_proj_orig)
        if np.max(h_proj_susp) > 0:
            h_proj_susp = h_proj_susp / np.max(h_proj_susp)

        # Resize to same length
        if len(h_proj_orig) != len(h_proj_susp):
            target_len = max(len(h_proj_orig), len(h_proj_susp))
            h_proj_orig = np.interp(
                np.linspace(0, len(h_proj_orig) - 1, target_len),
                np.arange(len(h_proj_orig)), h_proj_orig
            )
            h_proj_susp = np.interp(
                np.linspace(0, len(h_proj_susp) - 1, target_len),
                np.arange(len(h_proj_susp)), h_proj_susp
            )

        # Correlation
        correlation = np.corrcoef(h_proj_orig, h_proj_susp)[0, 1]
        correlation = max(0.0, correlation) if not np.isnan(correlation) else 0.0

        # Density ratio
        orig_density = np.sum(orig_bin > 0) / orig_bin.size
        susp_density = np.sum(susp_bin > 0) / susp_bin.size
        density_ratio = min(orig_density, susp_density) / max(orig_density, susp_density, 0.001)

        return float(np.mean([correlation, density_ratio]))

    def _topology_analysis(self, orig_bin: np.ndarray, susp_bin: np.ndarray) -> float:
        """Compare signature topology via skeletonization."""
        # Skeletonize (thinning)
        orig_skel = self._skeletonize(orig_bin)
        susp_skel = self._skeletonize(susp_bin)

        # Distance transform comparison
        dt_orig = cv2.distanceTransform(255 - orig_skel, cv2.DIST_L2, 5)
        dt_susp = cv2.distanceTransform(255 - susp_skel, cv2.DIST_L2, 5)

        # Normalize
        if np.max(dt_orig) > 0:
            dt_orig = dt_orig / np.max(dt_orig)
        if np.max(dt_susp) > 0:
            dt_susp = dt_susp / np.max(dt_susp)

        # Mean absolute difference
        diff = np.abs(dt_orig - dt_susp)
        mean_diff = np.mean(diff)
        similarity = max(0.0, 1.0 - mean_diff * 2.0)

        # Branch point count similarity
        orig_branches = self._count_branch_points(orig_skel)
        susp_branches = self._count_branch_points(susp_skel)
        max_branches = max(orig_branches, susp_branches, 1)
        branch_sim = 1.0 - abs(orig_branches - susp_branches) / max_branches

        return float(np.mean([similarity, branch_sim]))

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """Zhang-Suen thinning algorithm."""
        skel = binary.copy() // 255
        skel = skel.astype(np.uint8)
        prev = np.zeros(skel.shape, np.uint8)

        while True:
            # Sub-iteration 1
            eroded = cv2.erode(skel, np.ones((3, 3), np.uint8))
            opened = cv2.dilate(eroded, np.ones((3, 3), np.uint8))
            subset = skel - opened
            prev = skel.copy()
            skel = skel - subset

            # Check convergence
            if np.array_equal(skel, prev):
                break

        return skel * 255

    def _count_branch_points(self, skeleton: np.ndarray) -> int:
        """Count branch points in a skeleton."""
        skel = (skeleton > 0).astype(np.uint8)
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        neighbors = cv2.filter2D(skel, -1, kernel)
        # Branch points have 3+ neighbors
        branch_points = np.sum((neighbors >= 13) & (skel > 0))
        return int(branch_points)

    def _hu_moments_analysis(self, orig_bin: np.ndarray, susp_bin: np.ndarray) -> float:
        """Compare Hu Moments (invariant to scale, rotation, translation)."""
        # Find contours
        cnt_orig, _ = cv2.findContours(orig_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_susp, _ = cv2.findContours(susp_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnt_orig or not cnt_susp:
            return 0.0

        all_orig = np.vstack(cnt_orig)
        all_susp = np.vstack(cnt_susp)

        hu_orig = cv2.HuMoments(cv2.moments(all_orig)).flatten()
        hu_susp = cv2.HuMoments(cv2.moments(all_susp)).flatten()

        # Log scale
        hu_orig = -np.sign(hu_orig) * np.log10(np.abs(hu_orig) + 1e-10)
        hu_susp = -np.sign(hu_susp) * np.log10(np.abs(hu_susp) + 1e-10)

        # Normalized difference
        diff = np.abs(hu_orig - hu_susp)
        norm_diff = np.sum(diff) / (np.sum(np.abs(hu_orig)) + np.sum(np.abs(hu_susp)) + 1e-9)
        similarity = max(0.0, 1.0 - norm_diff)

        return float(similarity)

    def _orb_analysis(self, orig_rgb: np.ndarray, susp_rgb: np.ndarray) -> float:
        """ORB feature point matching for detailed comparison."""
        gray_orig = cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2GRAY) if orig_rgb.ndim == 3 else orig_rgb
        gray_susp = cv2.cvtColor(susp_rgb, cv2.COLOR_BGR2GRAY) if susp_rgb.ndim == 3 else susp_rgb

        kp1, des1 = self.orb.detectAndCompute(gray_orig, None)
        kp2, des2 = self.orb.detectAndCompute(gray_susp, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.5  # Neutral if not enough features

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) == 0:
            return 0.1

        # Good matches ratio
        matches = sorted(matches, key=lambda m: m.distance)
        good_matches = [m for m in matches if m.distance < 50]
        match_ratio = len(good_matches) / max(len(kp1), 1)

        # Average distance (lower = better)
        avg_dist = np.mean([m.distance for m in good_matches]) if good_matches else 100
        dist_score = max(0.0, 1.0 - avg_dist / 100.0)

        return float(np.mean([match_ratio, dist_score]))

    # ==================================================================
    # Annotation & Details
    # ==================================================================

    def _annotate_signature(
        self, suspect: np.ndarray, signature_region: np.ndarray, report: SignatureReport
    ) -> np.ndarray:
        """Annotate suspect image with signature analysis overlay."""
        output = suspect.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw signature region box
        if signature_region.shape[:2] != suspect.shape[:2]:
            h, w = output.shape[:2]
            margin = 20
            cv2.rectangle(output, (margin, margin),
                          (w - margin, h - margin),
                          report.verdict.color, 3)

        # Verdict banner
        banner_h = 40
        cv2.rectangle(output, (0, 0), (output.shape[1], banner_h),
                      report.verdict.color, -1)
        cv2.putText(output,
                    f"SIGNATURE: {report.verdict.label} ({report.confidence:.1%})",
                    (10, banner_h - 12), font, 0.55, (255, 255, 255), 1)

        # Scores panel
        y_off = banner_h + 5
        scores_text = [
            f"Geometry:    {report.geometry_similarity:.3f}",
            f"Density:     {report.density_similarity:.3f}",
            f"Topology:    {report.topology_similarity:.3f}",
            f"Hu Moments:  {report.hu_moments_similarity:.3f}",
            f"ORB Match:   {report.orb_match_score:.3f}",
        ]
        for text in scores_text:
            y_off += 18
            cv2.putText(output, text, (10, y_off), font, 0.4, (200, 200, 200), 1)

        return output

    def _generate_details(self, report: SignatureReport) -> List[str]:
        """Generate human-readable analysis details."""
        details = []
        verdict = report.verdict

        if verdict == SignatureVerdict.GENUINE:
            details.append("✅ Signature matches original with high confidence")
        elif verdict == SignatureVerdict.LIKELY_GENUINE:
            details.append("✓ Signature likely matches original")
        elif verdict == SignatureVerdict.INCONCLUSIVE:
            details.append("⚠ Insufficient evidence — manual review recommended")
        elif verdict == SignatureVerdict.SUSPICIOUS:
            details.append("⚠ Signature shows suspicious deviations from original")
        else:
            details.append("❌ Signature appears FORGED — significant deviations detected")

        if report.geometry_similarity < 0.6:
            details.append("• Geometric proportions differ significantly")
        if report.density_similarity < 0.6:
            details.append("• Stroke density pattern inconsistent with original")
        if report.topology_similarity < 0.6:
            details.append("• Skeleton topology shows major differences")
        if report.hu_moments_similarity < 0.6:
            details.append("• Shape moments deviate from expected values")
        if report.orb_match_score < 0.5:
            details.append("• Insufficient feature point matches found")

        return details
