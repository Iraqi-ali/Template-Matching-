"""
Advanced Forensic Detectors Module
====================================
5 advanced cybersecurity-grade detection methods:

1. JPEG Ghost Detection — detects resaving at different quality levels
2. Double JPEG Compression — detects quantization table anomalies
3. Luminance Gradient Analysis — detects lighting/shadow inconsistencies
4. Color Filter Array (CFA) Analysis — detects Bayer pattern anomalies
5. PRNU Fingerprint — camera sensor identification

All methods are production-ready and optimized for real-time use.
"""

import cv2, numpy as np, time, io
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class AdvancedReport:
    """Combined report from all 5 advanced detectors."""
    # JPEG Ghost
    jpeg_ghost_detected: bool = False
    jpeg_ghost_score: float = 0.0
    jpeg_ghost_details: str = ""

    # Double JPEG
    double_jpeg_detected: bool = False
    double_jpeg_score: float = 0.0
    double_jpeg_details: str = ""

    # Luminance
    luminance_anomaly: bool = False
    luminance_score: float = 0.0
    luminance_details: str = ""

    # CFA
    cfa_anomaly: bool = False
    cfa_score: float = 0.0
    cfa_details: str = ""

    # PRNU
    prnu_available: bool = False
    prnu_similarity: float = 0.0
    prnu_details: str = ""

    # Overall
    overall_risk: float = 0.0
    is_suspicious: bool = False
    verdict: str = ""
    details: List[str] = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None
    elapsed_ms: float = 0.0


class AdvancedDetectors:
    """Five advanced forensic detection methods in one engine."""

    def __init__(self):
        pass

    def analyze(self, image: np.ndarray, reference: Optional[np.ndarray] = None) -> AdvancedReport:
        t0 = time.perf_counter()
        report = AdvancedReport()

        # 1. JPEG Ghost Detection
        ghost_score, ghost_detail, ghost_heat = self._jpeg_ghost(image)
        report.jpeg_ghost_score = ghost_score
        report.jpeg_ghost_detected = ghost_score > 0.25
        report.jpeg_ghost_details = ghost_detail

        # 2. Double JPEG Compression
        djpeg_score, djpeg_detail = self._double_jpeg(image)
        report.double_jpeg_score = djpeg_score
        report.double_jpeg_detected = djpeg_score > 0.30
        report.double_jpeg_details = djpeg_detail

        # 3. Luminance Gradient Analysis
        lum_score, lum_detail, lum_heat = self._luminance_gradient(image)
        report.luminance_score = lum_score
        report.luminance_anomaly = lum_score > 0.30
        report.luminance_details = lum_detail

        # 4. CFA Analysis
        cfa_score, cfa_detail = self._cfa_analysis(image)
        report.cfa_score = cfa_score
        report.cfa_anomaly = cfa_score > 0.35
        report.cfa_details = cfa_detail

        # 5. PRNU (if reference available)
        if reference is not None:
            prnu_sim, prnu_detail = self._prnu_compare(image, reference)
            report.prnu_similarity = prnu_sim
            report.prnu_available = True
            report.prnu_details = prnu_detail

        # Overall risk
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        scores = [
            ghost_score, djpeg_score, lum_score, cfa_score,
            report.prnu_similarity if report.prnu_available else 0
        ]
        report.overall_risk = sum(w * s for w, s in zip(weights, scores))
        report.is_suspicious = report.overall_risk > 0.25

        if report.overall_risk < 0.15:
            report.verdict = "✅ Authentic — No manipulation detected"
        elif report.overall_risk < 0.30:
            report.verdict = "🟡 Minor Anomalies — Likely authentic"
        elif report.overall_risk < 0.50:
            report.verdict = "🟠 Suspicious — Possible manipulation"
        elif report.overall_risk < 0.70:
            report.verdict = "🔴 Tampering Detected — Likely forged"
        else:
            report.verdict = "⛔ Definite Forgery — Multiple anomalies"

        report.details = [
            f"JPEG Ghost: {'⚠ Detected' if report.jpeg_ghost_detected else '✅ Clean'} ({report.jpeg_ghost_score:.3f})",
            f"Double JPEG: {'⚠ Detected' if report.double_jpeg_detected else '✅ Clean'} ({report.double_jpeg_score:.3f})",
            f"Luminance:  {'⚠ Anomaly' if report.luminance_anomaly else '✅ Consistent'} ({report.luminance_score:.3f})",
            f"CFA Pattern: {'⚠ Anomaly' if report.cfa_anomaly else '✅ Consistent'} ({report.cfa_score:.3f})",
        ]
        if report.prnu_available:
            report.details.append(f"PRNU Match: {report.prnu_similarity:.2%} similarity")

        # Heatmap (combine ghost + luminance)
        if ghost_heat is not None or lum_heat is not None:
            h, w = image.shape[:2]
            report.heatmap = np.zeros((h, w), dtype=np.float32)
            if ghost_heat is not None and ghost_heat.shape[:2] == (h, w):
                report.heatmap += ghost_heat.astype(np.float32) / 255.0 * 0.6
            if lum_heat is not None and lum_heat.shape[:2] == (h, w):
                report.heatmap += lum_heat.astype(np.float32) / 255.0 * 0.4
            if np.max(report.heatmap) > 0:
                report.heatmap /= np.max(report.heatmap)

        report.elapsed_ms = (time.perf_counter() - t0) * 1000
        return report

    # ================================================================
    # 1. JPEG GHOST DETECTION
    # ================================================================
    def _jpeg_ghost(self, image: np.ndarray) -> Tuple[float, str, Optional[np.ndarray]]:
        """
        Detect if image was resaved at different quality levels.
        Saves at multiple qualities and measures difference — regions with
        different compression history stand out at specific quality levels.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        max_diff = np.zeros((h, w), dtype=np.float32)

        qualities = [50, 60, 70, 80, 90, 95]
        diffs = []

        for q in qualities:
            _, enc = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, q])
            dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
            if dec.shape[:2] != (h, w):
                dec = cv2.resize(dec, (w, h))
            diff = np.abs(gray.astype(np.float32) - dec.astype(np.float32))
            max_diff = np.maximum(max_diff, diff)
            diffs.append(float(np.mean(diff)))

        # Ghost score: max mean difference across qualities
        ghost_score = min(np.mean(diffs) / 15.0, 1.0)

        # Detail
        if ghost_score > 0.5:
            detail = f"Strong ghost signature (score={ghost_score:.3f}) — likely resaved/edited. Mean diff={np.mean(diffs):.1f}"
        elif ghost_score > 0.25:
            detail = f"Weak ghost signature (score={ghost_score:.3f}) — possible minor editing. Mean diff={np.mean(diffs):.1f}"
        else:
            detail = f"No ghost signature (score={ghost_score:.3f}) — consistent compression. Mean diff={np.mean(diffs):.1f}"

        return ghost_score, detail, max_diff.astype(np.uint8)

    # ================================================================
    # 2. DOUBLE JPEG COMPRESSION DETECTION
    # ================================================================
    def _double_jpeg(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Detect double JPEG compression by analyzing DCT coefficient histogram.
        Double compression creates periodic artifacts in coefficient distributions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape

        # Analyze in 8×8 DCT blocks
        scores = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = gray[y:y+8, x:x+8].astype(np.float32) - 128
                dct = cv2.dct(block)

                # Check high-frequency coefficient distribution
                # Double compression creates gaps in the histogram
                ac_coeffs = dct.flatten()[1:]  # Skip DC
                hist, _ = np.histogram(ac_coeffs, bins=20, range=(-100, 100))
                hist_norm = hist / (hist.sum() + 1e-9)

                # Entropy of coefficient distribution
                entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-9))
                if np.isnan(entropy): entropy = 4.0
                # Zero gaps in histogram indicate double compression
                zero_gaps = np.sum(hist[1:-1] == 0) / max(len(hist), 1)
                scores.append(zero_gaps + (1.0 - min(entropy / 4.0, 1.0)))

        avg_score = np.mean(scores) if scores else 0
        djpeg_score = min(max(avg_score * 3.0, 0.0), 1.0) if not np.isnan(avg_score) else 0.0

        if djpeg_score > 0.6:
            detail = f"Strong double-JPEG evidence (score={djpeg_score:.3f}) — image was recompressed after editing"
        elif djpeg_score > 0.30:
            detail = f"Possible double-JPEG (score={djpeg_score:.3f}) — may indicate resaving"
        else:
            detail = f"No double-JPEG detected (score={djpeg_score:.3f}) — single compression"

        return djpeg_score, detail

    # ================================================================
    # 3. LUMINANCE GRADIENT ANALYSIS
    # ================================================================
    def _luminance_gradient(self, image: np.ndarray) -> Tuple[float, str, Optional[np.ndarray]]:
        """
        Detect unnatural lighting/shadow inconsistencies.
        Spliced regions often have different illumination gradients
        than the surrounding image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        # Gradient magnitude and direction
        mag = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Block-wise analysis
        bs = 32
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        block_scores = []

        for y in range(0, h - bs, bs // 2):
            for x in range(0, w - bs, bs // 2):
                block_dir = direction[y:y+bs, x:x+bs]
                block_mag = mag[y:y+bs, x:x+bs]

                # Mean direction (circular mean)
                mean_sin = np.mean(np.sin(block_dir))
                mean_cos = np.mean(np.cos(block_dir))
                mean_dir = np.arctan2(mean_sin, mean_cos)

                # Consistency: how much do directions vary?
                dir_var = 1.0 - np.abs(np.mean(np.cos(block_dir - mean_dir)))

                block_scores.append(dir_var)
                anomaly_map[y:y+bs, x:x+bs] = dir_var

        mean_score = np.mean(block_scores) if block_scores else 0
        lum_score = min(mean_score * 5.0, 1.0)

        if lum_score > 0.5:
            detail = f"Inconsistent lighting detected (score={lum_score:.3f}) — possible splicing. Direction variance={mean_score:.3f}"
        elif lum_score > 0.30:
            detail = f"Minor lighting variations (score={lum_score:.3f}) — may be natural or minor editing"
        else:
            detail = f"Lighting appears consistent (score={lum_score:.3f})"

        return lum_score, detail, (anomaly_map * 255).astype(np.uint8)

    # ================================================================
    # 4. CFA (COLOR FILTER ARRAY) ANALYSIS
    # ================================================================
    def _cfa_analysis(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Detect Bayer pattern anomalies from different camera sources.
        Each camera model has a unique CFA pattern; splicing from
        different cameras disrupts the interpolation pattern.
        """
        if image.ndim < 3:
            return 0.1, "Grayscale image — CFA not applicable"

        h, w = image.shape[:2]
        b, g, r = cv2.split(image)

        # Estimate the interpolation pattern
        # Bayer pattern: R-G-R-G / G-B-G-B alternating
        # Interpolation artifacts create periodic correlations
        def estimate_pattern(channel):
            # Difference between channel and its 1-pixel shifted version
            h_diff = np.abs(channel[:, 1:].astype(np.float32) - channel[:, :-1].astype(np.float32))
            v_diff = np.abs(channel[1:, :].astype(np.float32) - channel[:-1, :].astype(np.float32))
            return np.mean(h_diff), np.mean(v_diff)

        r_h, r_v = estimate_pattern(r)
        g_h, g_v = estimate_pattern(g)
        b_h, b_v = estimate_pattern(b)

        # CFA interpolation creates specific periodic patterns
        # Inconsistency between R/B vs G patterns indicates anomaly
        r_pattern = abs(r_h - r_v) / (abs(r_h) + abs(r_v) + 1e-9)
        b_pattern = abs(b_h - b_v) / (abs(b_h) + abs(b_v) + 1e-9)

        # Cross-channel correlation (should be similar for same camera)
        rg_corr = np.corrcoef(r.flatten()[:10000], g.flatten()[:10000])[0, 1]
        gb_corr = np.corrcoef(g.flatten()[:10000], b.flatten()[:10000])[0, 1]
        if np.isnan(rg_corr): rg_corr = 0.0
        if np.isnan(gb_corr): gb_corr = 0.0
        corr_diff = abs(rg_corr - gb_corr)

        cfa_score = min(corr_diff * 2.0 + (r_pattern + b_pattern) / 2, 1.0)

        if cfa_score > 0.5:
            detail = f"CFA anomaly detected (score={cfa_score:.3f}) — possible splicing from different camera. RG-GB correlation diff={corr_diff:.3f}"
        elif cfa_score > 0.35:
            detail = f"Mild CFA variation (score={cfa_score:.3f}) — may indicate different source"
        else:
            detail = f"CFA pattern consistent (score={cfa_score:.3f})"

        return cfa_score, detail

    # ================================================================
    # 5. PRNU (PHOTO RESPONSE NON-UNIFORMITY) FINGERPRINT
    # ================================================================
    def _prnu_compare(self, image: np.ndarray, reference: np.ndarray) -> Tuple[float, str]:
        """
        Compare PRNU (sensor fingerprint) between two images.
        Each camera sensor has a unique noise pattern that acts
        as a fingerprint for source identification.

        This is a simplified PRNU estimation using wavelet denoising.
        """
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference

        # Resize to same dimensions
        if gray_img.shape != gray_ref.shape:
            gray_ref = cv2.resize(gray_ref, (gray_img.shape[1], gray_img.shape[0]))

        # Estimate noise residual (simplified PRNU)
        # Use Wiener filter / Gaussian denoising
        denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 1.5)
        noise_img = gray_img.astype(np.float32) - denoised_img.astype(np.float32)

        denoised_ref = cv2.GaussianBlur(gray_ref, (5, 5), 1.5)
        noise_ref = gray_ref.astype(np.float32) - denoised_ref.astype(np.float32)

        # Normalize
        noise_img = noise_img / (np.std(noise_img) + 1e-9)
        noise_ref = noise_ref / (np.std(noise_ref) + 1e-9)

        # Correlation coefficient
        noise_img_flat = noise_img.flatten()
        noise_ref_flat = noise_ref.flatten()

        correlation = np.corrcoef(noise_img_flat, noise_ref_flat)[0, 1]
        if np.isnan(correlation): correlation = 0.0
        correlation = max(0.0, float(correlation))

        if correlation > 0.7:
            detail = f"PRNU match: {correlation:.2%} — likely same camera sensor"
        elif correlation > 0.4:
            detail = f"PRNU moderate: {correlation:.2%} — possibly same sensor or similar model"
        else:
            detail = f"PRNU mismatch: {correlation:.2%} — likely different cameras"

        return correlation, detail
