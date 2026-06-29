"""
Document Authenticity Scorer (DAS)
===================================
Combines ALL forensic analyses into a weighted 0-100% authenticity score.
Calibrated for real-world forensic use with configurable weights.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class AuthenticityReport:
    overall_score: float = 100.0  # 0% (forged) → 100% (authentic)
    verdict: str = "Authentic"
    category_scores: Dict[str, float] = field(default_factory=dict)
    weighted_breakdown: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class AuthenticityScorer:
    WEIGHTS = {
        'document_forensics': 0.30,
        'forgery_detection': 0.20,
        'signature_verification': 0.15,
        'font_consistency': 0.10,
        'copy_move': 0.10,
        'metadata': 0.10,
        'fingerprint_match': 0.05,
    }

    def compute(
        self,
        forensics_report: Optional[any] = None,
        forgery_report: Optional[any] = None,
        signature_report: Optional[any] = None,
        font_report: Optional[any] = None,
        copy_move_report: Optional[any] = None,
        metadata_report: Optional[any] = None,
        fingerprint_match: Optional[Dict] = None,
    ) -> AuthenticityReport:
        report = AuthenticityReport()
        total_weight = 0.0; weighted_sum = 0.0

        def add_score(name, score, weight_key, label_ar):
            nonlocal total_weight, weighted_sum
            w = self.WEIGHTS[weight_key]
            report.category_scores[name] = round(score, 1)
            report.weighted_breakdown.append(f"{label_ar}: {score:.1f}% × {w:.0%}")
            weighted_sum += score * w; total_weight += w

        # 1. Document Forensics
        if forensics_report is not None:
            score = (1.0 - getattr(forensics_report, 'tamper_score', 0.0)) * 100
            add_score('document_forensics', score, 'document_forensics', 'الفحص الجنائي (5 طرق)')
            if getattr(forensics_report, 'is_tampered', False):
                report.risk_factors.append("تزوير مثبت بالفحص الجنائي")

        # 2. Forgery Detection
        if forgery_report is not None:
            score = (1.0 - getattr(forgery_report, 'risk_score', 0.0)) * 100
            add_score('forgery_detection', score, 'forgery_detection', 'كشف التزوير (ELA/SSIM)')

        # 3. Signature
        if signature_report is not None:
            score = getattr(signature_report, 'confidence', 1.0) * 100
            add_score('signature', score, 'signature_verification', 'التحقق من التوقيع')
            if getattr(signature_report, 'confidence', 1.0) < 0.5:
                report.risk_factors.append("توقيع مشبوه أو مزور")

        # 4. Font
        if font_report is not None:
            score = getattr(font_report, 'confidence', 1.0) * 100
            add_score('font', score, 'font_consistency', 'تناسق الخطوط')
            if not getattr(font_report, 'is_consistent', True):
                report.risk_factors.append("عدم تناسق في الخطوط")

        # 5. Copy-Move
        if copy_move_report is not None:
            score = (1.0 - getattr(copy_move_report, 'confidence', 0.0)) * 100
            add_score('copy_move', score, 'copy_move', 'كشف النسخ واللصق')
            if getattr(copy_move_report, 'has_clones', False):
                report.risk_factors.append("اكتشاف نسخ ولصق داخل المستند")

        # 6. Metadata
        if metadata_report is not None:
            score = (1.0 - getattr(metadata_report, 'anomaly_score', 0.0)) * 100
            add_score('metadata', score, 'metadata', 'تحليل Metadata')
            if getattr(metadata_report, 'is_suspicious', False):
                report.risk_factors.append("شذوذ في بيانات metadata")

        # 7. Fingerprint
        if fingerprint_match and fingerprint_match.get('matched'):
            score = fingerprint_match.get('similarity', 0) * 100
            add_score('fingerprint', score, 'fingerprint_match', 'مطابقة البصمة')
        elif fingerprint_match:
            total_weight += self.WEIGHTS['fingerprint_match']

        report.overall_score = round(weighted_sum / total_weight, 1) if total_weight > 0 else 50.0

        if report.overall_score >= 90: report.verdict = "✅ أصلي — Authentic"
        elif report.overall_score >= 70: report.verdict = "🟡 غالباً أصلي — Likely Authentic"
        elif report.overall_score >= 50: report.verdict = "🟠 مشكوك فيه — Questionable"
        elif report.overall_score >= 30: report.verdict = "🔴 مشبوه — Suspicious"
        else: report.verdict = "⛔ مزور — Forged"

        if report.risk_factors: report.recommendations.append("⚠ يوصى بمراجعة يدوية من قبل خبير جنائي")
        if report.overall_score < 70: report.recommendations.append("📋 يوصى بجمع المزيد من الأدلة والمقارنات")
        if not report.risk_factors: report.recommendations.append("✅ لم يتم اكتشاف عوامل خطر — المستند يبدو أصلياً")
        return report
