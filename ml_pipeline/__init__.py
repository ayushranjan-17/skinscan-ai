"""
SkinScan AI Machine Learning Pipeline Package.
Contains DullRazor preprocessing, Multi-Modal DenseNet121 architecture,
Focal Loss, Grad-CAM extraction, and Clinical Evaluation metrics.
"""

from .risk_stratification import map_to_4_tier_risk
from .dullrazor import preprocess_dermatoscopy_pipeline, apply_dullrazor, apply_color_constancy
from .evaluate_metrics import calculate_clinical_metrics

__all__ = [
    'map_to_4_tier_risk',
    'preprocess_dermatoscopy_pipeline',
    'apply_dullrazor',
    'apply_color_constancy',
    'calculate_clinical_metrics'
]
