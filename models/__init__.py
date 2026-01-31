"""
Model implementations for scalar PV predictors.
"""
from .voc_nn import (
    VocNN, VocNNConfig, VocTrainer, build_voc_model,
    SplitSplineNetConfig, UnifiedSplitSplineNet, physics_projection, predict_with_uncertainty
)
from .jsc_lgbm import JscLGBM, JscLGBMConfig, build_jsc_model
from .vmpp_lgbm import (
    VmppLGBM, VmppLGBMConfig, JmppLGBM, FFLGBM,
    build_vmpp_model, build_jmpp_model, build_ff_model
)
from .reconstruction import reconstruct_curve, continuity_loss
from .cvae import ConditionalVAE, cvae_loss
from .direct_curve import (
    DirectCurveNet, DirectCurveNetConfig,
    DirectCurveNetWithJsc, DirectCurveNetWithJscConfig,
    DirectCurveShapeNet, DirectCurveShapeNetConfig,
    DirectCurveLoss, DirectCurveLossWithJsc, DirectCurveShapeLoss,
    reconstruct_curve_direct, reconstruct_curve_direct_normalized,
    reconstruct_curve_shape,
    extract_voc_from_curve
)

__all__ = [
    'VocNN', 'VocNNConfig', 'VocTrainer', 'build_voc_model',
    'SplitSplineNetConfig', 'UnifiedSplitSplineNet', 'physics_projection', 'predict_with_uncertainty',
    'JscLGBM', 'JscLGBMConfig', 'build_jsc_model',
    'VmppLGBM', 'VmppLGBMConfig', 'JmppLGBM', 'FFLGBM',
    'build_vmpp_model', 'build_jmpp_model', 'build_ff_model',
    'reconstruct_curve', 'continuity_loss',
    'ConditionalVAE', 'cvae_loss',
    # Direct curve prediction (simplified, no Vmpp split)
    'DirectCurveNet', 'DirectCurveNetConfig',
    'DirectCurveNetWithJsc', 'DirectCurveNetWithJscConfig',
    'DirectCurveShapeNet', 'DirectCurveShapeNetConfig',  # RECOMMENDED
    'DirectCurveLoss', 'DirectCurveLossWithJsc', 'DirectCurveShapeLoss',
    'reconstruct_curve_direct', 'reconstruct_curve_direct_normalized',
    'reconstruct_curve_shape',
    'extract_voc_from_curve',
]
