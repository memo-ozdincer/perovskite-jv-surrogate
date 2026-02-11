"""
Scalar PV Predictors - Ultra-fast physics-informed ML models for solar cell parameters.

Main components:
- VocNN: Neural network for Voc prediction with Jacobian regularization
- JscLGBM: LightGBM for Jsc prediction with analytical ceiling
- VmppLGBM: LightGBM for Vmpp prediction
- FFLGBM: LightGBM for Fill Factor prediction

Usage:
    from scalar_predictors import ScalarPredictor
    predictor = ScalarPredictor('outputs/models')
    result = predictor.predict(params)
"""
from .inference import ScalarPredictor, BatchPredictor, PredictionResult, load_predictor

__version__ = '1.0.0'
__all__ = ['ScalarPredictor', 'BatchPredictor', 'PredictionResult', 'load_predictor']
