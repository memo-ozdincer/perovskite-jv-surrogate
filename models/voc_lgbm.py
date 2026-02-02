"""
LightGBM model for Voc prediction with analytical ceiling feature.
"""
import numpy as np
import lightgbm as lgb
from dataclasses import dataclass


@dataclass
class VocLGBMConfig:
    """
    Configuration for Voc LightGBM model.

    UPDATED v2.0: Defaults re-centered to MIDDLE of HPO search space
    to allow HPO to explore both directions effectively.
    HPO search ranges: num_leaves [31,255], max_depth [6,15], lr [0.01,0.1]
    """
    # Core parameters - CENTERED in search space (not at upper bounds)
    num_leaves: int = 127         # Middle of [31, 255]
    max_depth: int = 10           # Middle of [6, 15]
    learning_rate: float = 0.03   # Geometric mean of [0.01, 0.1]
    n_estimators: int = 1000      # Middle of [500, 2000]
    min_child_samples: int = 30   # Middle of [10, 50]
    subsample: float = 0.82       # Middle of [0.7, 0.95]
    colsample_bytree: float = 0.82
    reg_alpha: float = 0.01       # Geometric mean of [1e-4, 1.0]
    reg_lambda: float = 0.01

    # GPU settings
    device: str = 'gpu'
    gpu_platform_id: int = 0
    gpu_device_id: int = 0

    # Training settings
    early_stopping_rounds: int = 50
    verbose: int = -1
    n_jobs: int = -1
    random_state: int = 42

    # Feature settings
    use_ceiling_feature: bool = True
    use_physics_features: bool = True

    def to_lgb_params(self) -> dict:
        """Convert to LightGBM parameter dict."""
        return {
            'objective': 'regression',
            'metric': ['rmse', 'mae'],
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'device': self.device,
            'gpu_platform_id': self.gpu_platform_id,
            'gpu_device_id': self.gpu_device_id,
            'verbose': self.verbose,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'force_col_wise': True,
        }


class VocLGBM:
    """
    LightGBM model for Voc with physics-informed features.

    Predicts Voc / Voc_ceiling ratio to enforce physical bounds.
    """

    def __init__(self, config: VocLGBMConfig):
        self.config = config
        self.model = None

    def _prepare_features(
        self,
        raw_params: np.ndarray,
        physics_features: np.ndarray,
        voc_ceiling: np.ndarray
    ) -> np.ndarray:
        features_list = [raw_params]

        if self.config.use_physics_features:
            features_list.append(physics_features)

        if self.config.use_ceiling_feature:
            log_ceiling = np.log10(np.abs(voc_ceiling) + 1e-30).reshape(-1, 1)
            features_list.append(log_ceiling)

        return np.hstack(features_list)

    def _prepare_target(self, voc: np.ndarray, voc_ceiling: np.ndarray) -> np.ndarray:
        """
        Prepare target as ratio to ceiling.

        UPDATED v2.0: Extended clip range from [0, 2] to [0, 3] to avoid
        clipping extreme outlier samples which can cause gradient issues.
        """
        ratio = voc / (np.abs(voc_ceiling) + 1e-30)
        return np.clip(ratio, 0, 3.0)

    def _inverse_target(self, ratio: np.ndarray, voc_ceiling: np.ndarray) -> np.ndarray:
        return ratio * np.abs(voc_ceiling)

    def fit(
        self,
        X_raw: np.ndarray,
        X_physics: np.ndarray,
        y_voc: np.ndarray,
        voc_ceiling: np.ndarray,
        X_raw_val: np.ndarray = None,
        X_physics_val: np.ndarray = None,
        y_voc_val: np.ndarray = None,
        voc_ceiling_val: np.ndarray = None
    ) -> dict:
        X_train = self._prepare_features(X_raw, X_physics, voc_ceiling)
        y_train = self._prepare_target(y_voc, voc_ceiling)

        train_data = lgb.Dataset(X_train, label=y_train)

        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]

        valid_sets = [train_data]
        valid_names = ['train']

        if X_raw_val is not None:
            X_val = self._prepare_features(X_raw_val, X_physics_val, voc_ceiling_val)
            y_val = self._prepare_target(y_voc_val, voc_ceiling_val)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        self.model = lgb.train(
            self.config.to_lgb_params(),
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        return {'best_iteration': self.model.best_iteration}

    def predict(
        self,
        X_raw: np.ndarray,
        X_physics: np.ndarray,
        voc_ceiling: np.ndarray
    ) -> np.ndarray:
        X = self._prepare_features(X_raw, X_physics, voc_ceiling)
        ratio = self.model.predict(X)
        return self._inverse_target(ratio, voc_ceiling)

    def save(self, path: str):
        """Save model to file."""
        self.model.save_model(path)

    def load(self, path: str):
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)


def build_voc_model(config: VocLGBMConfig) -> VocLGBM:
    """Factory function."""
    return VocLGBM(config)
