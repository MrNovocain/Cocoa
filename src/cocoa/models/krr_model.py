import pandas as pd
import numpy as np
from typing import Any, Dict

from sklearn.kernel_ridge import KernelRidge

from .assets import KRR_PARAM_GRID
from .ml_models import BaseSklearnModel


class KRRModel(BaseSklearnModel):
    """A wrapper for the scikit-learn KernelRidge."""

    def __init__(
        self,
        alpha: float = KRR_PARAM_GRID["alpha"][0],
        kernel: str = KRR_PARAM_GRID["kernel"][0],
        gamma: float | None = KRR_PARAM_GRID.get("gamma", [None])[0],
        degree: int | None = KRR_PARAM_GRID.get("degree", [3])[0],
        **kwargs: Any
    ):
        hyperparams = {
            "alpha": alpha,
            "kernel": kernel,
            **kwargs
        }
        # Add kernel-specific params only if they are relevant and provided
        if kernel in ["rbf", "laplacian", "poly", "sigmoid"]:
            if gamma is not None:
                hyperparams["gamma"] = gamma
        if kernel == "poly":
            if degree is not None:
                hyperparams["degree"] = degree

        super().__init__(model_class=KernelRidge, **hyperparams)