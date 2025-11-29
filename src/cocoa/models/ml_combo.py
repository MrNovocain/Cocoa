import pandas as pd
import numpy as np
from typing import Any, Dict, Type

from .combo_base import BaseConvexCombinationModel
from .base_model import BaseModel


class MLConvexCombinationModel(BaseConvexCombinationModel):
    """
    A convex combination of two machine learning models (e.g., RF, XGBoost).

    This class uses the BaseConvexCombinationModel to combine two ML models,
    one trained on pre-break data and one on post-break data.
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        params_pre: Dict[str, Any],
        params_post: Dict[str, Any],
        break_index: int,
        gamma: float,
    ):
        super().__init__(break_index=break_index, gamma=gamma)

        self.model_class = model_class
        self.params_pre = params_pre
        self.params_post = params_post

        self.hyperparams = {
            'gamma': gamma,
            'break_index': break_index,
            'params_pre': params_pre,
            'params_post': params_post,
        }

    def _initialize_sub_models(self) -> None:
        """Instantiates the pre- and post-break ML models."""
        self.model_pre = self.model_class(**self.params_pre)
        self.model_post = self.model_class(**self.params_post)