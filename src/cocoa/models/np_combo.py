
import pandas as pd
import numpy as np

from .combo_base import BaseConvexCombinationModel
from .np_regime import NPRegimeModel
from .np_base import BaseKernel, BaseLocalEngine

class NPConvexCombinationModel(BaseConvexCombinationModel):
    """
    A convex combination of two non-parametric (NPRegimeModel) models.

    The final prediction is a weighted average:
    y_pred = gamma * pred_pre + (1 - gamma) * pred_post
    """

    def __init__(
        self,
        kernel: BaseKernel,
        local_engine: BaseLocalEngine,
        pre_bandwidth: float,
        post_bandwidth: float,
        break_index: int,
        gamma: float,
    ):
        super().__init__(break_index=break_index, gamma=gamma)

        self.kernel = kernel
        self.local_engine = local_engine
        self.pre_bandwidth = pre_bandwidth
        self.post_bandwidth = post_bandwidth

        self.hyperparams = {
            'gamma': gamma,
            'break_index': break_index,
            'pre_bandwidth': pre_bandwidth,
            'post_bandwidth': post_bandwidth,
            'kernel': kernel, # Storing for inspection/logging
            'local_engine': local_engine # Storing for inspection/logging
        }

    def _initialize_sub_models(self) -> None:
        """Instantiates the pre- and post-break NP models."""
        self.model_pre = NPRegimeModel(
            kernel=self.kernel, local_engine=self.local_engine, bandwidth=self.pre_bandwidth
        )
        self.model_post = NPRegimeModel(
            kernel=self.kernel, local_engine=self.local_engine, bandwidth=self.post_bandwidth
        )
