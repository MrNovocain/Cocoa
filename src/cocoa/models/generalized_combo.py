
import pandas as pd
import numpy as np
from .base_model import BaseModel
from .combo_base import BaseConvexCombinationModel
from .shrinkage import ShrinkageFunction

class GeneralizedNonLinearComboModel(BaseConvexCombinationModel):
    """
    A generalized combination model:
    y_pred = pred_post + beta * g(pred_pre - pred_post)
    """

    def __init__(
        self,
        model_pre: BaseModel,
        model_post: BaseModel,
        break_index: int,
        beta: float,
        shrinkage_func: ShrinkageFunction,
    ):
        # Initialize parent with dummy gamma since we use beta and generalized formula
        super().__init__(break_index=break_index, gamma=beta) 
        
        self.model_pre = model_pre
        self.model_post = model_post
        self.beta = beta
        self.shrinkage_func = shrinkage_func
        
        # Override hyperparams for logging
        self.hyperparams = {
            'beta': beta,
            'break_index': break_index,
            'shrinkage': shrinkage_func.get_name(),
        }

    def _initialize_sub_models(self) -> None:
        # Sub-models are passed in __init__, so nothing to do here
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
             # If sub-models are already fitted (which they usually are in our runner flow), 
             # we might not strictly need self.fit() called on this wrapper if we trust the runner.
             # But strictly speaking, we should check.
             # For this implementation, we assume the runner fits the sub-models or calls fit() on this.
             pass

        # We assume sub-models are fitted.
        pred_pre = self.model_pre.predict(X)
        pred_post = self.model_post.predict(X)
        
        # If post model wasn't active during fit (no post data), BaseConvex handles it, 
        # but here we have explicit models. 
        # If model_post is not fitted, it might raise error or return garbage.
        # In our runner, we ensure they are fitted.
        
        diff = pred_pre - pred_post
        shrunk_diff = self.shrinkage_func(diff)
        
        return pred_post + self.beta * shrunk_diff
