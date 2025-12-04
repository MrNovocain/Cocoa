
from typing import Dict
from .generalized_runner import GeneralizedComboExperimentRunner

class LogComboExperimentRunner(GeneralizedComboExperimentRunner):
    """
    Runner specifically for the Logarithmic Shrinkage combination.
    g(u) = sgn(u) * log(1 + |u|)
    """
    def __init__(self, combo_type: str, **kwargs):
        super().__init__(
            combo_type=combo_type,
            shrinkage_type="log",
            shrinkage_params={},
            model_name=kwargs.pop("model_name", f"{combo_type}_LogCombo"),
            **kwargs
        )

class PowerComboExperimentRunner(GeneralizedComboExperimentRunner):
    """
    Runner specifically for the Power Shrinkage combination.
    g(u) = sgn(u) * |u|^theta
    """
    def __init__(self, combo_type: str, theta: float = 0.5, **kwargs):
        super().__init__(
            combo_type=combo_type,
            shrinkage_type="power",
            shrinkage_params={"theta": theta},
            model_name=kwargs.pop("model_name", f"{combo_type}_PowerCombo_{theta}"),
            **kwargs
        )
