
import numpy as np
from abc import ABC, abstractmethod

class ShrinkageFunction(ABC):
    """
    Abstract base class for shrinkage functions g(u).
    """
    @abstractmethod
    def __call__(self, u: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class LinearShrinkage(ShrinkageFunction):
    """
    g(u) = u
    Standard WLL behavior.
    """
    def __call__(self, u: np.ndarray) -> np.ndarray:
        return u

    def get_name(self) -> str:
        return "linear"

class LogShrinkage(ShrinkageFunction):
    """
    g(u) = sgn(u) * log(1 + |u|)
    Dampens large values logarithmically.
    """
    def __call__(self, u: np.ndarray) -> np.ndarray:
        return np.sign(u) * np.log1p(np.abs(u))

    def get_name(self) -> str:
        return "log"

class PowerShrinkage(ShrinkageFunction):
    """
    g(u) = sgn(u) * |u|^theta
    Dampens large values polynomially (if 0 < theta < 1).
    """
    def __init__(self, theta: float = 0.5):
        self.theta = theta

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return np.sign(u) * (np.abs(u) ** self.theta)

    def get_name(self) -> str:
        return f"power_{self.theta}"

def get_shrinkage_function(name: str, **kwargs) -> ShrinkageFunction:
    """Factory function to get shrinkage function by name."""
    name = name.lower()
    if name == "linear":
        return LinearShrinkage()
    elif name == "log":
        return LogShrinkage()
    elif name == "power":
        theta = kwargs.get("theta", 0.5)
        return PowerShrinkage(theta=theta)
    else:
        raise ValueError(f"Unknown shrinkage function: {name}")
