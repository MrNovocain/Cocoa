import numpy as np
from .np_base import BaseKernel


class GaussianKernel(BaseKernel):
    """Gaussian kernel: K(u) = (1/sqrt(2*pi)) * exp(-u^2 / 2)."""

    def weight(self, u: float | np.ndarray) -> float | np.ndarray:
        """Computes the kernel weight for a given scaled distance u."""
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)


class EpanechnikovKernel(BaseKernel):
    """
    Epanechnikov kernel: K(u) = 0.75 * (1 - u^2) for |u| <= 1.
    This kernel is optimal in a statistical sense (minimizes AMISE).
    """

    def weight(self, u: float | np.ndarray) -> float | np.ndarray:
        """Computes the kernel weight for a given scaled distance u."""
        u_abs = np.abs(u)
        w = np.where(u_abs <= 1, 0.75 * (1 - u_abs**2), 0)
        return w