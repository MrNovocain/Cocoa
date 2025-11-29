"""Parametric simulation to validate MFV gamma tuning logic.

This module builds a stylized linear data-generating process with a structural
break. It exposes a small simulation harness that mirrors the MFV gamma tuning
workflow (forward folds on post-break data, grid search over ``gamma``). The
simulator also computes an oracle gamma based on the empirical break size and
estimation noise, letting us check whether MFV selects sensible weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _simulate_series(
    *,
    T: int,
    tau: int,
    phi: float,
    sigma_u: float,
    sigma_eps: float,
    alpha0: float,
    alpha1: float,
    beta0: float,
    beta1: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate (X, y) with an AR(1) regressor and a post-break linear mean."""

    X = np.zeros(T)
    for t in range(1, T):
        X[t] = phi * X[t - 1] + rng.normal(scale=sigma_u)

    eps = rng.normal(scale=sigma_eps, size=T)

    y = np.empty(T)
    pre_mask = np.arange(T) <= tau
    post_mask = ~pre_mask

    y[pre_mask] = alpha0 + alpha1 * X[pre_mask] + eps[pre_mask]
    y[post_mask] = beta0 + beta1 * X[post_mask] + eps[post_mask]

    return X, y, eps


def _fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return OLS coefficients (intercept, slope) for y ~ 1 + x."""

    X_design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return coef


def _predict_linear(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coef[0] + coef[1] * x


def _compute_oracle_gamma(
    *,
    x_post: np.ndarray,
    pre_coef: np.ndarray,
    post_coef: np.ndarray,
    alpha0: float,
    alpha1: float,
    beta0: float,
    beta1: float,
) -> float:
    """Empirical plug-in oracle from the quadratic risk derivation."""

    m_a = alpha0 + alpha1 * x_post
    m_b = beta0 + beta1 * x_post

    pre_error_sq = (_predict_linear(pre_coef, x_post) - m_a) ** 2
    post_error_sq = (_predict_linear(post_coef, x_post) - m_b) ** 2
    delta_sq = (m_a - m_b) ** 2

    v_pre = float(np.mean(pre_error_sq))
    v_post = float(np.mean(post_error_sq))
    d_sq = float(np.mean(delta_sq))

    denom = d_sq + v_pre + v_post
    return 0.0 if denom == 0 else v_post / denom


@dataclass
class GammaTuningResult:
    tuned_gamma: float
    oracle_gamma: float
    gamma_grid: Sequence[float]
    losses: Sequence[float]


class GammaMFVSimulator:
    """Minimal MFV-style simulator for the convex combination weight gamma."""

    def __init__(
        self,
        *,
        gamma_grid: Iterable[float] | None = None,
        Q: int = 4,
    ):
        self.gamma_grid: List[float] = (
            list(gamma_grid) if gamma_grid is not None else np.linspace(0.0, 1.0, 11).tolist()
        )
        self.Q = Q

    def _mfv_losses(
        self,
        *,
        x_pre: np.ndarray,
        y_pre: np.ndarray,
        x_post: np.ndarray,
        y_post: np.ndarray,
        tau: int,
        alpha0: float,
        alpha1: float,
        beta0: float,
        beta1: float,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Run forward MFV over post-break data and return gamma losses."""

        pre_coef = _fit_ols(x_pre, y_pre)

        T_post = len(x_post)
        block_size = T_post // (self.Q + 1)
        if block_size == 0:
            raise ValueError("Post-break sample too small for requested MFV folds.")

        cv_start_post = T_post - block_size * self.Q
        losses = np.zeros(len(self.gamma_grid))

        for q in range(self.Q):
            val_start = cv_start_post + q * block_size
            val_end = val_start + block_size

            x_val = x_post[val_start:val_end]
            y_val = y_post[val_start:val_end]

            if len(x_val) == 0:
                continue

            # Train on data up to the start of the validation slice
            x_train_fold = x_post[:val_start]
            y_train_fold = y_post[:val_start]

            post_coef = _fit_ols(x_train_fold, y_train_fold)

            y_pre_hat = _predict_linear(pre_coef, x_val)
            y_post_hat = _predict_linear(post_coef, x_val)

            for i, gamma in enumerate(self.gamma_grid):
                y_hat = gamma * y_pre_hat + (1.0 - gamma) * y_post_hat
                losses[i] += np.mean((y_val - y_hat) ** 2)

        losses /= self.Q

        post_full_coef = _fit_ols(x_post, y_post)
        oracle_gamma = _compute_oracle_gamma(
            x_post=x_post,
            pre_coef=pre_coef,
            post_coef=post_full_coef,
            alpha0=alpha0,
            alpha1=alpha1,
            beta0=beta0,
            beta1=beta1,
        )

        return losses, oracle_gamma, pre_coef, post_full_coef

    def run_simulation(
        self,
        *,
        T: int = 400,
        tau: int = 200,
        phi: float = 0.5,
        sigma_u: float = 1.0,
        sigma_eps: float = 1.0,
        alpha0: float = 0.0,
        alpha1: float = 1.0,
        beta0: float = 0.0,
        beta1: float = 1.0,
        random_seed: int | None = None,
    ) -> GammaTuningResult:
        rng = np.random.default_rng(random_seed)
        x, y, _ = _simulate_series(
            T=T,
            tau=tau,
            phi=phi,
            sigma_u=sigma_u,
            sigma_eps=sigma_eps,
            alpha0=alpha0,
            alpha1=alpha1,
            beta0=beta0,
            beta1=beta1,
            rng=rng,
        )

        x_pre, y_pre = x[: tau + 1], y[: tau + 1]
        x_post, y_post = x[tau + 1 :], y[tau + 1 :]

        losses, oracle_gamma, _, _ = self._mfv_losses(
            x_pre=x_pre,
            y_pre=y_pre,
            x_post=x_post,
            y_post=y_post,
            tau=tau,
            alpha0=alpha0,
            alpha1=alpha1,
            beta0=beta0,
            beta1=beta1,
        )

        best_idx = int(np.argmin(losses))
        tuned_gamma = float(self.gamma_grid[best_idx])

        return GammaTuningResult(
            tuned_gamma=tuned_gamma,
            oracle_gamma=oracle_gamma,
            gamma_grid=self.gamma_grid,
            losses=losses.tolist(),
        )

    def run_many(
        self,
        *,
        n_sim: int,
        random_seed: int | None = None,
        **kwargs,
    ) -> List[GammaTuningResult]:
        rng = np.random.default_rng(random_seed)
        seeds = rng.integers(low=0, high=2**32 - 1, size=n_sim)
        return [self.run_simulation(random_seed=int(seed), **kwargs) for seed in seeds]

