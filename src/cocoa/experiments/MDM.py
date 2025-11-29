import numpy as np
from scipy.stats import t

def compute_mdm(e_i, e_j, h):
    """
    Computes the Modified Diebold-Mariano (MDM) test statistic as in CGS (2020).

    This implementation follows the description for comparing two models' forecasts.
    The null hypothesis is that the two models have the same forecast accuracy.
    The one-sided alternative is that model j is better than model i.

    Args:
        e_i (np.ndarray): Forecast errors for model i (benchmark).
        e_j (np.ndarray): Forecast errors for model j (candidate).
        h (int): Forecast horizon.

    Returns:
        dict: A dictionary containing:
            'mdm_stat': The MDM statistic.
            'p_value': The one-sided p-value.
            'd_bar': The mean of the loss differential.
            'omega_hat_sq': The estimated long-run variance of the loss differential.
            'df': Degrees of freedom.
    """
    # 0. Input validation and cleaning
    e_i = np.asarray(e_i)
    e_j = np.asarray(e_j)

    if e_i.shape != e_j.shape:
        raise ValueError("Input error series must have the same shape.")

    # Combine and drop NaNs consistently across both series
    valid_indices = ~np.isnan(e_i) & ~np.isnan(e_j)
    e_i = e_i[valid_indices]
    e_j = e_j[valid_indices]
    
    T = len(e_i)
    if T == 0:
        return {
            'mdm_stat': np.nan,
            'p_value': np.nan,
            'd_bar': np.nan,
            'omega_hat_sq': np.nan,
            'df': 0
        }

    # 1. Define loss differential series d_t
    # d_t > 0 means model j has smaller squared error than model i.
    d_t = e_i**2 - e_j**2
    d_bar = np.mean(d_t)

    # 2. Estimate the long-run variance omega^2 of d_t
    L = h - 1
    d_t_demeaned = d_t - d_bar

    # Autocovariances
    gamma_hat = np.zeros(L + 1)
    for j in range(L + 1):
        if T > j:
            gamma_hat[j] = np.sum(d_t_demeaned[j:] * d_t_demeaned[:T-j]) / T
        else:
            gamma_hat[j] = 0

    # Long-run variance estimator (Newey-West type)
    omega_hat_sq = gamma_hat[0] + 2 * np.sum(gamma_hat[1:])
    
    use_bartlett = omega_hat_sq <= 0
    if use_bartlett:
        # Fallback to Bartlett HAC estimator if omega_hat_sq is negative
        if L > 0:
            weights = 1 - (np.arange(1, L + 1) / (L + 1))
            omega_hat_sq = gamma_hat[0] + 2 * np.sum(weights * gamma_hat[1:])
        else: # if L=0, it's just gamma_hat[0]
            omega_hat_sq = gamma_hat[0]

    # 3. Build the MDM statistic
    if omega_hat_sq <= 0: # If still non-positive, stat is undefined
        mdm_stat = np.nan
    elif not use_bartlett:
        # CGS/HLN small-sample correction
        correction_factor = T + 1 - 2 * h + (h * (h - 1)) / T
        mdm_stat = np.sqrt(correction_factor) * d_bar / np.sqrt(omega_hat_sq)
    else: # Use Bartlett variance with a different scaling
        # The prompt states: MDM = T * d_bar / omega_hat_Bart.
        # This is unusual (usually scales with sqrt(T)), but implemented as per instruction.
        mdm_stat = T * d_bar / np.sqrt(omega_hat_sq)
        
    # 4. Compute p-values
    df = T - 1
    if df <= 0 or np.isnan(mdm_stat):
        p_value = np.nan
    else:
        # One-sided test: H_a is that model j is better (MSFE_j < MSFE_i)
        # This corresponds to d_bar > 0. We look at the right tail of the t-distribution.
        p_value = 1 - t.cdf(mdm_stat, df=df)

    return {
        'mdm_stat': mdm_stat,
        'p_value': p_value,
        'd_bar': d_bar,
        'omega_hat_sq': omega_hat_sq,
        'df': df
    }

def get_significance_star(p_value):
    """
    Returns a significance star string based on the p-value.
    Matches CGS/proposal convention.
    """
    if p_value is None or np.isnan(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.10:
        return "*"
    else:
        return ""
