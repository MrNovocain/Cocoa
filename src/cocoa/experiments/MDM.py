import numpy as np
import pandas as pd
from scipy.stats import t
from cocoa.models.assets import DEFAULT_TARGET_COL

class MDM:
    """
    Performs and summarizes the Modified Diebold-Mariano (MDM) test for forecast accuracy.

    This class takes true values and two sets of forecasts, computes the MDM statistic,
    and provides a summary of the results, following the CGS (2020) methodology.
    """

    def __init__(self, y_true, f_i, f_j, h, model_i_name='Model_i', model_j_name='Model_j'):
        """
        Initializes the MDM test.

        Args:
            y_true (np.ndarray): The true observed values.
            f_i (np.ndarray): Forecasts from model i (the benchmark model).
            f_j (np.ndarray): Forecasts from model j (the candidate model).
            h (int): The forecast horizon.
            model_i_name (str): Name of the benchmark model.
            model_j_name (str): Name of the candidate model.
        """
        # Calculate errors from forecasts
        e_i = np.asarray(y_true) - np.asarray(f_i)
        e_j = np.asarray(y_true) - np.asarray(f_j)

        # Drop NaNs consistently across all series
        valid_indices = ~np.isnan(e_i) & ~np.isnan(e_j)
        self.e_i = e_i[valid_indices]
        self.e_j = e_j[valid_indices]
        
        self.h = h
        self.model_i_name = model_i_name
        self.model_j_name = model_j_name
        self.results = None

    def compute(self):
        """
        Computes the Modified Diebold-Mariano (MDM) test statistic.

        The result is stored in the `self.results` dictionary.
        """
        T = len(self.e_i)
        if T == 0:
            self.results = {
                'mdm_stat': np.nan, 'p_value': np.nan, 'd_bar': np.nan,
                'omega_hat_sq': np.nan, 'df': 0, 'msfe_i': np.nan, 'msfe_j': np.nan
            }
            return self.results

        # 1. Define loss differential and MSFEs
        d_t = self.e_i**2 - self.e_j**2
        d_bar = np.mean(d_t)
        msfe_i = np.mean(self.e_i**2)
        msfe_j = np.mean(self.e_j**2)

        # 2. Estimate the long-run variance of the loss differential
        L = self.h - 1
        d_t_demeaned = d_t - d_bar

        gamma_hat = np.zeros(L + 1)
        for j in range(L + 1):
            gamma_hat[j] = np.sum(d_t_demeaned[j:] * d_t_demeaned[:T-j]) / T if T > j else 0

        omega_hat_sq = gamma_hat[0] + 2 * np.sum(gamma_hat[1:])
        
        use_bartlett = omega_hat_sq <= 0
        if use_bartlett:
            if L > 0:
                weights = 1 - (np.arange(1, L + 1) / (L + 1))
                omega_hat_sq = gamma_hat[0] + 2 * np.sum(weights * gamma_hat[1:])
            else:
                omega_hat_sq = gamma_hat[0]

        # 3. Build the MDM statistic
        if omega_hat_sq <= 0:
            mdm_stat = np.nan
        elif not use_bartlett:
            # CGS/HLN small-sample correction
            correction_factor = T + 1 - 2 * self.h + (self.h * (self.h - 1)) / T
            mdm_stat = np.sqrt(correction_factor) * d_bar / np.sqrt(omega_hat_sq)
        else:
            # Fallback with Bartlett variance and alternative scaling
            mdm_stat = np.sqrt(T) * d_bar / np.sqrt(omega_hat_sq)
            
        # 4. Compute p-value from t-distribution
        df = T - 1
        p_value = 1 - t.cdf(mdm_stat, df=df) if df > 0 and not np.isnan(mdm_stat) else np.nan

        self.results = {
            'mdm_stat': mdm_stat, 'p_value': p_value, 'd_bar': d_bar,
            'omega_hat_sq': omega_hat_sq, 'df': df, 'msfe_i': msfe_i, 'msfe_j': msfe_j
        }
        return self.results

    @staticmethod
    def get_significance_star(p_value):
        """Returns a significance star string ('*', '**', '***') based on the p-value."""
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

    def summary(self):
        """Prints a formatted summary of the test results."""
        if not self.results:
            self.compute()
        
        stars = self.get_significance_star(self.results['p_value'])
        
        print("\n" + "="*50)
        print("      Modified Diebold-Mariano Test Results")
        print("="*50)
        print(f"H0: MSFE({self.model_i_name}) = MSFE({self.model_j_name})")
        print(f"Ha: MSFE({self.model_i_name}) > MSFE({self.model_j_name}) (i.e., {self.model_j_name} is better)")
        print("-"*50)
        print(f"Benchmark model (i): {self.model_i_name}")
        print(f"Candidate model (j): {self.model_j_name}")
        print(f"Forecast Horizon (h): {self.h}")
        print(f"Observations (T): {self.results.get('df', -1) + 1}")
        print("-"*50)
        print(f"MSFE '{self.model_i_name}': {self.results['msfe_i']:.6f}")
        print(f"MSFE '{self.model_j_name}': {self.results['msfe_j']:.6f}")
        print(f"Loss Differential (d_bar): {self.results['d_bar']:.6f}")
        print(f"MDM Statistic: {self.results['mdm_stat']:.4f}")
        print(f"P-value (one-sided): {self.results['p_value']:.4f} {stars}")
        print("="*50)
        if stars:
            if self.results['mdm_stat'] > 0:
                level = {3: 1, 2: 5, 1: 10}.get(len(stars), "(unknown)")
                print(f"Conclusion: Reject H0. '{self.model_j_name}' is significantly better than '{self.model_i_name}' at the {level}% level.")
            else:
                # This case is unlikely with a one-sided test but included for completeness
                print(f"Conclusion: '{self.model_i_name}' has a lower MSFE, but the result is not significant for Ha.")
        else:
            print("Conclusion: Cannot reject H0. No significant difference in predictive accuracy found.")
            if self.results['msfe_j'] < self.results['msfe_i']:
                print(f"'{self.model_j_name}' has a lower MSFE, but the difference is not statistically significant.")
            elif self.results['msfe_i'] < self.results['msfe_j']:
                print(f"'{self.model_i_name}' has a lower MSFE, but the difference is not statistically significant.")
        print("="*50 + "\n")
