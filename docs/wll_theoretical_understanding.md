# Theoretical Understanding of WLL (CGS Method)

Based on the paper **"A New Nonparametric Combination Forecasting with Structural Breaks"** by Cai, Gao, and Selk (CGS), we can confirm the following theoretical underpinnings of the Weighted Local Linear (WLL) estimator.

## 1. Convex Combination and Bias-Variance Trade-off

The WLL estimator $\hat{m}_{wll}(x)$ is constructed as a convex combination of two estimators:
1.  $\hat{m}^{(1)}(x)$: The local linear estimator using **pre-break** data.
2.  $\hat{m}^{(2)}(x)$: The local linear estimator using **post-break** data.

The combination is controlled by a weight parameter $\gamma$ (which determines the effective weight $s_b$):

$$ \hat{m}_{wll}(x) \approx s_b \hat{m}^{(1)}(x) + (1 - s_b) \hat{m}^{(2)}(x) $$

### Why it works (MSE Perspective)
The goal is to minimize the Mean Squared Error (MSE) of the estimator. The MSE decomposes into squared bias and variance:

$$ MSE[\hat{m}_{wll}(x)] = \text{Bias}_{wll}^2(x) + \text{Var}_{wll}(x) $$

*   **Variance Reduction**: By including pre-break data (via $\gamma > 0$), the effective sample size increases. The paper shows that the asymptotic variance of the WLL estimator is strictly smaller than that of the post-break-only estimator ($\hat{m}^{(2)}(x)$).
    *   *Reference*: "Evidently, $s_{wll} < s_{(2)}$ so that the asymptotic variance for $\hat{m}_{wll}(x)$ is smaller than that for $\hat{m}_{(2)}(x)$..." (Section 2.3.2).

*   **Bias Introduction**: Including pre-break data introduces a bias because the pre-break mean function $m^{(1)}(x)$ differs from the current post-break mean function $m^{(2)}(x)$.

The convex combination works by finding the optimal $\gamma$ that balances this **reduction in variance** against the **increase in bias**.

## 2. Bias as a First-Order Term of $\lambda(x)$

The bias of the WLL estimator is explicitly derived in Equation (2.8) of the paper:

$$ B_{wll}(x) = s_b \lambda(x) + s_b B_1(x) + (1 - s_b) B_2(x) $$

Where:
*   $\lambda(x) = m^{(1)}(x) - m^{(2)}(x)$ is the **break size function** (the difference between pre- and post-break means).
*   $B_1(x)$ and $B_2(x)$ are the standard nonparametric asymptotic biases (proportional to $h^2 m''(x)$).

### Key Insight
The term $s_b \lambda(x)$ is the **dominant (first-order) term** in the bias expression when a structural break exists ($\lambda(x) \neq 0$).
*   The standard nonparametric biases $B_1(x)$ and $B_2(x)$ vanish at a rate of $h^2$ (second-order).
*   The break bias term $s_b \lambda(x)$ depends on the weight $s_b$ and the magnitude of the break $\lambda(x)$.

Thus, as you correctly noted: **The bias of WLL is basically a first-order term of $\lambda(x)$**, where $\lambda(x)$ represents the difference between the two underlying models ($m_1$ and $m_2$).

The tuning procedure (MFV) effectively selects $\gamma$ (and thus $s_b$) such that this induced bias $s_b \lambda(x)$ does not overwhelm the variance reduction benefits.

## 3. Extension: Generalized Non-linear Combination Estimator

We introduce a new estimator class that generalizes the standard convex combination by applying a non-linear transformation to the difference between the pre- and post-break estimators.

The proposed estimator form is:
$$ \hat{m}_g(x) = \hat{m}^{(2)}(x) + \beta(x) \cdot g\left( \hat{m}^{(1)}(x) - \hat{m}^{(2)}(x) \right) $$

### Derivation and Rationale

1.  **Standard WLL as a Special Case**:
    Recall the standard WLL estimator:
    $$ \hat{m}_{wll}(x) = \gamma \hat{m}^{(1)}(x) + (1-\gamma) \hat{m}^{(2)}(x) $$
    Rearranging terms:
    $$ \hat{m}_{wll}(x) = \hat{m}^{(2)}(x) + \gamma \left( \hat{m}^{(1)}(x) - \hat{m}^{(2)}(x) \right) $$
    This corresponds to the proposed form with $\beta(x) = \gamma$ (constant) and $g(u) = u$ (identity function).
    The bias term here is proportional to the break size $\lambda(x) \approx m^{(1)}(x) - m^{(2)}(x)$, i.e., **Bias $\propto \lambda$**.

2.  **Controlling Bias Growth via $g(\cdot)$**:
    When the structural break $\lambda$ is large, the linear bias term $\gamma \lambda$ becomes dominant and harmful. To mitigate this, we replace the linear difference with a **sub-linear function** $g(\cdot)$.
    *   **Sub-linear functions**: Examples include $g(u) = \text{sgn}(u)\log(1+|u|)$ or $g(u) = \text{sgn}(u)|u|^\theta$ with $0 < \theta < 1$.
    *   **Effect**: As the break size $|\lambda| \to \infty$, the bias term $\beta(x) g(\lambda)$ grows much slower than linearly.
        *   For log: Bias $\propto \log(\lambda)$
        *   For power: Bias $\propto \lambda^\theta$

3.  **Role of $\beta(x)$**:
    $\beta(x)$ acts as a generalized weight or scaling factor, similar to $\gamma$. It can be tuned (e.g., via MFV) to optimally balance the variance reduction from the "shrunk" pre-break information against the residual non-linear bias.

This formulation effectively **"shrinks" the break magnitude**, allowing the model to borrow strength from pre-break data even in the presence of larger breaks, without incurring a prohibitive bias penalty.

## Rolling Window Strategy: Recent to Past

In the context of **structural break detection** (specifically using the Mohr-Selk method), rolling from **recent to past** is crucial for the following reasons:

1.  **Stability of the "True" Break**: By starting with the most recent data (where the full history is available), we establish a robust estimate of the structural break ($T^*$) using the maximum amount of information. As we roll the origin *backward* (removing recent observations), we can test if this detected break remains stable or if it shifts/disappears when less post-break data is available.
2.  **Mimicking Real-Time Discovery**: In a real-world scenario, you are standing at "today" and looking back. Rolling backwards simulates the process of "how long ago would I have been able to detect this break?" It helps identify the point in time where the break signal becomes strong enough to be actionable.
3.  **Avoiding "Trimming" Issues**: The Mohr-Selk test often requires trimming the sample ends. By fixing the pilot break from the full sample and moving backwards, we ensure that we don't accidentally trim away the break we are trying to study in the early iterations, which might happen if we rolled forward from a point too close to the break.

Essentially, it's a stress test for the break detection: **"At what point in the past did the data stop supporting the existence of this break?"**
