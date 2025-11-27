import numpy as np
import pandas as pd
from typing import Type, Any, Dict, Tuple

from ..models.base_model import BaseModel
from ..models import NPConvexCombinationModel


def bias_variance_decomposition(
    model_class: Type[BaseModel],
    hyperparams: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bootstrap_rounds: int = 50,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Performs bias-variance decomposition for a given model using bootstrapping.

    The decomposition is: E[(y - f_hat(x))^2] = Bias^2 + Variance + Irreducible Error

    This function estimates the average Bias^2 and average Variance over the test set.
    The Mean Squared Error (MSE) on the test set is the sum of these two components
    (assuming the irreducible error is part of the bias term as we can't separate it
    from the true data generating process).

    Args:
        model_class (Type[BaseModel]): The class of the model to evaluate (e.g., RFModel).
        hyperparams (Dict[str, Any]): Dictionary of hyperparameters to instantiate the model.
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        X_test (pd.DataFrame): Test feature data for evaluation.
        y_test (pd.Series): True target values for the test data.
        n_bootstrap_rounds (int): The number of bootstrap samples to create.
        random_seed (int): Seed for the random number generator for reproducibility.

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - avg_mse (float): The average Mean Squared Error on the test set.
            - avg_bias_sq (float): The estimated squared bias component of the error.
            - avg_variance (float): The estimated variance component of the error.
    """
    if not (isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.Series)):
        raise TypeError("X_train and y_train must be pandas objects.")
    if len(X_train) != len(y_train):
        raise ValueError("Training features and target must have the same length.")

    # Set seed for reproducible bootstrap sampling
    rng = np.random.default_rng(random_seed)

    # Store predictions for each bootstrap round on the fixed test set
    # Shape: (n_bootstrap_rounds, n_test_samples)
    all_predictions = np.zeros((n_bootstrap_rounds, len(X_test)))

    n_train_samples = len(X_train)

    print(f"Starting bias-variance decomposition with {n_bootstrap_rounds} rounds...")

    for i in range(n_bootstrap_rounds):
        # 1. Create a bootstrap sample from the training data (sampling with replacement)
        bootstrap_indices = rng.choice(
            n_train_samples, size=n_train_samples, replace=True
        )
        # We use .loc to preserve the original index, which is crucial for
        # the NPConvexCombinationModel to correctly split pre/post break data.
        X_boot = X_train.loc[X_train.index[bootstrap_indices]]
        y_boot = y_train.loc[y_train.index[bootstrap_indices]]

        # 2. Instantiate and fit a new model on the bootstrap sample
        if model_class == NPConvexCombinationModel:
            # Special handling for the convex combination model, which requires
            # constructing its sub-models before instantiation.
            from ..models import NPRegimeModel, GaussianKernel, LocalPolynomialEngine

            kernel = GaussianKernel()
            poly_order = hyperparams.get('poly_order', 1)
            engine = LocalPolynomialEngine(order=poly_order)

            # Use the correct hyperparameter names from the combo runner
            model_pre = NPRegimeModel(
                kernel=kernel,
                local_engine=engine,
                bandwidth=hyperparams['bandwidth_pre']
            )
            model_post = NPRegimeModel(
                kernel=kernel,
                local_engine=engine,
                bandwidth=hyperparams['bandwidth_post']
            )

            model = model_class(
                model_pre=model_pre,
                model_post=model_post,
                break_index=hyperparams['break_index'],
                gamma=hyperparams['gamma']
            )
        else:
            # Standard instantiation for all other models
            model = model_class(**hyperparams)
        model.fit(X_boot, y_boot)

        # 3. Predict on the original, fixed test set
        y_pred = model.predict(X_test)
        all_predictions[i, :] = y_pred

        if (i + 1) % 10 == 0:
            print(f"  Completed round {i+1}/{n_bootstrap_rounds}...")

    # Ensure y_test is a numpy array for calculations
    y_test_np = y_test.values

    # 4. Calculate bias and variance components for each test point
    # Average prediction for each test point across all bootstrap models
    avg_predictions = np.mean(all_predictions, axis=0)

    # Squared Bias: (E[f_hat(x)] - y)^2
    # We use (avg_predictions - y_test)^2 as our estimate
    squared_bias = (avg_predictions - y_test_np) ** 2

    # Variance: E[(f_hat(x) - E[f_hat(x)])^2]
    # We use the variance of predictions for each point as our estimate
    variance = np.var(all_predictions, axis=0)

    # Total MSE for each point
    mse = (all_predictions - y_test_np) ** 2

    # 5. Average over all test points
    avg_mse = np.mean(mse)
    avg_bias_sq = np.mean(squared_bias)
    avg_variance = np.mean(variance)

    print("Decomposition complete.")
    return avg_mse, avg_bias_sq, avg_variance