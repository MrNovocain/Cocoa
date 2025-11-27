import numpy as np
import pandas as pd
from typing import Type
from ..models.base_model import BaseModel


def bias_variance_decomposition(
    base_model: BaseModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bootstrap_rounds: int = 50,
):
    """
    Performs bias-variance decomposition for a given model using bootstrapping.

    This implementation is model-agnostic. It works by cloning a pre-configured
    'base_model' and refitting it on bootstrap samples.

    Args:
        base_model (BaseModel): A configured and fitted model instance. It will be
                                cloned and refit in each bootstrap round.
        X_train, y_train: The original training data.
        X_test, y_test: The out-of-sample data for evaluation.
        n_bootstrap_rounds (int): The number of bootstrap samples to generate.

    Returns:
        A tuple containing:
        - avg_mse (float): The average Mean Squared Error over all rounds.
        - avg_bias_sq (float): The squared bias component.
        - avg_variance (float): The variance component.
    """
    print(f"Starting bias-variance decomposition with {n_bootstrap_rounds} rounds...")

    # Store predictions for each bootstrap round for each test point
    n_test = len(X_test)
    all_predictions = np.zeros((n_bootstrap_rounds, n_test))

    for i in range(n_bootstrap_rounds):
        # Create a bootstrap sample of the training data
        bootstrap_indices = np.random.choice(
            len(X_train), size=len(X_train), replace=True
        )
        X_boot = X_train.iloc[bootstrap_indices]
        y_boot = y_train.iloc[bootstrap_indices]

        # Clone the base model to get an unfitted instance with the same hyperparameters
        model_instance = base_model.clone()

        # Fit the model on the bootstrap sample
        model_instance.fit(X_boot, y_boot)

        # Store predictions on the original test set
        all_predictions[i, :] = model_instance.predict(X_test)

    # Calculate average MSE across all bootstrap rounds
    avg_mse = np.mean((all_predictions - y_test.values[np.newaxis, :]) ** 2)

    # Calculate squared bias and variance
    avg_prediction = np.mean(all_predictions, axis=0)
    avg_bias_sq = np.mean((avg_prediction - y_test.values) ** 2)
    avg_variance = np.mean(np.var(all_predictions, axis=0))

    return avg_mse, avg_bias_sq, avg_variance