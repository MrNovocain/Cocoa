import os
import random

import numpy as np


def set_global_seed(seed: int | None = None) -> int:
    """Set deterministic seeds for Python and NumPy."""
    if seed is None:
        seed_str = os.getenv("RANDOM_SEED", "42")
        seed = int(seed_str)

    random.seed(seed)
    np.random.seed(seed)

    return seed
