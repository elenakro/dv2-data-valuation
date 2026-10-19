"""Dataset loading + label noise injection.

Keeps a single interface for all methods: get_dataset(name, n, noise_rate, seed)
returns (X_train, y_train, X_val, y_val, X_test, y_test, noisy_idx).
"""

import numpy as np
from sklearn.model_selection import train_test_split


def inject_label_noise(y, noise_rate, n_classes, rng):
    """Flip labels of a random fraction to a different random class.

    Returns the corrupted labels and the indices of corrupted points.
    """
    n = len(y)
    n_noisy = int(round(noise_rate * n))
    noisy_idx = rng.choice(n, size=n_noisy, replace=False)

    y_noisy = y.copy()
    for i in noisy_idx:
        other_classes = [c for c in range(n_classes) if c != y[i]]
        y_noisy[i] = rng.choice(other_classes)
    return y_noisy, noisy_idx


def get_dataset(name, n_train, noise_rate=0.0, seed=42):
    """Load a dataset by name and prepare splits.

    TODO (per member):
      - Elena: wire up OpenDataVal loaders for 'electricity', '2dplanes', 'MiniBooNE'
      - add CIFAR10 / IMDB embeddings loader
    """
    rng = np.random.default_rng(seed)
    raise NotImplementedError(f"TODO: load dataset {name}")
