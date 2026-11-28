import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_noise_detection(
    values,
    valuation_indices,
    noisy_indices,
):
    """
    Αξιολογεί την ικανότητα του Data-OOB
    να εντοπίζει αλλοιωμένες ετικέτες.

    Επιλέγει τα Ν δείγματα με τη χαμηλότερη
    τιμή Data-OOB, όπου Ν είναι το πλήθος
    των πραγματικά αλλοιωμένων δειγμάτων.
    """

    values = np.asarray(values)
    valuation_indices = np.asarray(valuation_indices)
    noisy_indices = np.asarray(noisy_indices)

    if len(values) != len(valuation_indices):
        raise ValueError(
            "Values and valuation indices must have equal length."
        )

    if len(np.unique(valuation_indices)) != len(valuation_indices):
        raise ValueError("Duplicate valuation indices found.")

    if not set(noisy_indices).issubset(set(valuation_indices)):
        raise ValueError(
            "Some noisy indices are missing from valuation results."
        )

    n_noisy = len(noisy_indices)

    if n_noisy == 0:
        raise ValueError("No corrupted samples were provided.")

    if n_noisy > len(values):
        raise ValueError("More noisy samples than available values.")

    # Επιλογή των δειγμάτων με τις χαμηλότερες τιμές
    lowest_positions = np.argsort(values, kind="stable")[:n_noisy]

    predicted_indices = valuation_indices[lowest_positions]

    # Δημιουργία δυαδικών διανυσμάτων
    y_true = np.isin(valuation_indices, noisy_indices)
    y_pred = np.isin(valuation_indices, predicted_indices)

    return {
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "predicted_indices": predicted_indices,
        "actual_noisy_samples": n_noisy,
        "detected_samples": len(predicted_indices),
    }
