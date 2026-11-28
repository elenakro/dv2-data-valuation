import numpy as np

from evaluate_noise_detection import evaluate_noise_detection


def main():
    print("=" * 50)
    print("NOISE DETECTION EVALUATION TEST")
    print("=" * 50)

    n_samples = 100
    n_noisy = 10

    # Δείκτες δειγμάτων
    indices = np.arange(n_samples)

    # Τα πρώτα 10 δείγματα είναι αλλοιωμένα
    noisy_indices = np.arange(n_noisy)

    # Τιμές Data-OOB
    # Τα αλλοιωμένα έχουν χαμηλές τιμές
    values = np.full(n_samples, 0.9)
    values[noisy_indices] = 0.1

    results = evaluate_noise_detection(
        values=values,
        valuation_indices=indices,
        noisy_indices=noisy_indices,
    )

    print("\nRESULTS")
    print("-" * 50)

    print("Precision:", round(results["precision"], 4))
    print("Recall:", round(results["recall"], 4))
    print("F1-score:", round(results["f1_score"], 4))

    print("Actual noisy samples:", results["actual_noisy_samples"])
    print("Detected samples:", results["detected_samples"])

    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1_score"] == 1.0

    print("\nTEST COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
