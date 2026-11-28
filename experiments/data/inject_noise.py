import numpy as np


def inject_label_noise(y, noise_rate=0.05, random_state=42):
    """
    Εισάγει ελεγχόμενο θόρυβο στις ετικέτες.

    Υποστηρίζει αρχικά δυαδική και πολυταξική ταξινόμηση.

    Returns:
        noisy_labels: Οι αλλοιωμένες ετικέτες.
        noisy_indices: Οι δείκτες των αλλοιωμένων δειγμάτων.
        original_labels: Αντίγραφο των αρχικών ετικετών.
    """

    if not 0 <= noise_rate <= 1:
        raise ValueError("noise_rate must be between 0 and 1.")

    y = np.asarray(y)
    original_labels = y.copy()
    noisy_labels = y.copy()

    classes = np.unique(y)

    if len(classes) < 2:
        raise ValueError("At least two classes are required.")

    rng = np.random.default_rng(random_state)

    n_samples = len(y)
    n_noisy = int(round(n_samples * noise_rate))

    noisy_indices = rng.choice(
        n_samples,
        size=n_noisy,
        replace=False,
    )

    for index in noisy_indices:
        possible_labels = classes[classes != original_labels[index]]

        noisy_labels[index] = rng.choice(possible_labels)

    return noisy_labels, noisy_indices, original_labels
