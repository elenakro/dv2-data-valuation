import numpy as np

from inject_noise import inject_label_noise


def main():
    y = np.array([0, 1] * 500)

    print("=" * 50)
    print("LABEL NOISE TEST")
    print("=" * 50)

    for rate in [0.05, 0.10, 0.20]:

        noisy_y, noisy_indices, original_y = inject_label_noise(
            y,
            noise_rate=rate,
            random_state=42,
        )

        actual_noise = np.sum(noisy_y != original_y)

        print(f"\nNoise rate: {rate * 100:.0f}%")
        print("Expected noisy samples:", round(len(y) * rate))
        print("Actual noisy samples:", actual_noise)
        print("Detected corrupted indices:", len(noisy_indices))

        assert actual_noise == len(noisy_indices)
        assert np.all(noisy_y[noisy_indices] != original_y[noisy_indices])

    print("\nTEST COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
