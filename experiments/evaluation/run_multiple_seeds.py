import sys
import csv
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from evaluation.run_noise_detection_experiment import run_experiment


def main():

    # Τρία διαφορετικά random seeds
    random_seeds = [42, 123, 2026]

    # Τρία ποσοστά θορύβου
    noise_rates = [0.05, 0.10, 0.20]

    all_results = []

    total_experiments = len(random_seeds) * len(noise_rates)
    experiment_number = 0

    print("=" * 75)
    print("DATA-OOB MULTIPLE SEEDS EXPERIMENT")
    print("=" * 75)

    for seed in random_seeds:

        for noise_rate in noise_rates:

            experiment_number += 1

            print(
                f"\nExperiment {experiment_number}/{total_experiments}"
            )
            print(f"Random seed: {seed}")
            print(f"Noise rate: {noise_rate * 100:.0f}%")

            results = run_experiment(
                noise_rate=noise_rate,
                random_state=seed,
            )

            results["random_seed"] = seed

            all_results.append(results)

    # Αποθήκευση αναλυτικών αποτελεσμάτων
    output_dir = PROJECT_DIR / "results" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_file = output_dir / "noise_detection_all_seeds.csv"

    fieldnames = [
        "random_seed",
        "noise_rate",
        "precision",
        "recall",
        "f1_score",
        "test_accuracy",
        "execution_time",
    ]

    with open(detailed_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Υπολογισμός μέσου όρου και τυπικής απόκλισης
    summary = []

    for noise_rate in noise_rates:

        selected = [
            result for result in all_results
            if result["noise_rate"] == noise_rate
        ]

        summary_row = {
            "noise_rate": noise_rate,
        }

        metrics = [
            "precision",
            "recall",
            "f1_score",
            "test_accuracy",
            "execution_time",
        ]

        for metric in metrics:

            values = np.array([
                result[metric] for result in selected
            ])

            summary_row[f"{metric}_mean"] = float(np.mean(values))
            summary_row[f"{metric}_std"] = float(np.std(values, ddof=1))

        summary.append(summary_row)

    # Αποθήκευση συνοπτικών αποτελεσμάτων
    summary_file = output_dir / "noise_detection_summary.csv"

    with open(summary_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary)

    # Εμφάνιση τελικών αποτελεσμάτων
    print("\n" + "=" * 90)
    print("FINAL RESULTS - MEAN ± STANDARD DEVIATION")
    print("=" * 90)

    for row in summary:

        print(f"\nNoise: {row['noise_rate'] * 100:.0f}%")

        for metric in [
            "precision",
            "recall",
            "f1_score",
            "test_accuracy",
            "execution_time",
        ]:

            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]

            print(f"{metric}: {mean:.4f} ± {std:.4f}")

    print("\nFILES SAVED:")
    print(detailed_file)
    print(summary_file)

    print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
