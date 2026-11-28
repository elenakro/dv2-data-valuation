import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from calculate_data_oob import calculate_data_oob


SEEDS = [42, 123, 2026]

N_ESTIMATORS = 100
MAX_SAMPLES = 0.8

REMOVAL_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def generate_dataset(seed):

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        random_state=seed
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.625,
        stratify=y_temp,
        random_state=seed
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test, seed):

    start = time.perf_counter()

    model = LogisticRegression(
        max_iter=1000,
        random_state=seed
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    execution_time = time.perf_counter() - start

    return accuracy, execution_time


def run_experiment(seed):

    print("\n" + "=" * 65)
    print(f"POINT REMOVAL - SEED {seed}")
    print("=" * 65)

    # 1. Generate data

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = generate_dataset(seed)

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")

    # 2. Calculate Data-OOB

    print("\nCalculating Data-OOB...")

    oob_start = time.perf_counter()

    oob_results = calculate_data_oob(
        X_train,
        y_train,
        X_val,
        y_val,
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        random_state=seed,
        n_jobs=1
    )

    oob_time = time.perf_counter() - oob_start

    values = np.asarray(oob_results["values"])
    indices = np.asarray(oob_results["indices"])

    if len(values) != len(X_train):
        raise ValueError("Incorrect number of Data-OOB values.")

    if not np.array_equal(
        np.sort(indices),
        np.arange(len(X_train))
    ):
        raise ValueError("Unexpected Data-OOB indices.")

    print(f"Data-OOB time: {oob_time:.3f}s")

    # 3. Create rankings

    lowest_first = np.argsort(values)

    highest_first = np.argsort(values)[::-1]

    rng = np.random.default_rng(seed)
    random_order = rng.permutation(len(X_train))

    strategies = {
        "remove_lowest_first": lowest_first,
        "remove_highest_first": highest_first,
        "random_removal": random_order
    }

    results = []

    # 4. Run removal experiments

    for strategy_name, ranking in strategies.items():

        print(f"\nStrategy: {strategy_name}")

        for fraction in REMOVAL_FRACTIONS:

            n_remove = int(len(X_train) * fraction)

            removed_positions = ranking[:n_remove]

            keep_mask = np.ones(len(X_train), dtype=bool)
            keep_mask[removed_positions] = False

            X_remaining = X_train[keep_mask]
            y_remaining = y_train[keep_mask]

            accuracy, training_time = train_and_evaluate(
                X_remaining,
                y_remaining,
                X_test,
                y_test,
                seed
            )

            row = {
                "seed": seed,
                "strategy": strategy_name,
                "removal_fraction": fraction,
                "removed_samples": n_remove,
                "remaining_samples": len(y_remaining),
                "test_accuracy": accuracy,
                "training_time": training_time,
                "data_oob_time": oob_time
            }

            results.append(row)

            print(
                f"Removed: {fraction:>4.0%} | "
                f"Remaining: {len(y_remaining):>3} | "
                f"Accuracy: {accuracy:.4f}"
            )

    return results


def main():

    print("=" * 65)
    print("DATA-OOB - POINT REMOVAL WITH RANDOM BASELINE")
    print("=" * 65)

    all_results = []

    # 5. Run all seeds

    for seed in SEEDS:

        seed_results = run_experiment(seed)
        all_results.extend(seed_results)

    results_df = pd.DataFrame(all_results)

    # 6. Save raw results

    raw_dir = os.path.join(
        PROJECT_ROOT, "results", "raw"
    )

    figures_dir = os.path.join(
        PROJECT_ROOT, "results", "figures"
    )

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    raw_path = os.path.join(
        raw_dir,
        "point_removal_all_seeds.csv"
    )

    results_df.to_csv(raw_path, index=False)

    # 7. Calculate summary statistics

    summary_df = (
        results_df
        .groupby(["strategy", "removal_fraction"])
        ["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_df["std"] = summary_df["std"].fillna(0)

    summary_path = os.path.join(
        raw_dir,
        "point_removal_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    # 8. Create graph

    plt.figure(figsize=(11, 7))

    for strategy in results_df["strategy"].unique():

        data = summary_df[
            summary_df["strategy"] == strategy
        ]

        plt.errorbar(
            data["removal_fraction"] * 100,
            data["mean"],
            yerr=data["std"],
            marker="o",
            capsize=4,
            label=strategy
        )

    plt.xlabel("Removed Training Samples (%)")
    plt.ylabel("Test Accuracy")
    plt.title("Data-OOB Point Removal - 3 Seeds")

    plt.xticks(
        [x * 100 for x in REMOVAL_FRACTIONS]
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure_path = os.path.join(
        figures_dir,
        "point_removal_comparison.png"
    )

    plt.savefig(figure_path, dpi=300)
    plt.close()

    # 9. Final report

    print("\n" + "=" * 65)
    print("POINT REMOVAL EXPERIMENT COMPLETED")
    print("=" * 65)

    print(f"\nTotal experiments: {len(results_df)}")

    print("\nSUMMARY RESULTS:")
    print(summary_df.to_string(index=False))

    print("\nFILES CREATED:")
    print(raw_path)
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()
