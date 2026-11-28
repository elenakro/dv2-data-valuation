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

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from calculate_data_oob import calculate_data_oob


RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_SAMPLES = 0.8

REMOVAL_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def generate_dataset():
    """
    Generate a reproducible synthetic classification dataset.
    """

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        random_state=RANDOM_STATE
    )

    # 60% training, 40% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Temporary data: 15% validation, 25% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.625,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train Logistic Regression and evaluate on the fixed test set.
    """

    start_time = time.perf_counter()

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    execution_time = time.perf_counter() - start_time

    return accuracy, execution_time


def run_point_removal():
    print("=" * 65)
    print("DATA-OOB - POINT REMOVAL EXPERIMENT")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Generate dataset
    # --------------------------------------------------

    print("\n[1/5] Generating dataset...")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = generate_dataset()

    print(f"Training samples:   {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples:       {len(y_test)}")

    # --------------------------------------------------
    # 2. Calculate Data-OOB values
    # --------------------------------------------------

    print("\n[2/5] Calculating Data-OOB values...")

    oob_start = time.perf_counter()

    oob_results = calculate_data_oob(
        X_train,
        y_train,
        X_val,
        y_val,
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    oob_time = time.perf_counter() - oob_start

    values = np.asarray(oob_results["values"])
    indices = np.asarray(oob_results["indices"])

    if len(values) != len(X_train):
        raise ValueError(
            "The number of Data-OOB values does not match "
            "the number of training samples."
        )

    if len(indices) != len(X_train):
        raise ValueError(
            "The number of Data-OOB indices does not match "
            "the training set."
        )

    if not np.array_equal(
        np.sort(indices),
        np.arange(len(X_train))
    ):
        raise ValueError(
            "Unexpected Data-OOB indices. "
            "Cannot safely map values to training samples."
        )

    # Values and indices are aligned.
    # Lowest values first.
    ascending_order = np.argsort(values)

    # Highest values first.
    descending_order = np.argsort(values)[::-1]

    print(f"Data-OOB calculation time: {oob_time:.3f} seconds")
    print(f"Data-OOB values: {len(values)}")

    # --------------------------------------------------
    # 3. Point removal experiments
    # --------------------------------------------------

    print("\n[3/5] Running point removal experiments...")

    results = []

    removal_strategies = {
        "remove_lowest_first": ascending_order,
        "remove_highest_first": descending_order
    }

    for strategy_name, ranking in removal_strategies.items():

        print(f"\nStrategy: {strategy_name}")

        for fraction in REMOVAL_FRACTIONS:

            n_remove = int(len(X_train) * fraction)

            # Select samples to remove
            removed_positions = ranking[:n_remove]

            # Keep all remaining samples
            keep_mask = np.ones(len(X_train), dtype=bool)
            keep_mask[removed_positions] = False

            X_remaining = X_train[keep_mask]
            y_remaining = y_train[keep_mask]

            if len(y_remaining) == 0:
                raise ValueError("No training samples remain.")

            accuracy, training_time = train_and_evaluate(
                X_remaining,
                y_remaining,
                X_test,
                y_test
            )

            row = {
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
                f"Remaining: {len(y_remaining):>4} | "
                f"Test Accuracy: {accuracy:.4f} | "
                f"Time: {training_time:.3f}s"
            )

    # --------------------------------------------------
    # 4. Save results
    # --------------------------------------------------

    print("\n[4/5] Saving results...")

    results_df = pd.DataFrame(results)

    raw_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "raw"
    )

    figures_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "figures"
    )

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(
        raw_dir,
        "point_removal_results.csv"
    )

    results_df.to_csv(csv_path, index=False)

    print(f"CSV saved: {csv_path}")

    # --------------------------------------------------
    # 5. Create graph
    # --------------------------------------------------

    print("\n[5/5] Creating graph...")

    plt.figure(figsize=(10, 6))

    for strategy_name in removal_strategies:

        strategy_data = results_df[
            results_df["strategy"] == strategy_name
        ]

        plt.plot(
            strategy_data["removal_fraction"] * 100,
            strategy_data["test_accuracy"],
            marker="o",
            label=strategy_name
        )

    plt.xlabel("Removed Training Samples (%)")
    plt.ylabel("Test Accuracy")
    plt.title("Data-OOB Point Removal Experiment")

    plt.xticks(
        [x * 100 for x in REMOVAL_FRACTIONS]
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure_path = os.path.join(
        figures_dir,
        "point_removal_accuracy.png"
    )

    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Graph saved: {figure_path}")

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("POINT REMOVAL EXPERIMENT COMPLETED")
    print("=" * 65)

    print("\nResults:")
    print(results_df.to_string(index=False))

    print("\nFiles created:")
    print(f"- {csv_path}")
    print(f"- {figure_path}")


if __name__ == "__main__":
    run_point_removal()
