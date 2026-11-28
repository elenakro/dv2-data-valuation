import os
import sys
import time
import gc
import tracemalloc
import warnings

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from calculate_data_oob import calculate_data_oob


# Experiment configuration
DATASET_SIZES = [500, 2000, 10000]
SEEDS = [42, 123, 2026]

N_ESTIMATORS = 100
MAX_SAMPLES = 0.8


def generate_dataset(n_samples, seed):

    X, y = make_classification(
        n_samples=n_samples,
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


def run_experiment(n_samples, seed):

    print("\n" + "=" * 65)
    print(f"COST EVALUATION | SIZE={n_samples} | SEED={seed}")
    print("=" * 65)

    gc.collect()

    # 1. Dataset generation

    dataset_start = time.perf_counter()

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = generate_dataset(n_samples, seed)

    dataset_time = time.perf_counter() - dataset_start

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")

    # 2. Data-OOB calculation

    print("\nCalculating Data-OOB...")

    gc.collect()
    tracemalloc.start()

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

    current_memory, peak_memory = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    values = np.asarray(oob_results["values"])
    counts = np.asarray(oob_results["counts"])

    # 3. Final Logistic Regression training

    print("\nTraining final Logistic Regression...")

    model_start = time.perf_counter()

    model = LogisticRegression(
        max_iter=1000,
        random_state=seed
    )

    model.fit(X_train, y_train)

    model_time = time.perf_counter() - model_start

    # 4. Test evaluation

    predictions = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, predictions)

    # 5. Total measured time

    total_time = dataset_time + oob_time + model_time

    result = {
        "dataset_size": n_samples,
        "training_samples": len(y_train),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "seed": seed,
        "n_estimators": N_ESTIMATORS,
        "max_samples": MAX_SAMPLES,
        "dataset_generation_time": dataset_time,
        "data_oob_time": oob_time,
        "model_training_time": model_time,
        "total_time": total_time,
        "python_peak_memory_mb": peak_memory / (1024 ** 2),
        "python_current_memory_mb": current_memory / (1024 ** 2),
        "mean_oob_count": float(np.mean(counts)),
        "min_oob_count": int(np.min(counts)),
        "max_oob_count": int(np.max(counts)),
        "number_of_data_oob_values": len(values),
        "test_accuracy": test_accuracy
    }

    print("\nRESULTS")
    print("-" * 65)

    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def main():

    print("=" * 65)
    print("DATA-OOB COST EVALUATION")
    print("=" * 65)

    all_results = []

    for size in DATASET_SIZES:

        for seed in SEEDS:

            result = run_experiment(size, seed)

            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    raw_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "raw"
    )

    os.makedirs(raw_dir, exist_ok=True)

    raw_path = os.path.join(
        raw_dir,
        "cost_evaluation_all_seeds.csv"
    )

    results_df.to_csv(raw_path, index=False)

    summary_df = (
        results_df
        .groupby("dataset_size")
        .agg({
            "data_oob_time": ["mean", "std"],
            "model_training_time": ["mean", "std"],
            "total_time": ["mean", "std"],
            "python_peak_memory_mb": ["mean", "std"],
            "mean_oob_count": ["mean", "std"],
            "test_accuracy": ["mean", "std"]
        })
    )

    summary_path = os.path.join(
        raw_dir,
        "cost_evaluation_summary.csv"
    )

    summary_df.to_csv(summary_path)

    print("\n" + "=" * 65)
    print("COST EVALUATION COMPLETED")
    print("=" * 65)

    print("\nSUMMARY:")
    print(summary_df.to_string())

    print("\nFILES CREATED:")
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()
