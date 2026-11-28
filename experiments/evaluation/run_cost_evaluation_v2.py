import os
import sys
import time
import gc
import tracemalloc
import warnings
import threading

import numpy as np
import pandas as pd
import psutil

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from calculate_data_oob import calculate_data_oob


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

DATASET_SIZES = [500, 2000, 10000]
SEEDS = [42, 123, 2026]

N_ESTIMATORS = 100
MAX_SAMPLES = 0.8

# RSS sampling interval in seconds
RSS_SAMPLE_INTERVAL = 0.01

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# RSS MEMORY MONITOR
# ============================================================

class RSSMemoryMonitor:

    def __init__(self, interval=RSS_SAMPLE_INTERVAL):

        self.process = psutil.Process(os.getpid())
        self.interval = interval

        self.stop_event = threading.Event()
        self.thread = None

        self.baseline_rss = 0
        self.peak_rss = 0
        self.final_rss = 0

    def _monitor(self):

        while not self.stop_event.is_set():

            try:

                current_rss = self.process.memory_info().rss

                if current_rss > self.peak_rss:
                    self.peak_rss = current_rss

            except psutil.Error:
                pass

            self.stop_event.wait(self.interval)

    def start(self):

        gc.collect()

        self.baseline_rss = self.process.memory_info().rss
        self.peak_rss = self.baseline_rss

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.stop_event.set()

        if self.thread is not None:
            self.thread.join()

        self.final_rss = self.process.memory_info().rss

        # Include final measurement in the peak
        self.peak_rss = max(
            self.peak_rss,
            self.final_rss
        )

        return {
            "rss_baseline_mb": self.baseline_rss / (1024 ** 2),

            "rss_peak_mb": self.peak_rss / (1024 ** 2),

            "rss_final_mb": self.final_rss / (1024 ** 2),

            "rss_increase_mb": max(
                0,
                self.peak_rss - self.baseline_rss
            ) / (1024 ** 2)
        }


# ============================================================
# DATASET GENERATION
# ============================================================

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

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(n_samples, seed):

    print("\n" + "=" * 70)
    print(f"COST EVALUATION V2 | SIZE={n_samples} | SEED={seed}")
    print("=" * 70)

    gc.collect()

    # --------------------------------------------------------
    # 1. Dataset generation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 2. Data-OOB calculation
    # --------------------------------------------------------

    print("\nCalculating Data-OOB...")

    gc.collect()

    # Python memory measurement
    tracemalloc.start()

    # Actual process RAM measurement
    memory_monitor = RSSMemoryMonitor()
    memory_monitor.start()

    oob_start = time.perf_counter()

    try:

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

    finally:

        rss_results = memory_monitor.stop()

        current_memory, peak_memory = (
            tracemalloc.get_traced_memory()
        )

        tracemalloc.stop()

    values = np.asarray(oob_results["values"])
    counts = np.asarray(oob_results["counts"])

    # --------------------------------------------------------
    # 3. Final Logistic Regression training
    # --------------------------------------------------------

    print("\nTraining final Logistic Regression...")

    model_start = time.perf_counter()

    model = LogisticRegression(
        max_iter=1000,
        random_state=seed
    )

    model.fit(X_train, y_train)

    model_time = time.perf_counter() - model_start

    # --------------------------------------------------------
    # 4. Test evaluation
    # --------------------------------------------------------

    predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # 5. Total measured time
    # --------------------------------------------------------

    total_time = (
        dataset_time +
        oob_time +
        model_time
    )

    # --------------------------------------------------------
    # 6. Results
    # --------------------------------------------------------

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

        # Python memory measurements
        "python_peak_memory_mb": (
            peak_memory / (1024 ** 2)
        ),

        "python_current_memory_mb": (
            current_memory / (1024 ** 2)
        ),

        # Actual process RSS measurements
        "rss_baseline_mb": rss_results["rss_baseline_mb"],

        "rss_peak_mb": rss_results["rss_peak_mb"],

        "rss_final_mb": rss_results["rss_final_mb"],

        "rss_increase_mb": rss_results["rss_increase_mb"],

        # Data-OOB statistics
        "mean_oob_count": float(np.mean(counts)),

        "min_oob_count": int(np.min(counts)),

        "max_oob_count": int(np.max(counts)),

        "number_of_data_oob_values": len(values),

        # Model quality
        "test_accuracy": test_accuracy
    }

    print("\nRESULTS")
    print("-" * 70)

    for key, value in result.items():

        print(f"{key}: {value}")

    return result


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("DATA-OOB COST EVALUATION V2")
    print("RSS MEMORY MONITORING ENABLED")
    print("=" * 70)

    print("\nConfiguration:")
    print("Dataset sizes:", DATASET_SIZES)
    print("Seeds:", SEEDS)
    print("Estimators:", N_ESTIMATORS)
    print("Max samples:", MAX_SAMPLES)
    print("RSS sampling interval:", RSS_SAMPLE_INTERVAL)

    all_results = []

    # --------------------------------------------------------
    # Run all experiments
    # --------------------------------------------------------

    for size in DATASET_SIZES:

        for seed in SEEDS:

            result = run_experiment(size, seed)

            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    raw_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "raw"
    )

    os.makedirs(raw_dir, exist_ok=True)

    raw_path = os.path.join(
        raw_dir,
        "cost_evaluation_v2_all_seeds.csv"
    )

    results_df.to_csv(
        raw_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    summary_columns = [

        "data_oob_time",

        "model_training_time",

        "total_time",

        "python_peak_memory_mb",

        "rss_baseline_mb",

        "rss_peak_mb",

        "rss_final_mb",

        "rss_increase_mb",

        "mean_oob_count",

        "test_accuracy"
    ]

    summary_df = (

        results_df

        .groupby("dataset_size")

        [summary_columns]

        .agg(["mean", "std"])
    )

    summary_path = os.path.join(
        raw_dir,
        "cost_evaluation_v2_summary.csv"
    )

    summary_df.to_csv(summary_path)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COST EVALUATION V2 COMPLETED")
    print("=" * 70)

    print("\nSUMMARY:")
    print(summary_df.to_string())

    print("\nFILES CREATED:")

    print(raw_path)

    print(summary_path)

    print("\nRSS MEMORY MEASUREMENTS:")
    print("rss_baseline_mb = RAM before Data-OOB")
    print("rss_peak_mb = maximum sampled process RAM")
    print("rss_final_mb = RAM after Data-OOB")
    print("rss_increase_mb = peak RSS minus baseline RSS")

    print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":

    main()
