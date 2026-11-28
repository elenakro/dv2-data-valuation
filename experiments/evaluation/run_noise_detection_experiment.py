import sys
from pathlib import Path
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from calculate_data_oob import calculate_data_oob
from data.inject_noise import inject_label_noise
from evaluation.evaluate_noise_detection import evaluate_noise_detection


def run_experiment(noise_rate, random_state=42):

    print("\n" + "=" * 60)
    print(f"DATA-OOB NOISE DETECTION - NOISE {noise_rate * 100:.0f}%")
    print("=" * 60)

    # 1. Δημιουργία dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        random_state=random_state,
    )

    # 2. Διαχωρισμός τελικού test set (25%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    # 3. Διαχωρισμός training (60%) και validation (15%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=random_state,
        stratify=y_train_val,
    )

    # 4. Εισαγωγή θορύβου μόνο στο training set
    y_train_noisy, noisy_indices, original_labels = inject_label_noise(
        y_train,
        noise_rate=noise_rate,
        random_state=random_state,
    )

    print(f"Training samples:   {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples:       {len(y_test)}")
    print(f"Noisy samples:      {len(noisy_indices)}")

    # 5. Υπολογισμός Data-OOB με validation set
    start_time = time.perf_counter()

    oob_results = calculate_data_oob(
        X_train=X_train,
        y_train=y_train_noisy,
        X_test=X_val,
        y_test=y_val,
        n_estimators=100,
        max_samples=0.8,
        random_state=random_state,
        n_jobs=1,
    )

    # 6. Αξιολόγηση Noise Detection
    evaluation_results = evaluate_noise_detection(
        values=oob_results["values"],
        valuation_indices=oob_results["indices"],
        noisy_indices=noisy_indices,
    )

    # 7. Ανεξάρτητη τελική αξιολόγηση
    final_model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    final_model.fit(X_train, y_train_noisy)

    test_predictions = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    total_time = time.perf_counter() - start_time

    print("\nRESULTS")
    print("-" * 60)

    print(f"Precision:       {evaluation_results['precision']:.4f}")
    print(f"Recall:          {evaluation_results['recall']:.4f}")
    print(f"F1-score:        {evaluation_results['f1_score']:.4f}")
    print(f"Test Accuracy:   {test_accuracy:.4f}")
    print(f"Execution time:  {total_time:.2f} seconds")

    return {
        "noise_rate": noise_rate,
        "precision": evaluation_results["precision"],
        "recall": evaluation_results["recall"],
        "f1_score": evaluation_results["f1_score"],
        "test_accuracy": test_accuracy,
        "execution_time": total_time,
    }


def main():

    noise_rates = [0.05, 0.10, 0.20]

    all_results = []

    for rate in noise_rates:

        results = run_experiment(
            noise_rate=rate,
            random_state=42,
        )

        all_results.append(results)

    print("\nFINAL SUMMARY")
    print("=" * 85)

    print(
        f"{'Noise':<12}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'F1-score':<14}"
        f"{'Accuracy':<14}"
        f"{'Time':<12}"
    )

    print("-" * 85)

    for result in all_results:

        print(
            f"{result['noise_rate'] * 100:>5.0f}%"
            f"{result['precision']:>14.4f}"
            f"{result['recall']:>14.4f}"
            f"{result['f1_score']:>14.4f}"
            f"{result['test_accuracy']:>14.4f}"
            f"{result['execution_time']:>12.2f}"
        )


if __name__ == "__main__":
    main()
