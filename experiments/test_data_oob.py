import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from calculate_data_oob import calculate_data_oob


def main():

    print("=" * 50)
    print("DATA-OOB TEST")
    print("=" * 50)

    # 1. Δημιουργία συνθετικού dataset
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    # 2. Διαχωρισμός δεδομένων
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("Dataset samples:", len(X))
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))

    # 3. Έλεγχος αρχικής ακρίβειας Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nLogistic Regression accuracy:", round(accuracy, 4))

    # 4. Υπολογισμός Data-OOB
    print("\nCalculating Data-OOB...")

    results = calculate_data_oob(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_estimators=100,
        max_samples=0.8,
        random_state=42,
        n_jobs=1,
    )

    values = results["values"]
    counts = results["counts"]

    # 5. Έλεγχος αποτελεσμάτων
    print("\nDATA-OOB RESULTS")
    print("-" * 50)

    print("Number of values:", len(values))
    print("Mean Data-OOB value:", round(float(np.mean(values)), 4))
    print("Minimum value:", float(np.min(values)))
    print("Maximum value:", float(np.max(values)))
    print("Mean OOB evaluation count:", round(float(np.mean(counts)), 2))
    print("Minimum OOB evaluation count:", int(np.min(counts)))
    print("Maximum OOB evaluation count:", int(np.max(counts)))

    print(
        "Samples without OOB evaluation:",
        int(np.sum(counts == 0)),
    )

    print(
        "Execution time:",
        round(results["execution_time"], 3),
        "seconds",
    )

    print("\nTEST COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
