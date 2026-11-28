import time

from sklearn.linear_model import LogisticRegression

from pydvl.utils import Dataset, Utility
from pydvl.value.oob.oob import compute_data_oob


def calculate_data_oob(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=100,
    max_samples=0.8,
    random_state=42,
    n_jobs=1,
):
    """
    Υπολογίζει τις τιμές Data-OOB για ένα σύνολο δεδομένων.

    Parameters
    ----------
    X_train : array-like
        Δεδομένα εκπαίδευσης.

    y_train : array-like
        Ετικέτες εκπαίδευσης.

    X_test : array-like
        Δεδομένα αξιολόγησης.

    y_test : array-like
        Ετικέτες αξιολόγησης.

    n_estimators : int
        Αριθμός Bagging estimators.

    max_samples : float
        Ποσοστό δειγμάτων ανά bootstrap estimator.

    random_state : int
        Σταθερός τυχαίος σπόρος.

    n_jobs : int
        Αριθμός παράλληλων διεργασιών.

    Returns
    -------
    dict
        Τιμές Data-OOB, μετρητές OOB και χρόνος εκτέλεσης.
    """

    start_time = time.perf_counter()

    dataset = Dataset(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    utility = Utility(
        model=model,
        data=dataset,
        scorer="accuracy",
    )

    result = compute_data_oob(
        utility,
        n_est=n_estimators,
        max_samples=max_samples,
        n_jobs=n_jobs,
        seed=random_state,
        progress=False,
    )

    execution_time = time.perf_counter() - start_time

    values = result.values
    counts = result.counts
    indices = result.indices

    return {
        "values": values,
        "counts": counts,
        "indices": indices,
        "execution_time": execution_time,
        "result": result,
    }
