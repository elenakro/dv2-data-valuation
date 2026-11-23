import numpy as np

from src.models import SafeLogisticRegression

REMOVAL_FRACTIONS = np.round(np.arange(0, 0.55, 0.05), 2)


def noisy_label_detection(values, noisy_idx):
    # flag the k lowest-valued points, k = number of noisy points
    k = len(noisy_idx)
    flagged = np.argsort(values, kind="stable")[:k]
    tp = len(np.intersect1d(flagged, noisy_idx))
    precision = tp / len(flagged) if len(flagged) else 0.0
    recall = tp / k if k else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return precision, recall, f1


def point_removal(values, X_train, y_train, X_test, y_test,
                  order="high_first", fractions=REMOVAL_FRACTIONS):
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    ranking = np.argsort(values, kind="stable")
    if order == "high_first":
        ranking = ranking[::-1]
    accs = []
    for f in fractions:
        keep = ranking[int(round(f * len(values))):]
        model = SafeLogisticRegression(n_classes=n_classes).fit(X_train[keep], y_train[keep])
        accs.append(float(np.mean(model.predict(X_test) == y_test)))
    return accs