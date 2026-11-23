"""Run once: python -m src.data_loading
Writes data/splits/<dataset>_n<size>_noise<pct>.npz, used by all methods."""

import os

import numpy as np

DATA_DIR = "data"
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
DATA_SEED = 0
N_TEST, N_VAL = 1000, 500
TRAIN_SIZES = (500, 2000, 10000)
NOISE_RATES = (0.05, 0.10, 0.20)

OPENML_IDS = {"electricity": 44080, "2dplanes": 727}  # same IDs as OpenDataVal
DATASETS = ("electricity", "2dplanes", "imdb")

IMDB_URL = ("https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/"
            "master/IMDB-Dataset.csv")  # same source as OpenDataVal
IMDB_POOL = N_TEST + N_VAL + max(TRAIN_SIZES)
# logistic regression on 768 dims is ~30x slower, too slow for the sampling methods
IMDB_PCA_DIMS = 64


def inject_label_noise(y, noise_rate, n_classes, rng):
    n_noisy = int(round(noise_rate * len(y)))
    noisy_idx = np.sort(rng.choice(len(y), size=n_noisy, replace=False))
    y_noisy = y.copy()
    for i in noisy_idx:
        other = [c for c in range(n_classes) if c != y[i]]
        y_noisy[i] = rng.choice(other)
    return y_noisy, noisy_idx


def standardize(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


def load_openml(name):
    from sklearn.datasets import fetch_openml

    d = fetch_openml(data_id=OPENML_IDS[name], as_frame=False, parser="auto")
    X, y = d["data"], d["target"]
    cats = list((d.get("categories") or {}).keys())
    if cats:  # OpenDataVal also drops categorical columns
        keep = [i for i, f in enumerate(d["feature_names"]) if f not in cats]
        X = X[:, keep]
    X = np.asarray(X, dtype=np.float64)
    _, y = np.unique(y, return_inverse=True)
    return standardize(X), y


def embed_distilbert(texts, batch_size=64):
    # same model and settings as OpenDataVal's imdb-embeddings
    import torch
    from transformers import DistilBertModel, DistilBertTokenizerFast

    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").eval()
    out = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        inputs = tok(batch, max_length=200, padding=True, truncation=True,
                     return_tensors="pt")
        with torch.no_grad():
            out.append(model(**inputs)[0][:, 0].numpy())
        print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(out)


def load_imdb():
    # only the IMDB_POOL rows we use are embedded (all 50k would take hours on CPU)
    cache = os.path.join(DATA_DIR, f"imdb_distilbert_{IMDB_POOL}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        X, y = d["X"], d["y"]
    else:
        import pandas as pd

        print("Downloading IMDB reviews...", flush=True)
        df = pd.read_csv(IMDB_URL)
        rows = np.random.default_rng(DATA_SEED).permutation(len(df))[:IMDB_POOL]
        texts = df["review"].values[rows]
        y = (df["sentiment"].values[rows] == "positive").astype(int)
        print(f"Embedding {IMDB_POOL} reviews with DistilBERT (CPU, ~10-20 min)...", flush=True)
        X = embed_distilbert(texts)
        os.makedirs(DATA_DIR, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y)

    from sklearn.decomposition import PCA

    pca = PCA(IMDB_PCA_DIMS, random_state=DATA_SEED).fit(X[N_TEST + N_VAL:])  # train rows only
    print(f"[imdb] PCA {X.shape[1]} -> {IMDB_PCA_DIMS} dims keeps "
          f"{pca.explained_variance_ratio_.sum():.0%} of the variance", flush=True)
    return standardize(pca.transform(X)), y


def load_dataset(name):
    if name == "imdb":
        X, y = load_imdb()
        order = np.arange(len(y))  # rows are already shuffled
    else:
        X, y = load_openml(name)
        order = np.random.default_rng(DATA_SEED).permutation(len(y))
    return X, y, order


def split_path(name, n, noise):
    return os.path.join(SPLIT_DIR, f"{name}_n{n}_noise{int(round(noise * 100))}.npz")


def prepare_splits(datasets=DATASETS):
    # test = first 1000 shuffled rows, val = next 500, train = next n.
    # So val/test are the same for every n, and train(500) is inside train(2000).
    # Noise only on training labels.
    os.makedirs(SPLIT_DIR, exist_ok=True)
    for name in datasets:
        print(f"[{name}] loading", flush=True)
        X, y, order = load_dataset(name)
        n_classes = len(np.unique(y))
        test, val = order[:N_TEST], order[N_TEST:N_TEST + N_VAL]
        train_pool = order[N_TEST + N_VAL:]
        print(f"[{name}] {X.shape[0]} rows, {X.shape[1]} features, {n_classes} classes", flush=True)
        for n in TRAIN_SIZES:
            if n > len(train_pool):
                print(f"[{name}] skip n={n}: only {len(train_pool)} training rows")
                continue
            train = train_pool[:n]
            for noise in NOISE_RATES:
                rng = np.random.default_rng([DATA_SEED, n, int(round(noise * 100))])
                y_noisy, noisy_idx = inject_label_noise(y[train], noise, n_classes, rng)
                np.savez_compressed(
                    split_path(name, n, noise),
                    X_train=X[train], y_train=y_noisy, y_train_clean=y[train],
                    noisy_idx=noisy_idx,
                    X_val=X[val], y_val=y[val], X_test=X[test], y_test=y[test])
        print(f"[{name}] splits written", flush=True)


def load_split(name, n, noise):
    path = split_path(name, n, noise)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run: python -m src.data_loading")
    return dict(np.load(path))


if __name__ == "__main__":
    prepare_splits()