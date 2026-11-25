"""Run from the repo root, in this order:
    python -m src.data_loading                       # once: datasets + splits
    python -m experiments.run_banzhaf      smoke     # Generate values for 12 configurations and save at experiments/banzhaf_results/
"""

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"  # also applies to pyDVL's worker process
warnings.filterwarnings("ignore")

import pickle
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil
from pydvl.utils.score import Scorer
from pydvl.utils.utility import Utility
from pydvl.value import Dataset
from pydvl.value.semivalues import compute_msr_banzhaf_semivalues
from pydvl.value.stopping import MaxTime, MinUpdates

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import DATASETS, NOISE_RATES, DATA_SEED, N_TEST, N_VAL, load_split
from src.models import SafeLogisticRegression
import numpy as np

N_BANZHAF_SAMPLES = 10000
N = [500, 2000]
BATCH_SIZE = 64
MAX_SECONDS = 12600
RESULTS_DIR = Path("experiments/banzhaf_results")

from concurrent.futures import Executor, Future

from pydvl.parallel.backend import CancellationPolicy, ParallelBackend


class InProcessExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:
            fut.set_exception(e)
        return fut

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass


class InProcessParallelBackend(ParallelBackend, backend_name="in_process"):
    @classmethod
    def executor(cls, max_workers=None, *, config=None,
                 cancel_futures=CancellationPolicy.PENDING):
        return InProcessExecutor()

    def get(self, v, *args, **kwargs):
        return v

    def put(self, v, *args, **kwargs):
        return v

    def wrap(self, fun, **kwargs):
        return fun

    def wait(self, v, *args, **kwargs):
        return list(v), []

    def _effective_n_jobs(self, n_jobs):
        return 1
    

counter=0
class CountingLogisticRegression(SafeLogisticRegression):
    """Increments a memory-mapped counter on every fit() (single worker only)."""

    def __init__(self, n_classes=2):
        super().__init__(n_classes=n_classes)
        self.n_classes = n_classes

    def fit(self, X, y, *args, **kwargs):
        counter += 1
        return super().fit(X, y, *args, **kwargs)

class PeakMemory:
    """Peak RSS in MB, including child processes."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()

    def _rss(self):
        proc = psutil.Process(os.getpid())
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total / 2**20

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline = self._rss()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._rss())


def run_experiment(noise_rate, dataset_name, n):
    print(f"\nLoading dataset: {dataset_name} (noise rate {noise_rate})")
    d = load_split(dataset_name, n, noise_rate)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    model = CountingLogisticRegression(n_classes=2)

    dataset = Dataset(x_train=X_train, y_train=y_train, x_test=X_val, y_test=y_val)
    scorer = Scorer("accuracy", default=0.0, range=(0.0, 1.0))
    utility = Utility(model, dataset, scorer)
    stopping = MinUpdates(n_updates=N_BANZHAF_SAMPLES) | MaxTime(seconds=MAX_SECONDS)

    print("Starting Banzhaf valuation...")
    with PeakMemory() as mem:
        start = time.perf_counter()
        valuation = compute_msr_banzhaf_semivalues(
            u=utility, done=stopping, batch_size=64, seed=DATA_SEED,
            progress=True, n_jobs=1,
            parallel_backend=InProcessParallelBackend(),
        )
        elapsed = time.perf_counter() - start

    n_fit_calls = counter

    print("Banzhaf valuation completed.")
    print(f"Updates (min/mean/max): {valuation.counts.min()} / "
          f"{valuation.counts.mean():.1f} / {valuation.counts.max()}")
    print(f"fit() calls: {n_fit_calls}")
    print(f"Execution time: {elapsed:.2f} s")

    filename = f"banzhaf_{n}_{N_BANZHAF_SAMPLES}_{noise_rate}_{dataset_name}.pkl"
    results = {
        "result": valuation,
        "execution_time_seconds": elapsed,
        "n_train": n,
        "n_n": N_VAL,
        "n_test": N_TEST,
        "n_noisy": int(round(noise_rate * len(y_train))),
        "x_train": X_train,
        "x_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "DATA_SEED": DATA_SEED,
        "noisy_idx": d["noisy_idx"],
        "peak_mem": mem.peak,
        "baseline_mem": mem.baseline,
        "fit_count": n_fit_calls,
    }
    with open(RESULTS_DIR / filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to: {filename}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for noise_rate in NOISE_RATES:
        for dataset_name in DATASETS:
            for n in N:
                run_experiment(noise_rate, dataset_name, n)


if __name__ == "__main__":
    main()