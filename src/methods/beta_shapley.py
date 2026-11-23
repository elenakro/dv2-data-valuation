import math
import os
import threading
import time

import numpy as np
import psutil
from scipy.special import betaln, gammaln
from pydvl.utils import Dataset, Utility
from pydvl.value import MaxUpdates, compute_generic_semivalues
from pydvl.value.sampler import PermutationSampler

from src.models import SafeLogisticRegression


# pyDVL's compute_beta_shapley_semivalues overflows for n > ~1025:
# the sampler weight n*comb(n-1,k) is too large to convert to float.
# We compute coefficient * weight in log space and use a sampler with weight 1.

class UnitWeightPermutationSampler(PermutationSampler):
    @classmethod
    def weight(cls, n, subset_len):
        return 1.0


def log_beta_coefficient(alpha, beta):
    const = betaln(alpha, beta)

    def coefficient(n, k):
        j = k + 1
        log_w = betaln(j + beta - 1, n - j + alpha) - const
        log_comb = gammaln(n) - gammaln(k + 1) - gammaln(n - k)
        return float(np.exp(log_w + log_comb + math.log(n)))

    return coefficient


class MaxUpdatesWithSnapshots(MaxUpdates):
    # saves the running values at each checkpoint (for the convergence plot)

    def __init__(self, n_updates, checkpoints):
        super().__init__(n_updates)
        self.checkpoints = sorted(c for c in checkpoints if c < n_updates)
        self.snapshots = {}

    def _check(self, result):
        if len(result.counts) > 0:
            done = int(np.max(result.counts))
            for c in self.checkpoints:
                if c <= done and c not in self.snapshots:
                    v = np.zeros(len(result.values))
                    v[result.indices] = result.values
                    self.snapshots[c] = v
        return super()._check(result)


class PeakMemory:
    # peak RSS in MB, including child processes (pyDVL runs in a worker)

    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()

    def _rss(self):
        p = psutil.Process(os.getpid())
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total / 2**20

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline = self._rss()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._rss())


def compute_beta_shapley(X_train, y_train, X_val, y_val, alpha=16, beta=1,
                         n_permutations=50, seed=0, checkpoints=()):
    n_classes = len(np.unique(np.concatenate([y_train, y_val])))
    # pyDVL computes the utility on the "test" part of Dataset -> we give it the validation set
    data = Dataset(X_train, y_train, X_val, y_val)
    utility = Utility(SafeLogisticRegression(n_classes=n_classes), data,
                      scorer="accuracy", catch_errors=False)
    sampler = UnitWeightPermutationSampler(utility.data.indices, seed=seed)
    done = MaxUpdatesWithSnapshots(n_permutations, checkpoints)

    with PeakMemory() as mem:
        t0 = time.perf_counter()
        result = compute_generic_semivalues(
            sampler, utility, log_beta_coefficient(alpha, beta), done, n_jobs=1)
        elapsed = time.perf_counter() - t0

    values = np.zeros(len(X_train))
    values[result.indices] = result.values
    if np.std(values) == 0:
        raise RuntimeError("all values are equal - the utility is probably failing")

    info = {
        "time_s": elapsed,
        "utility_evals": 2 * int(np.sum(result.counts)),  # U(S+i) and U(S) per update
        "n_permutations": int(np.max(result.counts)),
        "peak_mem_mb": mem.peak,
        "baseline_mem_mb": mem.baseline,
    }
    return values, info, done.snapshots