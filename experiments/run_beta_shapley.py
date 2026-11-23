"""Run from the repo root, in this order:
    python -m src.data_loading                       # once: datasets + splits
    python -m experiments.run_beta_shapley smoke     # ~3 min: checks + time estimate
    python -m experiments.run_beta_shapley run       # full grid, hours (can be restarted)
    python -m experiments.run_beta_shapley analyze   # tables + plots
"""

import argparse
import logging
import os
import platform
import sys
import time
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"  # also for pyDVL's worker process
warnings.filterwarnings("ignore")
logging.getLogger("pydvl").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from scipy import stats

from src.data_loading import DATASETS, NOISE_RATES, load_split
from src.evaluation import REMOVAL_FRACTIONS, noisy_label_detection, point_removal
from src.methods.beta_shapley import compute_beta_shapley

OUT = os.path.join("results", "beta_shapley")
SIZES = (500, 2000)
SEEDS = (0, 1, 2)
ALPHA, BETA = 16, 1
PERMUTATIONS = 100
CHECKPOINTS = (5, 10, 25, 50, 100, 200)
STABILITY_NOISE = 0.10


def run_one(dataset, n, noise, seed, perms, out_dir):
    tag = f"{dataset}_n{n}_noise{int(round(noise * 100))}_seed{seed}"
    values_path = os.path.join(out_dir, "values", f"{tag}.npy")
    if os.path.exists(values_path):
        return None

    d = load_split(dataset, n, noise)
    values, info, snaps = compute_beta_shapley(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        alpha=ALPHA, beta=BETA, n_permutations=perms, seed=seed,
        checkpoints=CHECKPOINTS)
    p, r, f1 = noisy_label_detection(values, d["noisy_idx"])
    key = dict(dataset=dataset, n=n, noise=noise, seed=seed)

    removal = []
    for order in ("high_first", "low_first"):
        accs = point_removal(values, d["X_train"], d["y_train"], d["X_test"], d["y_test"], order)
        removal += [dict(key, order=order, fraction=f, test_acc=a)
                    for f, a in zip(REMOVAL_FRACTIONS, accs)]

    snaps[info["n_permutations"]] = values
    conv = [dict(key, permutations=c, f1=noisy_label_detection(v, d["noisy_idx"])[2],
                 spearman_vs_final=stats.spearmanr(v, values).correlation)
            for c, v in sorted(snaps.items())]

    # values file is saved last: it marks the run as done
    append_csv(os.path.join(out_dir, "runs.csv"),
               [dict(key, alpha=ALPHA, beta=BETA, precision=p, recall=r, f1=f1, **info)])
    append_csv(os.path.join(out_dir, "removal.csv"), removal)
    append_csv(os.path.join(out_dir, "convergence.csv"), conv)
    os.makedirs(os.path.dirname(values_path), exist_ok=True)
    np.save(values_path, values)
    return info, f1


def append_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def grid(args):
    # quality: all noise rates with seed 0, stability: seeds 1,2 at 10% noise
    # -> 5 runs per (dataset, n), also used as the 5 timing repetitions
    for n in args.sizes:
        for dataset in args.datasets:
            for noise in args.noise:
                yield dataset, n, noise, args.seeds[0]
            for seed in args.seeds[1:]:
                for noise in (args.noise if args.full_grid else [STABILITY_NOISE]):
                    yield dataset, n, noise, seed


def cmd_run(args):
    os.makedirs(OUT, exist_ok=True)
    save_environment(os.path.join(OUT, "environment.txt"))
    jobs = list(grid(args))
    t_start = time.time()
    for i, (dataset, n, noise, seed) in enumerate(jobs, 1):
        try:
            res = run_one(dataset, n, noise, seed, args.perms, OUT)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"[{i}/{len(jobs)}] {dataset} n={n} noise={noise:.0%} seed={seed}: "
                  f"FAILED, skipping (it will be retried on the next `run`)", flush=True)
            continue
        if res is None:
            print(f"[{i}/{len(jobs)}] {dataset} n={n} noise={noise:.0%} seed={seed}: already done")
            continue
        info, f1 = res
        print(f"[{i}/{len(jobs)}] {dataset} n={n} noise={noise:.0%} seed={seed}: "
              f"F1={f1:.2f} time={info['time_s']:.0f}s evals={info['utility_evals']} "
              f"peak={info['peak_mem_mb']:.0f}MB (total {(time.time() - t_start) / 3600:.1f}h)",
              flush=True)


def cmd_smoke(args):
    # 3 permutations per setting, then extrapolate the time of the full grid
    import shutil

    out = os.path.join("results", "beta_shapley_smoke")
    shutil.rmtree(out, ignore_errors=True)
    total_h = 0.0
    for dataset in args.datasets:
        for n in args.sizes:
            info, f1 = run_one(dataset, n, 0.10, 0, 3, out)
            per_eval = info["time_s"] / info["utility_evals"]
            run_h = per_eval * 2 * n * args.perms / 3600
            n_runs = len(args.noise) + (len(args.seeds) - 1) * (
                len(args.noise) if args.full_grid else 1)
            total_h += run_h * n_runs
            print(f"{dataset:12s} n={n:5d}: {1000 * per_eval:.2f} ms/eval -> "
                  f"~{60 * run_h:.0f} min per run, {n_runs} runs = {run_h * n_runs:.1f} h",
                  flush=True)
    print(f"\nEstimated full grid with {args.perms} permutations: ~{total_h:.1f} h")
    print("(rough: one run of 3 permutations per setting; real runs vary +-30%)")


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def cmd_analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = pd.read_csv(os.path.join(OUT, "runs.csv"))
    removal = pd.read_csv(os.path.join(OUT, "removal.csv"))
    conv = pd.read_csv(os.path.join(OUT, "convergence.csv"))
    keys = ["dataset", "n", "noise"]

    quality = runs.groupby(keys)[["precision", "recall", "f1"]].agg(["mean", "std"]).round(3)
    quality.to_csv(os.path.join(OUT, "table_quality.csv"))

    # Spearman between every pair of seeds (same data, different sampling)
    rows = []
    for (ds, n, noise), g in runs.groupby(keys):
        vals = {s: np.load(os.path.join(OUT, "values",
                                        f"{ds}_n{n}_noise{int(round(noise * 100))}_seed{s}.npy"))
                for s in g["seed"]}
        seeds = sorted(vals)
        rhos = [stats.spearmanr(vals[a], vals[b]).correlation
                for i, a in enumerate(seeds) for b in seeds[i + 1:]]
        if rhos:
            rows.append(dict(dataset=ds, n=n, noise=noise, pairs=len(rhos),
                             spearman_mean=np.mean(rhos), spearman_min=np.min(rhos)))
    stability = pd.DataFrame(rows).round(3)
    stability.to_csv(os.path.join(OUT, "table_stability.csv"), index=False)

    # cost does not depend on the noise rate -> all 5 runs of a (dataset, n) are timings
    cost = runs.groupby(["dataset", "n"]).agg(
        runs=("time_s", "size"),
        time_median_s=("time_s", "median"),
        time_mean_s=("time_s", "mean"),
        time_ci95_s=("time_s", ci95),
        utility_evals=("utility_evals", "median"),
        peak_mem_mb=("peak_mem_mb", "median"),
        peak_mem_ci95_mb=("peak_mem_mb", ci95)).round(2)
    cost.to_csv(os.path.join(OUT, "table_cost.csv"))

    print("Quality (noisy-label detection, mean/std over seeds)\n", quality, "\n")
    print("Stability (Spearman between seeds)\n", stability.to_string(index=False), "\n")
    print("Cost (error bars = 95% CI of the mean over noise x seed runs)\n", cost, "\n")

    # point removal, noise 10%, mean over seeds
    sizes = sorted(runs["n"].unique())
    dsets = list(runs["dataset"].unique())
    fig, axes = plt.subplots(len(sizes), len(dsets), figsize=(4 * len(dsets), 3.3 * len(sizes)),
                             squeeze=False)
    for i, n in enumerate(sizes):
        for j, ds in enumerate(dsets):
            ax = axes[i][j]
            g = removal[(removal.dataset == ds) & (removal.n == n) & np.isclose(removal.noise, 0.10)]
            for order, label in (("high_first", "remove high value first"),
                                 ("low_first", "remove low value first")):
                m = g[g.order == order].groupby("fraction")["test_acc"].mean()
                ax.plot(m.index, m.values, marker="o", ms=3, label=label)
            ax.set_title(f"{ds}, n={n}")
            ax.set_xlabel("fraction removed")
            ax.set_ylabel("test accuracy")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Beta Shapley ({ALPHA},{BETA}) point removal, noise 10%, mean of {len(SEEDS)} seeds")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_point_removal.png"), dpi=150)

    # convergence on one dataset, mean +- std over seeds
    ds0 = args.conv_dataset
    g = conv[(conv.dataset == ds0) & (conv.n == min(sizes)) & np.isclose(conv.noise, 0.10)]
    if len(g):
        agg = g.groupby("permutations").agg(f1=("f1", "mean"), f1_sd=("f1", "std"),
                                            rho=("spearman_vs_final", "mean"),
                                            rho_sd=("spearman_vs_final", "std"))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.errorbar(agg.index, agg.f1, yerr=agg.f1_sd, marker="o", capsize=3)
        a1.set_ylabel("noisy-label F1")
        a2.errorbar(agg.index, agg.rho, yerr=agg.rho_sd, marker="o", capsize=3)
        a2.set_ylabel("Spearman vs final values")
        for a in (a1, a2):
            a.set_xscale("log")
            a.set_xticks(agg.index)
            a.set_xticklabels([str(x) for x in agg.index])
            a.minorticks_off()
            a.set_xlabel("sampled permutations")
            a.grid(alpha=0.3)
        fig.suptitle(f"Beta Shapley convergence: {ds0}, n={min(sizes)}, noise 10% "
                     f"(error bars: std over seeds)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "plot_convergence.png"), dpi=150)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for ds in dsets:
        c = cost.loc[ds]
        a1.errorbar(c.index, c.time_mean_s, yerr=c.time_ci95_s, marker="o", capsize=3, label=ds)
        a2.plot(c.index, c.peak_mem_mb, marker="o", label=ds)
    a1.set_ylabel("wall-clock time (s)")
    a2.set_ylabel("peak memory (MB)")
    for a in (a1, a2):
        a.set_xlabel("training points n")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Beta Shapley cost (error bars: 95% CI of the mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_cost.png"), dpi=150)
    print(f"Tables and plots written to {OUT}")


def save_environment(path):
    import psutil
    import importlib.metadata as md

    lines = [f"date: {time.strftime('%Y-%m-%d %H:%M')}",
             f"os: {platform.platform()}",
             f"cpu: {platform.processor()}",
             f"physical cores: {psutil.cpu_count(logical=False)}, "
             f"logical: {psutil.cpu_count(logical=True)}",
             f"ram: {psutil.virtual_memory().total / 2**30:.1f} GB",
             f"python: {sys.version.split()[0]}"]
    for pkg in ("numpy", "scipy", "scikit-learn", "pandas", "pyDVL", "opendataval", "torch"):
        try:
            lines.append(f"{pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            pass
    lines.append(f"settings: alpha={ALPHA} beta={BETA} seeds={SEEDS} checkpoints={CHECKPOINTS}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["smoke", "run", "analyze"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    ap.add_argument("--noise", nargs="+", type=float, default=list(NOISE_RATES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--perms", type=int, default=PERMUTATIONS)
    ap.add_argument("--full-grid", action="store_true")  # all noise x all seeds
    ap.add_argument("--conv-dataset", default="electricity")
    args = ap.parse_args()
    {"smoke": cmd_smoke, "run": cmd_run, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__": 
    main()