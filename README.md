# DV2 - Data Valuation Benchmark

Semester project for Information Management Systems (NTUA, Fall 2026).

Benchmarking three data valuation methods on tabular and embedding datasets:

- Beta Shapley (Kwon & Zou, AISTATS 2022)
- Data Banzhaf (Wang & Jia, AISTATS 2023)
- Data-OOB (Kwon & Zou, ICML 2023)

Utility model: logistic regression.
Datasets: 2 tabular (OpenML) + 1 embeddings (CIFAR10 or IMDB).

## Team

- TBD: Beta Shapley, repo setup, data pipeline
- TBD: Data Banzhaf, evaluation module
- TBD: Data-OOB, scaling and quality-cost analysis

## Setup

See requirements.txt. Python 3.11 recommended.
Create a venv, activate it, run: pip install -r requirements.txt

## Repo structure

- src/            source code
- experiments/    experiment scripts
- results/        gitignored outputs
- data/           gitignored datasets
- notebooks/      EDA and plots
- report/         final report

## Workflow

- main is always working. No direct pushes.
- Each member works on a feature branch: feature/<name>.
- Pull requests reviewed by one other member before merge.
- Final timing experiments run on one agreed machine.
