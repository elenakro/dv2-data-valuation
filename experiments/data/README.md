# Data Preparation

## 1. Purpose

This directory stores the datasets used by the Data-OOB implementation.

## 2. Directory Structure

- `raw/`: Original datasets, downloaded without modifications.
- `processed/`: Prepared datasets, including training and test splits.

## 3. Data Preparation Procedure

The preparation procedure will include:

1. Loading the original dataset.
2. Checking missing values and data types.
3. Separating features and target labels.
4. Encoding categorical features when necessary.
5. Creating reproducible training and test splits.
6. Saving the processed datasets.
7. Applying controlled label noise when required by the experiments.

## 4. Reproducibility

All random operations must use explicitly defined random seeds.

The same training and test splits must be used across all experiments to ensure fair evaluation.

## 5. Label Noise

The experiments will consider label-noise rates of:

- 5%
- 10%
- 20%

The original labels must be preserved so that noisy-label detection can be evaluated against the known corrupted samples.

## 6. Current Status

The directory structure has been created. Dataset loading and preprocessing scripts are pending.
## 7. Implemented Scripts

### load_datasets.py

Loads a CSV dataset, separates features and target labels, and creates stratified training and test sets.

### inject_noise.py

Introduces controlled label noise into classification datasets.

Supported noise rates:

- 5%
- 10%
- 20%

The function preserves the original labels and returns the indices of corrupted samples.

### test_inject_noise.py

Tests the label-noise implementation using a synthetic dataset of 1,000 samples.

## 8. Test Results

The label-noise implementation was successfully tested.

| Noise Rate | Expected Corrupted Samples | Actual Corrupted Samples |
|---|---:|---:|
| 5% | 50 | 50 |
| 10% | 100 | 100 |
| 20% | 200 | 200 |

All tests passed successfully.
