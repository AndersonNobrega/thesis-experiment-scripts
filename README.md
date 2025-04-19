# NILM Experiment Scripts

This repository contains scripts and configuration files used to generate results for my master's thesis on Non-Intrusive Load Monitoring (NILM).

## Project Structure

- `scripts/`: Shell scripts for training, evaluation, and running complete experiments.
- `conf/`: Configuration files for training and evaluation.
- `data/`: Input datasets, including real and simulated household data.
- `plots/`: Output plots generated from evaluation results.
- `results/`: Evaluation output (created automatically).
- `utils/`: Helper Python scripts for data augmentation, result extraction, and visualization.

## Requirements

- Python environment with necessary NILM dependencies.
- Environment variables:
  - `NIALM_PATH`: Path to the Python virtual environment.
  - `SET_PATH`: Path to NILM setup scripts.
- Datasets stored under the `data/` directory.

## Scripts

All sh scripts used to automate the experiments are located in the `scripts/` directory.

- `train.sh`: Trains models using specified configurations and augmentation methods.
- `eval.sh`: Evaluates pretrained models across multiple runs and augmentation strategies.
- `complete.sh`: Runs both training and evaluation in sequence.

## How to Run

```bash
cd scripts/
./eval.sh [--simple_eval | --hard_eval]
./train.sh [--simple_eval | --hard_eval]
./complete.sh
```

Defaults to `--hard_eval` if no argument is provided.