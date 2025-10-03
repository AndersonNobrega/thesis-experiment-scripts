# Reproducibility Scripts: Data Augmentation for Non-Intrusive Load Monitoring

This repository contains the reproducibility package for the master's thesis titled "Data Augmentation for Non-Intrusive Load Monitoring: A Review and Empirical Study". It provides the necessary scripts and configuration files to replicate the experiments, generate the results, and create the plots presented in the dissertation.

**Important Note on Intellectual Property:** The source code for the custom data augmentation methods and the specific raw datasets used in the study are proprietary and **are not included** in this repository. However, the thesis provides detailed pseudocode for all algorithms (Section 4.2) and a complete description of the data characteristics (Section 4.1). This repository enables researchers to apply the same experimental pipeline to their own datasets.

## Project Structure

- `scripts/`: Holds the main shell scripts used to automate the training, evaluation, and running complete experiments.
- `conf/`: Contains all configuration file used for the different experimental setups (Baseline, Random Assignment, Synthetic Data, Merged Augmentation).
- `data/`: This directory is a **placeholder**. You must place your own NILM datasets here, following the expected format.
- `plots/`: This directory will be created automatically to store the plots and figures generated from the results.
- `results/`: This directory will be created automatically to store the raw output files from the evaluation scripts.
- `utils/`: Contains auxiliary Python scripts responsible for tasks such as parsing results, generating plots, and other data handling operations.

## Requirements

1. **Clone the repository:**

    ```bash
    git clone https://github.com/AndersonNobrega/thesis-experiment-scripts.git
    ```

2. **Set up Python Environment:** Ensure you have a working Python environment with all necessary dependencies for NILM analysis (e.g., NumPy, Pandas, TensorFlow, NILMTK).
3. **Configure Environment Variables:** The scripts rely on two environment variables. Please export them in your shell session:
    - `NIALM_PATH`: The absolute path to your Python virtual environment's `bin` directory.
    - `SET_PATH`: The absolute path to any setup scripts required by your NILM environment.
4. **Add Your Data:** Place your preprocessed NILM dataset(s) into the `data/` directory. The scripts expect the data to be in `.dat` format for use with WaveNILM.

## Usage

All automation scripts are located in the `scripts/` directory. They can be executed directly from the command line.

The scripts accept one of two flags to define the evaluation scenario:

- `--simple_eval`: Runs the **Intra-House** evaluation scenario.
- `--hard_eval`: Runs the **Inter-House** evaluation scenario. This is the default if no flag is provided.

### Main Scripts

- `train.sh`: Executes the model training process based on the configurations in the `conf/` directory.

    ```bash
    # Example: Train models for the Inter-House scenario
    ./scripts/train.sh --hard_eval
    ```

- `eval.sh`: Evaluates previously trained models.

    ```bash
    # Example: Evaluate models for the Intra-House scenario
    ./scripts/eval.sh --simple_eval
    ```

- `complete.sh`: Runs the entire pipeline in sequence: training followed by evaluation.

    ```bash
    # Example: Run the full Inter-House experiment
    ./scripts/complete.sh --hard_eval
    ```

- `create_plots.sh`: Runs all scripts for generating plots from the available results.

    ```bash
    # Example: Generate all plots
    ./scripts/create_plots.sh
    ```

## How to Cite

If you use this code or the experimental setup in your research, please cite the original dissertation:

```bibtex
@mastersthesis{Nobrega2025,
  author  = {Anderson N{\'o}brega Amorim},
  title   = {Data Augmentation for Non-Intrusive Load Monitoring: A Review and Empirical Study},
  school  = {Universidade Federal de Campina Grande},
  year    = {2025},
  address = {Campina Grande, Para{\'i}ba, Brasil}
}
```
