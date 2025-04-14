#!/usr/bin/env python3
"""
This module aggregates training data from several houses and outputs appliance‐specific files.
It supports three modes:
    * synthetic_modelling: aggregates original data from five houses and then adds synthetic appliance data.
    * random_assignment: aggregates original data from five houses and “increases” them by a provided factor.
    * signal_transform: (not implemented yet – simply aggregates the original data)
If no mode is provided, then the original data is simply aggregated.
"""

from datetime import datetime
import os
import pickle
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple, List

# Original house data for aggregation (for casa_igor, casa_anderson, casa_leandro, casa_jefferson, casa_mario)
FACTOR_BY_HOUSE = {
    "casa_igor": 40,
    "casa_anderson": 35,
    "casa_leandro": 22,
    "casa_jefferson": 15,
    "casa_mario": 10,
}

HOUSE_FILES = {
    "casa_igor": Path("data/casa_igor/casa_igor_train.dat"),
    "casa_anderson": Path("data/casa_anderson/casa_anderson_train.dat"),
    "casa_leandro": Path("data/casa_leandro/casa_leandro_train.dat"),
    # "casa_jefferson": Path("data/casa_jefferson/casa_jefferson_train.dat"),
    # "casa_mario": Path("data/casa_mario/casa_mario_train.dat"),
}

# Synthetic appliance data files (used in synthetic_modelling mode)
SYNTHETIC_FILES = {
    "ar_condicionado": Path("data/casa_simulada/train_ar_condicionado.dat"),
    "chuveiro": Path("data/casa_simulada/train_chuveiro.dat"),
    "refrigerador": Path("data/casa_simulada/train_refrigerador.dat"),
}

# Output paths for appliance-specific aggregated data
OUTPUT_FILES = {
    "ar_condicionado": Path("/home/anderson/temp/individual_appliances/residencial/train_ar_condicionado.dat"),
    "chuveiro": Path("/home/anderson/temp/individual_appliances/residencial/train_chuveiro.dat"),
    "refrigerador": Path("/home/anderson/temp/individual_appliances/residencial/train_refrigerador.dat"),
}


def load_pickle(file_path: Path) -> np.ndarray:
    """Load a pickle file from the given path."""
    with file_path.open("rb") as f:
        return pickle.load(f)


def save_pickle(data: np.ndarray, file_path: Path) -> None:
    """Save the given data to a pickle file at the given path."""
    os.makedirs(file_path.parent, exist_ok=True)
    with file_path.open("wb") as f:
        pickle.dump(data, f)


def increase_arr(repeat_value: int, initial_timestamp: float, values: np.ndarray, synthetic: bool = False) -> Tuple[np.ndarray, float]:
    """
    Increase (repeat) the array values by a factor.

    The function creates an expanded array with new timestamps.
    The new array shape is (n_rows * repeat_value, 5, 2). For each repeat, the timestamp is incremented by 60 seconds per row.
    """
    start = initial_timestamp
    new_arr = np.empty((values.shape[0] * repeat_value, values.shape[1], values.shape[2]))
    cont = 0

    for i in range(repeat_value):
        for j in range(values.shape[0]):
            current_timestamp = start + (60 * (j + 1))
            if synthetic:
                new_arr[cont] = [
                    [current_timestamp, current_timestamp],
                    values[j][1],
                    values[j][2],
                ]
            else:
                new_arr[cont] = [
                    [current_timestamp, current_timestamp],
                    values[j][1],
                    values[j][2],
                    values[j][3],
                    values[j][4],
                ]
            cont += 1
        # Update start to the last timestamp of this block
        start = current_timestamp

    return new_arr, start


def extract_appliance_channels(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given data with shape (n, 5, 2), extract appliance-specific channels.
    We assume:
      - Column 0: timestamp (both elements)
      - Column 1: common channel (e.g. aggregate power)
      - Column 2: ar_condicionado
      - Column 3: chuveiro
      - Column 4: refrigerador
    """
    arr_ar_condicionado = data[:, [0, 1, 2], :]
    arr_chuveiro = data[:, [0, 1, 3], :]
    arr_refrigerador = data[:, [0, 1, 4], :]
    return arr_ar_condicionado, arr_chuveiro, arr_refrigerador


def aggregate_house_data(file_paths: List[Path]) -> np.ndarray:
    """
    Load and concatenate data from a list of file paths.
    Assumes that each file is a pickle file containing an array of shape (n, 5, 2).
    """
    data_list = [load_pickle(fp) for fp in file_paths]
    # Concatenate along axis 0 (time axis)
    return np.concatenate(data_list, axis=0)

def mode_synthetic_modelling() -> None:
    """
      1. Load the original house data from the five houses.
      2. Aggregate the data.
      3. Extract appliance-specific channels.
      4. Load synthetic appliance data and concatenate it to the corresponding channel.
      5. Save the final arrays.
    """
    print("Running synthetic_modelling mode...")
    # Load original data from houses
    factor = 1
    increased_data_list = []
    date_str = "2020-02-16 14:30:00"
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    curr_timestamp = dt.timestamp()
    for house_name, file_path in HOUSE_FILES.items():
        print(f"Processing {house_name} ...")
        data = load_pickle(file_path)
        increased_data, curr_timestamp = increase_arr(factor, curr_timestamp, data)
        increased_data_list.append(increased_data)

    # Aggregate the increased data from all houses
    aggregated_increased_data = np.concatenate(increased_data_list, axis=0)
    arr_ar, arr_ch, arr_ref = extract_appliance_channels(aggregated_increased_data)

    # Load synthetic data for each appliance
    synthetic_ar = load_pickle(SYNTHETIC_FILES["ar_condicionado"])
    synthetic_ch = load_pickle(SYNTHETIC_FILES["chuveiro"])
    synthetic_ref = load_pickle(SYNTHETIC_FILES["refrigerador"])
    
    synthetic_ar, _ = increase_arr(1, curr_timestamp, synthetic_ar, True)
    synthetic_ch, _ = increase_arr(1, curr_timestamp, synthetic_ch, True)
    synthetic_ref, _ = increase_arr(1, curr_timestamp, synthetic_ref, True)

    # Concatenate synthetic data to original channels along time axis
    final_ar = np.concatenate((arr_ar, synthetic_ar), axis=0)
    final_ch = np.concatenate((arr_ch, synthetic_ch), axis=0)
    final_ref = np.concatenate((arr_ref, synthetic_ref), axis=0)

    # Save outputs
    save_pickle(final_ar, OUTPUT_FILES["ar_condicionado"])
    save_pickle(final_ch, OUTPUT_FILES["chuveiro"])
    save_pickle(final_ref, OUTPUT_FILES["refrigerador"])
    
    print(f"Ar condicionado shape: {final_ar.shape}")
    print(f"Chuveiro shape: {final_ch.shape}")
    print(f"Refrigerador shape: {final_ref.shape}")
    print("synthetic_modelling mode completed.")


def mode_random_assignment(factor: int) -> None:
    """
      1. Load the original house data for each house.
      2. For each house, increase (repeat) the data by the provided factor.
      3. Aggregate the increased arrays.
      4. Extract appliance-specific channels.
      5. Save the final arrays.
    """
    print("Running random_assignment mode with factor =", factor)
    increased_data_list = []
    date_str = "2020-02-16 14:30:00"
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    curr_timestamp = dt.timestamp()
    for house_name, file_path in HOUSE_FILES.items():
        print(f"Processing {house_name} ...")
        data = load_pickle(file_path)
        increased_data, curr_timestamp = increase_arr(FACTOR_BY_HOUSE[house_name], curr_timestamp, data)
        increased_data_list.append(increased_data)

    # Aggregate the increased data from all houses
    aggregated_increased_data = np.concatenate(increased_data_list, axis=0)
    arr_ar, arr_ch, arr_ref = extract_appliance_channels(aggregated_increased_data)

    # Save outputs
    save_pickle(arr_ar, OUTPUT_FILES["ar_condicionado"])
    save_pickle(arr_ch, OUTPUT_FILES["chuveiro"])
    save_pickle(arr_ref, OUTPUT_FILES["refrigerador"])

    print(f"Ar condicionado shape: {arr_ar.shape}")
    print(f"Chuveiro shape: {arr_ch.shape}")
    print(f"Refrigerador shape: {arr_ref.shape}")
    print("random_assignment mode completed.")


def mode_signal_transform() -> None:
    print("Running signal_transform mode (no transformation implemented yet)...")
    original_data = aggregate_house_data(list(HOUSE_FILES.values()))
    arr_ar, arr_ch, arr_ref = extract_appliance_channels(original_data)

    save_pickle(arr_ar, OUTPUT_FILES["ar_condicionado"])
    save_pickle(arr_ch, OUTPUT_FILES["chuveiro"])
    save_pickle(arr_ref, OUTPUT_FILES["refrigerador"])

    print(f"Ar condicionado shape: {arr_ar.shape}")
    print(f"Chuveiro shape: {arr_ch.shape}")
    print(f"Refrigerador shape: {arr_ref.shape}")
    print("signal_transform mode completed (no transformation applied).")


def mode_default() -> None:
    """
    Simply aggregate the original data from the houses,
    extract appliance channels, and save the results.
    """
    print("Running default mode (original aggregation)...")
    factor = 1
    increased_data_list = []
    date_str = "2020-02-16 14:30:00"
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    curr_timestamp = dt.timestamp()
    for house_name, file_path in HOUSE_FILES.items():
        print(f"Processing {house_name} ...")
        data = load_pickle(file_path)
        increased_data, curr_timestamp = increase_arr(factor, curr_timestamp, data)
        increased_data_list.append(increased_data)

    aggregated_increased_data = np.concatenate(increased_data_list, axis=0)
    arr_ar, arr_ch, arr_ref = extract_appliance_channels(aggregated_increased_data)

    save_pickle(arr_ar, OUTPUT_FILES["ar_condicionado"])
    save_pickle(arr_ch, OUTPUT_FILES["chuveiro"])
    save_pickle(arr_ref, OUTPUT_FILES["refrigerador"])

    print(f"Ar condicionado shape: {arr_ar.shape}")
    print(f"Chuveiro shape: {arr_ch.shape}")
    print(f"Refrigerador shape: {arr_ref.shape}")
    print("Default mode completed.")


def mode_merged() -> None:
    """
    Merged mode:
      1. Increase (repeat) each house’s data by the provided factor (random assignment).
      2. Aggregate the increased data and extract appliance-specific channels.
      3. Load the synthetic data for each appliance and concatenate it along the time axis.
      4. Save the final aggregated arrays.
    """
    print("Running merged mode (random assignment + synthetic data)...")
    increased_data_list = []
    date_str = "2020-02-16 14:30:00"
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    curr_timestamp = dt.timestamp()
    for house_name, file_path in HOUSE_FILES.items():
        print(f"Processing {house_name} ...")
        data = load_pickle(file_path)
        increased_data, curr_timestamp = increase_arr(FACTOR_BY_HOUSE[house_name], curr_timestamp, data)
        increased_data_list.append(increased_data)

    # Aggregate the increased data from all houses
    aggregated_increased_data = np.concatenate(increased_data_list, axis=0)
    arr_ar, arr_ch, arr_ref = extract_appliance_channels(aggregated_increased_data)

    # Load synthetic data for each appliance
    synthetic_ar = load_pickle(SYNTHETIC_FILES["ar_condicionado"])
    synthetic_ch = load_pickle(SYNTHETIC_FILES["chuveiro"])
    synthetic_ref = load_pickle(SYNTHETIC_FILES["refrigerador"])
    
    synthetic_ar, _ = increase_arr(1, curr_timestamp, synthetic_ar, True)
    synthetic_ch, _ = increase_arr(1, curr_timestamp, synthetic_ch, True)
    synthetic_ref, _ = increase_arr(1, curr_timestamp, synthetic_ref, True)

    # Concatenate synthetic data to the random assignment result
    final_ar = np.concatenate((arr_ar, synthetic_ar), axis=0)
    final_ch = np.concatenate((arr_ch, synthetic_ch), axis=0)
    final_ref = np.concatenate((arr_ref, synthetic_ref), axis=0)

    # Save outputs
    save_pickle(final_ar, OUTPUT_FILES["ar_condicionado"])
    save_pickle(final_ch, OUTPUT_FILES["chuveiro"])
    save_pickle(final_ref, OUTPUT_FILES["refrigerador"])

    print(f"Ar condicionado shape: {final_ar.shape}")
    print(f"Chuveiro shape: {final_ch.shape}")
    print(f"Refrigerador shape: {final_ref.shape}")
    print("Merged mode completed.")


def get_args() -> Namespace:
    parser = ArgumentParser(
        allow_abbrev=False,
        description="Aggregate and transform house data for individual appliance training.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--synthetic_modelling",
        action="store_true",
        help="Aggregate original house data and then add synthetic appliance data.",
    )
    parser.add_argument(
        "--random_assign",
        action="store_true",
        help="Aggregate original house data and increase it by a provided integer factor.",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Apply synthetic data and random assign.",
    )
    parser.add_argument(
        "--signal_transform",
        action="store_true",
        help="Apply signal transformation (not implemented yet).",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    if args.synthetic_modelling:
        mode_synthetic_modelling()
    elif args.random_assign:
        mode_random_assignment(15)
    elif args.signal_transform:
        mode_signal_transform()
    elif args.merged:
        mode_merged()
    else:
        mode_default()


if __name__ == "__main__":
    main()
