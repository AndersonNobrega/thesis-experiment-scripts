#!/usr/bin/env python3
"""
Aggregate training data from several houses and output appliance‐specific files.

Modes:
  * synthetic_modelling: aggregates original data and adds synthetic appliance data.
  * random_assign: aggregates original data and repeats it by a predefined factor.
  * merged: applies both random_assign and synthetic_modelling.
  * default: aggregates original data only.

Evaluation splits:
  * hard_eval (default): uses data from three houses.
  * simple_eval: uses data from five houses.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import os
import pickle
import numpy as np


HOUSE_FILES_HARD = {
    "casa_igor": Path("data/casa_igor/casa_igor_train.dat"),
    "casa_anderson": Path("data/casa_anderson/casa_anderson_train.dat"),
    "casa_leandro": Path("data/casa_leandro/casa_leandro_train.dat"),
}

FACTOR_BY_HOUSE_HARD = {
    "casa_igor": 40,
    "casa_anderson": 35,
    "casa_leandro": 22,
    "casa_jefferson": 15,
    "casa_mario": 10,
}

HOUSE_FILES_SIMPLE = {
    "casa_andrey": Path("data/casa_andrey/casa_andrey_train.dat"),
    "casa_igor": Path("data/casa_igor/casa_igor_train.dat"),
    "casa_anderson": Path("data/casa_anderson/casa_anderson_train.dat"),
    "casa_leandro": Path("data/casa_leandro/casa_leandro_train.dat"),
    "casa_diego": Path("data/casa_diego/casa_diego_train.dat"),
}

FACTOR_BY_HOUSE_SIMPLE = {
    "casa_andrey": 35,
    "casa_diego": 6,
    "casa_igor": 25,
    "casa_anderson": 20,
    "casa_leandro": 18,
    "casa_jefferson": 15,
    "casa_mario": 10,
}

SYNTHETIC_FILES = {
    "ar_condicionado": Path("data/casa_simulada/train_ar_condicionado.dat"),
    "chuveiro": Path("data/casa_simulada/train_chuveiro.dat"),
    "refrigerador": Path("data/casa_simulada/train_refrigerador.dat"),
}

OUTPUT_FILES = {
    "ar_condicionado": Path(
        "/home/anderson/temp/individual_appliances/residencial/train_ar_condicionado.dat"
    ),
    "chuveiro": Path(
        "/home/anderson/temp/individual_appliances/residencial/train_chuveiro.dat"
    ),
    "refrigerador": Path(
        "/home/anderson/temp/individual_appliances/residencial/train_refrigerador.dat"
    ),
}

APPLIANCE_INDICES = {
    "ar_condicionado": 2,
    "chuveiro": 3,
    "refrigerador": 4,
}


class DataAggregator:
    def __init__(self, house_files: Dict[str, Path], repeat_factor: Dict[str, int]):
        self._house_files = house_files
        self._repeat_factor = repeat_factor
        self._base_timestamp = datetime.strptime(
            "2020-02-16 14:30:00", "%Y-%m-%d %H:%M:%S"
        ).timestamp()

    def _load(self, path: Path) -> np.ndarray:
        with path.open("rb") as f:
            return pickle.load(f)

    def _save_aggregated(self, data: np.ndarray, appliance: str) -> None:
        out_path = OUTPUT_FILES[appliance]
        os.makedirs(out_path.parent, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(data, f)

    def _save_all_aggregate(
        self, data_map: Dict[str, np.ndarray], print_shapes: bool = False
    ) -> None:
        for appliance, arr in data_map.items():
            self._save_aggregated(arr, appliance)
            if print_shapes:
                print(f"{appliance} shape: {arr.shape}")

    def _repeat_array(
        self, repeat: int, values: np.ndarray, synthetic: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Repeat `values` `repeat` times, updating timestamps by +60s per row.
        If `synthetic`, only propagate timestamp and channel columns.
        """
        n_rows, _, _ = values.shape
        cols = values.shape[1] if not synthetic else 2 + 1
        expanded = np.empty((n_rows * repeat, cols, 2))
        idx = 0
        for _ in range(repeat):
            for row in values:
                self._base_timestamp += 60
                if synthetic:
                    expanded[idx] = [
                        [self._base_timestamp, self._base_timestamp],
                        row[1],
                        row[2]
                    ]
                else:
                    expanded[idx] = row
                    expanded[idx][0] = [self._base_timestamp, self._base_timestamp]
                idx += 1

        return expanded

    def _aggregate_data(self, repeats: Dict[str, int]) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for name, path in self._house_files.items():
            print(f"Processing {name} ...")
            data = self._load(path)
            augmented_arr = self._repeat_array(repeats.get(name, 1), data)
            chunks.append(augmented_arr)

        return np.concatenate(chunks, axis=0)

    def _load_synthetic(self) -> Dict[str, np.ndarray]:
        synth_data: Dict[str, np.ndarray] = {}
        for appliance, path in SYNTHETIC_FILES.items():
            data = self._load(path)
            augmented_arr = self._repeat_array(1, data, synthetic=True)
            synth_data[appliance] = augmented_arr

        return synth_data

    def _extract_channels(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        channels: Dict[str, np.ndarray] = {}
        for appliance, idx in APPLIANCE_INDICES.items():
            channels[appliance] = data[:, [0, 1, idx], :]
        return channels

    def mode_default(self) -> None:
        print("Running default mode...")
        repeats = {name: 1 for name in self._house_files}
        agg = self._aggregate_data(repeats)
        channels = self._extract_channels(agg)
        self._save_all_aggregate(channels, print_shapes=True)
        print("default mode completed.")

    def mode_random_assignment(self) -> None:
        print("Running random assignment mode...")
        repeats = {name: self._repeat_factor.get(name, 1) for name in self._house_files}
        agg = self._aggregate_data(repeats)
        channels = self._extract_channels(agg)
        self._save_all_aggregate(channels, print_shapes=True)
        print("random_assignment mode completed.")

    def mode_synthetic_modelling(self) -> None:
        print("Running synthetic_modelling mode...")
        agg = self._aggregate_data({name: 1 for name in self._house_files})
        channels = self._extract_channels(agg)
        synth = self._load_synthetic()
        final = {
            ap: np.concatenate((channels[ap], synth[ap]), axis=0) for ap in channels
        }
        self._save_all_aggregate(final, print_shapes=True)
        print("synthetic_modelling mode completed.")

    def mode_merged(self) -> None:
        print("Running merged mode (random assignment + synthetic data)...")
        repeats = {name: self._repeat_factor.get(name, 1) for name in self._house_files}
        agg = self._aggregate_data(repeats)
        channels = self._extract_channels(agg)
        synth = self._load_synthetic()
        final = {
            ap: np.concatenate((channels[ap], synth[ap]), axis=0) for ap in channels
        }
        self._save_all_aggregate(final, print_shapes=True)
        print("Merged mode completed.")


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Aggregate and transform house data for individual appliance training.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--synthetic_modelling",
        action="store_true",
        help="Add synthetic data after aggregation.",
    )
    parser.add_argument(
        "--random_assign",
        action="store_true",
        help="Repeat data using predefined factors.",
    )
    parser.add_argument(
        "--merged", action="store_true", help="Apply both synthetic and random assign."
    )
    parser.add_argument(
        "--simple_eval", action="store_true", help="Use data from five eval houses."
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    house_files = HOUSE_FILES_SIMPLE if args.simple_eval else HOUSE_FILES_HARD
    repeat_factor = FACTOR_BY_HOUSE_SIMPLE if args.simple_eval else FACTOR_BY_HOUSE_HARD

    aggregator = DataAggregator(house_files, repeat_factor)
    if args.synthetic_modelling:
        aggregator.mode_synthetic_modelling()
    elif args.random_assign:
        aggregator.mode_random_assignment()
    elif args.merged:
        aggregator.mode_merged()
    else:
        aggregator.mode_default()


if __name__ == "__main__":
    main()
