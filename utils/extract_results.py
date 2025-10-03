import argparse
import json
import re
from pathlib import Path

import numpy as np

# List of subscenario identifiers to collect
SUBSCENARIOS = [
    "casa_diego",
    "casa_andrey",
    "casa_anderson",
    "casa_leandro",
    "casa_igor",
]

# Regex to identify subscenario lines
SUB_PATTERN = re.compile(r"Subsc?enario:\s*(\w+)", re.IGNORECASE)

# Patterns for all required metrics
PATTERNS = {
    "total_accuracy": re.compile(r"estimated accuracy \(window size 1\): ([\d\.]+)%"),
    "total_ar_condicionado": re.compile(r"appliance ar_condicionado: ([\d\.]+)%"),
    "total_chuveiro": re.compile(r"appliance chuveiro: ([\d\.]+)%"),
    "total_refrigerador": re.compile(r"appliance refrigerador: ([\d\.]+)%"),
}


def process_file(txt_path: Path) -> dict:
    """
    Extract metrics from a single .txt file and compute averages per subscenario.
    """
    metrics = {sub: {key: [] for key in PATTERNS} for sub in SUBSCENARIOS}

    with txt_path.open("r") as file:
        current_subscenario = None
        for line in file:
            if "Subcenario:  casa_diego" in line:
                current_subscenario = "casa_diego"
            elif "Subcenario:  casa_andrey" in line:
                current_subscenario = "casa_andrey"
            elif "Subcenario:  casa_anderson" in line:
                current_subscenario = "casa_anderson"
            elif "Subcenario:  casa_igor" in line:
                current_subscenario = "casa_igor"
            elif "Subcenario:  casa_leandro" in line:
                current_subscenario = "casa_leandro"
            elif "Subscenario: casa_tipo1" in line:
                current_subscenario = None

            if current_subscenario:
                for key, pattern in PATTERNS.items():
                    match_val = pattern.search(line)
                    if match_val:
                        metrics[current_subscenario][key].append(
                            float(match_val.group(1))
                        )

    averages = {
        sub: {
            metric: values if values else None
            for metric, values in data.items()
        }
        for sub, data in metrics.items()
    }

    return averages


def main():
    parser = argparse.ArgumentParser(
        description="Process .txt result files to extract and average metrics per subscenario."
    )
    parser.add_argument(
        "--dir", type=Path, help="Path to directory containing .txt result files"
    )
    args = parser.parse_args()

    if not args.dir.is_dir():
        parser.error(f"Path '{args.dir}' is not a directory.")

    for txt_file in args.dir.glob("*.txt"):
        print(f"Processing: {txt_file.name}")
        averages = process_file(txt_file)

        out_path = txt_file.with_suffix(".json")
        with out_path.open("w") as out_f:
            json.dump(averages, out_f, indent=2)
        print(f"Written: {out_path.name}\n")


if __name__ == "__main__":
    main()
