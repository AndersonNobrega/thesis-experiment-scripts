import json
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

from paths import PLOTS_PATH, RESULT_PATH

# Constants
INPUT_FILES = {
    "Merged": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/merged.json"
    ).as_posix(),
    "No Args": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/no_args.json"
    ).as_posix(),
    "Random Assign": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/random_assign.json"
    ).as_posix(),
    "Synthetic": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/synthetic.json"
    ).as_posix(),
}
PLOT_FIGSIZE = (16, 12)
MAX_LIMIT_BUFFER = 0.05  # 5% buffer

# Set seaborn theme
sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})


def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[0][2]


def process_bin_data(bin_data):
    bin_data.sort(key=lambda x: x[0])  # Sort by bin start
    bin_centers = [(start + end) / 2 for start, end, _ in bin_data]
    values = [val for _, _, val in bin_data]
    return bin_centers, values


def plot_histograms(input_files: List[Path], output_file: Path):
    all_x = []
    all_y = []
    processed_data = {}

    # Load and process all data
    for title, file_path in input_files.items():
        bin_data = load_data(file_path)
        bin_centers, values = process_bin_data(bin_data)
        processed_data[title] = (bin_centers, values)
        all_x.extend(bin_centers)
        all_y.extend(values)

    # Determine symmetric x-limits centered at 0
    max_abs_x = max(abs(min(all_x)), abs(max(all_x)))
    max_abs_x *= 1 + MAX_LIMIT_BUFFER  # add buffer

    xlim = (-max_abs_x, max_abs_x)

    # Determine y-limits
    max_y = max(all_y)
    ylim = (0, max_y * (1 + MAX_LIMIT_BUFFER))

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=PLOT_FIGSIZE)
    axes = axes.flatten()

    for ax, (title, (bin_centers, values)) in zip(axes, processed_data.items()):
        ax.fill_between(bin_centers, values, color="lightblue", alpha=0.6)
        ax.plot(bin_centers, values, linestyle="-", marker="")
        ax.set_title(title)
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(axis="x", zorder=0)

    plt.tight_layout()
    plt.savefig(output_file.as_posix(), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    plot_histograms(
        input_files=INPUT_FILES, output_file=PLOTS_PATH / "histogram_kernel.png"
    )


if __name__ == "__main__":
    main()
