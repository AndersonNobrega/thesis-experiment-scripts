from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from paths import PLOTS_PATH, RESULT_PATH

# Constants
INPUT_FILES = {
    "Merged": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/loss/merged.csv"
    ).as_posix(),
    "No Augment": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/loss/no_args.csv"
    ).as_posix(),
    "Random Assign": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/loss/random_assign.csv"
    ).as_posix(),
    "Synthetic": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/loss/synthetic.csv"
    ).as_posix(),
}
PLOT_FIGSIZE = (12, 8)
OUTPUT_FILE = PLOTS_PATH / "training_loss_ar_conditioner.png"

sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})


def plot_overlapped_lines(input_files: List[Path], output_file: Path):
    plt.figure(figsize=PLOT_FIGSIZE)

    colors = sns.color_palette("muted", n_colors=len(input_files))

    for (title, file_path), color in zip(input_files.items(), colors):
        df = pd.read_csv(file_path)
        plt.plot(df["Step"], df["Value"], label=title, color=color, linewidth=2)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(output_file.as_posix(), dpi=300, bbox_inches="tight")
    plt.close


def main():
    plot_overlapped_lines(input_files=INPUT_FILES, output_file=OUTPUT_FILE)


if __name__ == "__main__":
    main()
