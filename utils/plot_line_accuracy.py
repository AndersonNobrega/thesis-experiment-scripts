from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from paths import PLOTS_PATH, RESULT_PATH, CONF_PATH

style_path = CONF_PATH / "paper.mplstyle"
sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})
plt.style.use(style_path)

# Constants
INPUT_FILES = {
    "A": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/acc/no_args.csv"
    ).as_posix(),
    "B": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/acc/random_assign.csv"
    ).as_posix(),
    "C": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/acc/synthetic.csv"
    ).as_posix(),
    "D": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/acc/merged.csv"
    ).as_posix(),
}
pt = 1.0 / 72.27
golden = (1 + 5**0.5) / 2
fig_width = 441.0 * pt
PLOT_FIGSIZE = (fig_width, fig_width / golden)
OUTPUT_FILE = PLOTS_PATH / "training_accuracy_ar_conditioner.png"


def plot_overlapped_lines(input_files: List[Path], output_file: Path):
    plt.figure(figsize=PLOT_FIGSIZE, constrained_layout=True)

    colors = sns.color_palette("muted", n_colors=len(input_files))

    for (title, file_path), color in zip(input_files.items(), colors):
        df = pd.read_csv(file_path)
        plt.plot(df["Step"], df["Value"], label=title, color=color, linewidth=1.1)

    plt.xlabel("Epochs")
    plt.ylabel("Estimated Accuracy")
    plt.ylim(0.5, 1)
    plt.legend(title="Experiments", loc="lower right")
    plt.savefig(output_file.as_posix(), dpi=300, bbox_inches="tight")
    plt.close


def main():
    plot_overlapped_lines(input_files=INPUT_FILES, output_file=OUTPUT_FILE)


if __name__ == "__main__":
    main()
