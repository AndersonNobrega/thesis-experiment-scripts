import pickle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns
import numpy as np
import pandas as pd
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from paths import PLOTS_PATH, RESULT_PATH, CONF_PATH

style_path = CONF_PATH / "paper.mplstyle"
sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})
plt.style.use(style_path)

pt = 1.1 / 72.27
golden = (1 + 5**0.5) / 2
fig_width = 441.0 * pt
PLOT_FIGSIZE = (fig_width, fig_width / golden)


def plot_consumption(house: str, day: int, eval_mode: str) -> None:
    # Load ground truth
    with open(
        RESULT_PATH.joinpath(
            f"./{eval_mode}/experiment_merged_run3/dat/casa_{house}.dat"
        ).as_posix(),
        "rb",
    ) as f:
        arr_gt = pickle.load(f)

    # Load predictions
    with open(
        RESULT_PATH.joinpath(
            f"./{eval_mode}/experiment_no_args_run3/dat/casa_{house}_predicted.dat"
        ).as_posix(),
        "rb",
    ) as f:
        arr1 = pickle.load(f)
    with open(
        RESULT_PATH.joinpath(
            f"./{eval_mode}/experiment_random_assign_run3/dat/casa_{house}_predicted.dat"
        ).as_posix(),
        "rb",
    ) as f:
        arr2 = pickle.load(f)
    with open(
        RESULT_PATH.joinpath(
            f"./{eval_mode}/experiment_synthetic_modelling_run3/dat/casa_{house}_predicted.dat"
        ).as_posix(),
        "rb",
    ) as f:
        arr3 = pickle.load(f)
    with open(
        RESULT_PATH.joinpath(
            f"./{eval_mode}/experiment_merged_run3/dat/casa_{house}_predicted.dat"
        ).as_posix(),
        "rb",
    ) as f:
        arr4 = pickle.load(f)

    labels = ["A", "B", "C", "D"]
    arrays = [arr1, arr2, arr3, arr4]

    indexes = [2, 3, 4, 5]
    index_labels = {2: "Air Conditioner", 3: "Electric Shower", 4: "Refrigerator", 5: "Other"}

    ground_truth_color = "#333333"

    fig, axes = plt.subplots(
        2, 2, figsize=PLOT_FIGSIZE, sharey=True, sharex=True, constrained_layout=True
    )
    axes = axes.flatten()

    for ax, idx in zip(axes, indexes):
        df = []

        for arr, label in zip(arrays, labels):
            values = arr[1440 // 2 * day : 1440 // 2 * (day + 1), idx, 0]

            for i, val in enumerate(values):
                df.append(
                    {
                        "Sample": i,
                        "Consumption (W)": val,
                        "Augmentation Technique": label,
                    }
                )

        if idx == 5:
            values_gt = np.maximum(
                arr_gt[1440 // 2 * day : 1440 // 2 * (day + 1), 1, 0]
                - (
                    arr_gt[1440 // 2 * day : 1440 // 2 * (day + 1), 2, 0]
                    + arr_gt[1440 // 2 * day : 1440 // 2 * (day + 1), 3, 0]
                    + arr_gt[1440 // 2 * day : 1440 // 2 * (day + 1), 4, 0]
                ),
                0,
            )
        else:
            values_gt = arr_gt[1440 // 2 * day : 1440 // 2 * (day + 1), idx, 0]

        df = pd.DataFrame(df)

        sns.lineplot(
            data=df,
            x="Sample",
            y="Consumption (W)",
            hue="Augmentation Technique",
            ax=ax,
            legend=False,
        )

        sns.lineplot(
            x=np.arange(len(values_gt)),
            y=values_gt,
            ax=ax,
            color=ground_truth_color,
            label="Ground Truth",
            legend=False,
            zorder=999,
        )

        ax.set_title(f"{index_labels[idx]}", fontsize=10)
        ax.tick_params(axis="x", rotation=20)

        ax.set_xlabel("")
        ax.set_ylabel("")

        if ax.legend_:
            ax.legend_.remove()

        palette = sns.color_palette()

        legend_elements = [
            Line2D([0], [0], color=palette[i], lw=1.1, label=label)
            for i, label in enumerate(labels)
        ] + [Line2D([0], [0], color=ground_truth_color, lw=1.1, label="Ground Truth")]

        ax.legend(
            handles=legend_elements,
            # title="Experiment",
            # bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            fontsize=6,
        )

    bbox = axes[0].get_position()
    for ax in axes[1:]:
        bbox = Bbox.union([bbox, ax.get_position()])

    fig.text(
        bbox.x0 + bbox.width / 2, -0.02, "Minute", ha="center", va="center", fontsize=10
    )
    fig.text(
        -0.02,
        bbox.y0 + bbox.height / 2,
        "Consumption (W)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=10,
    )
    fig.savefig(
        PLOTS_PATH.joinpath(f"./predictions_{house}_{eval_mode}.png").as_posix(),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Aggregate and transform house data for individual appliance training.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--simple_eval", action="store_true", help="Use data from five eval houses."
    )
    parser.add_argument(
        "--house", type=str, default="andrey", help="House name (default: 'andrey')"
    )
    parser.add_argument("--day", type=int, default=1, help="Day number (default: 1)")

    return parser.parse_args()


def main() -> None:
    args = get_args()
    eval_mode = "simple_eval" if args.simple_eval else "hard_eval"

    plot_consumption(args.house, args.day, eval_mode)


if __name__ == "__main__":
    main()
