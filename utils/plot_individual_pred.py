import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from paths import PLOTS_PATH, RESULT_PATH, CONF_PATH

style_path = CONF_PATH / "paper.mplstyle"
sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})
plt.style.use(style_path)

pt = 1.1 / 72.27
golden = (1 + 5**0.5) / 1.9
fig_width = 441.0 * pt
PLOT_FIGSIZE = (fig_width, fig_width / golden)


def plot_consumption(house: str, day: int, eval_mode: str) -> None:
    # Load ground truth
    with open(
        RESULT_PATH / f"{eval_mode}/experiment_merged_run3/dat/casa_{house}.dat",
        "rb",
    ) as f:
        arr_gt = pickle.load(f)

    # Load predictions
    def load_pred(exp: str):
        path = RESULT_PATH / f"{eval_mode}/{exp}/dat/casa_{house}_predicted.dat"
        with open(path, "rb") as f:
            return pickle.load(f)

    arrays = [
        load_pred("experiment_no_args_run3"),
        load_pred("experiment_random_assign_run3"),
        load_pred("experiment_synthetic_modelling_run3"),
        load_pred("experiment_merged_run3")
    ]
    labels = ["A", "B", "C", "D"]

    indexes = [2, 3, 4, 5]
    index_labels = {2: "Air Conditioner", 3: "Shower", 4: "Fridge", 5: "Other"}
    ground_truth_color = "#333333"

    for idx in indexes:
        df = []
        for arr, label in zip(arrays, labels):
            values = arr[1440 * day : 1440 * (day + 1), idx, 0]
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
                arr_gt[1440 * day : 1440 * (day + 1), 1, 0]
                - (
                    arr_gt[1440 * day : 1440 * (day + 1), 2, 0]
                    + arr_gt[1440 * day : 1440 * (day + 1), 3, 0]
                    + arr_gt[1440 * day : 1440 * (day + 1), 4, 0]
                ),
                0,
            )
        else:
            values_gt = arr_gt[1440 * day : 1440 * (day + 1), idx, 0]

        df = pd.DataFrame(df)

        plt.figure(figsize=PLOT_FIGSIZE)
        ax = sns.lineplot(
            data=df,
            x="Sample",
            y="Consumption (W)",
            hue="Augmentation Technique",
        )

        sns.lineplot(
            x=np.arange(len(values_gt)),
            y=values_gt,
            color=ground_truth_color,
            label="Ground Truth",
        )

        ax.set_xlabel("Minute")
        ax.set_ylabel("Consumption (W)")
        ax.legend(loc="upper left")
        # ax.legend(
        #     bbox_to_anchor=(1.02, 1),
        #     loc="upper left",
        #     borderaxespad=0,
        #     frameon=True,
        # )

        plot_name = f"predictions_{house}_{eval_mode}_{index_labels[idx].replace(' ', '_').lower()}.png"
        plt.savefig(PLOTS_PATH / plot_name, dpi=300, bbox_inches="tight")
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
