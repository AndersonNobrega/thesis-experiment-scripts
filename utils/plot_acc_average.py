from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from paths import RESULT_PATH, PLOTS_PATH, CONF_PATH

style_path = CONF_PATH / "paper.mplstyle"
sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})
plt.style.use(style_path)

pt = 1.1 / 72.27
golden = (1 + 5**0.5) / 2
fig_width = 441.01772 * pt
PLOT_FIGSIZE = (fig_width, fig_width / golden)


def get_args() -> Namespace:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--simple_eval", action="store_true", help="Use data from five eval houses."
    )

    return parser.parse_args()


def plot_bar_chart(data: pd.DataFrame, output_file) -> None:
    plt.figure(figsize=PLOT_FIGSIZE, constrained_layout=True)
    ax = sns.barplot(data=data, x="Experimento", y="Acuracia", hue="Aparelho")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)

    plt.ylabel("Estimated Accuracy")
    plt.xlabel("Experiment")
    plt.ylim(0, 100)
    plt.legend(loc="upper left")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = get_args()
    mode = "simple_eval" if args.simple_eval else "hard_eval"
    input_file = RESULT_PATH / mode / "average_acc.csv"
    output_file = PLOTS_PATH / f"eval_average_acc_{mode}.png"

    data = pd.read_csv(input_file)
    data["Acuracia"] = data["Acuracia"].round(1)

    plot_bar_chart(data, output_file)


if __name__ == "__main__":
    main()
