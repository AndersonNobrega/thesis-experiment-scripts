from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from paths import RESULT_PATH, PLOTS_PATH

sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})


def get_args() -> Namespace:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--simple_eval", action="store_true", help="Use data from five eval houses."
    )

    return parser.parse_args()


def plot_bar_chart(data: pd.DataFrame, output_file) -> None:
    plt.figure(figsize=(12, 8))
    plt.grid(axis="x")
    ax = sns.barplot(data=data, x="Experimento", y="Acuracia", hue="Aparelho")
    for container in ax.containers:
        ax.bar_label(container)

    plt.ylabel("Acurácia (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
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
