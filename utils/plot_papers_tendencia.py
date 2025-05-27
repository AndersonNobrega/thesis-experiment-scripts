import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from paths import PLOTS_PATH, CONF_PATH

style_path = CONF_PATH / "paper.mplstyle"
sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})
plt.style.use(style_path)

# Constants
YEARS = tuple(str(year) for year in range(2012, 2025))
WIDTH = 0.7
OUTPUT_FILE = PLOTS_PATH / "tendencia_papers.png"

pt = 1.0 / 72.27
golden = (1 + 5**0.5) / 2
fig_width = 441.0 * pt
PLOT_FIGSIZE = (fig_width, fig_width / golden)

# Weight counts per category
WEIGHT_COUNTS = {
    "Random Appliance Activation Assignement": np.array(
        [1, 0, 0, 1, 2, 1, 3, 4, 2, 1, 0, 2, 2]
    ),
    "Statistical Modeling/Simulation": np.array(
        [0, 0, 1, 0, 1, 2, 1, 1, 1, 0, 0, 3, 7]
    ),
    "Synthetic Datasets": np.array([0, 0, 0, 0, 1, 1, 1, 1, 4, 0, 0, 2, 1]),
    "Class Imbalance": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1]),
    "Signal Transformation": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1]),
    "Generative Adversarial Networks": np.array(
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 4, 5, 4]
    ),
    "Reviews": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 2]),
}


def calculate_total_weight(weight_counts):
    total = sum(sum(weight_counts[key]) for key in weight_counts)
    for key in weight_counts.keys():
        print(f"{key}: {sum(weight_counts[key])}")
    print(f"Total: {total}")


def create_stacked_bar_plot(years, weight_counts, width, output_path):
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
    bottom = np.zeros(len(years))

    for label, weight_count in weight_counts.items():
        ax.bar(
            years,
            weight_count,
            width,
            label=label,
            edgecolor="black",
            bottom=bottom,
            zorder=3,
            linewidth=0.5,
        )
        bottom += weight_count

    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.set_xlabel("Year")
    ax.set_ylabel("Articles")
    ax.legend(loc="upper left")

    fig.savefig(output_path.as_posix(), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    calculate_total_weight(WEIGHT_COUNTS)
    create_stacked_bar_plot(YEARS, WEIGHT_COUNTS, WIDTH, OUTPUT_FILE)


if __name__ == "__main__":
    main()
