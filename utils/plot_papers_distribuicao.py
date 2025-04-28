import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from paths import PLOTS_PATH

# Constants
YEARS = tuple(str(year) for year in range(2012, 2025))
WEIGHT_COUNTS = [1, 0, 1, 1, 4, 5, 5, 8, 8, 10, 5, 12, 18]
BAR_WIDTH = 0.7
OUTPUT_FILE = PLOTS_PATH / "distribuicao_papers.png"

sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})


def compute_trendline(x_values, y_values, degree=2):
    coefficients = np.polyfit(x_values, y_values, degree)
    return np.polyval(coefficients, x_values)


def create_distribution_plot(years, counts, trendline, width, output_path):
    _, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, zorder=0)

    ax.bar(years, counts, width, edgecolor="black", zorder=3)
    ax.plot(
        years,
        trendline,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Trendline",
        zorder=4,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Articles")
    ax.grid(axis="x")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_path.as_posix(), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    x_numeric = np.arange(len(YEARS))
    trend = compute_trendline(x_numeric, WEIGHT_COUNTS)
    create_distribution_plot(YEARS, WEIGHT_COUNTS, trend, BAR_WIDTH, OUTPUT_FILE)


if __name__ == "__main__":
    main()
