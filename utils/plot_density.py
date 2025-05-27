import json
from matplotlib.transforms import Bbox
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from paths import PLOTS_PATH, RESULT_PATH, CONF_PATH

style_path = CONF_PATH / "paper.mplstyle"
sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})
plt.style.use(style_path)

# Constants
INPUT_FILES = {
    "Experiment A": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/no_args.json"
    ).as_posix(),
    "Experiment B": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/random_assign.json"
    ).as_posix(),
    "Experiment C": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/synthetic.json"
    ).as_posix(),
    "Experiment D": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/merged.json"
    ).as_posix(),
}

pt = 1.0 / 72.27
golden = (1 + 5**0.5) / 2
fig_width = 441.0 * pt
fig_height = fig_width / golden
PLOT_FIGSIZE = (fig_width, fig_height)
MAX_LIMIT_BUFFER = 0.05  # 5% buffer
DENSITY_POINTS = 500  # Number of points in smoothed curve


def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[0][2]


def extract_samples_from_bins(bin_data):
    """Expand histogram bin data into samples."""
    samples = []
    for start, end, count in bin_data:
        # Approximate by filling count samples at the bin center
        center = (start + end) / 2
        samples.extend([center] * int(count))
    return samples


def plot_density(input_files, output_file):
    all_samples = []
    processed_data = {}

    # Load and process all data
    for title, file_path in input_files.items():
        bin_data = load_data(file_path)
        samples = extract_samples_from_bins(bin_data)
        processed_data[title] = samples
        all_samples.extend(samples)

    # Determine symmetric x-limits centered at 0
    max_abs_x = max(abs(min(all_samples)), abs(max(all_samples)))
    max_abs_x *= 1 + MAX_LIMIT_BUFFER  # add buffer
    xlim = (-max_abs_x, max_abs_x)

    # Create common x values for density estimation
    x_values = np.linspace(*xlim, DENSITY_POINTS)

    # Plotting
    fig, axes = plt.subplots(
        2, 2, figsize=PLOT_FIGSIZE, sharex=True, sharey=True, constrained_layout=True
    )
    axes = axes.flatten()

    max_density = 0
    densities = {}

    # Calculate densities first
    for title, samples in processed_data.items():
        if len(samples) > 1:
            kde = gaussian_kde(samples)
            density = kde(x_values)
        else:
            density = np.zeros_like(x_values)

        densities[title] = density
        max_density = max(max_density, density.max())

    ylim = (0, max_density * (1 + MAX_LIMIT_BUFFER))

    # Now plot
    for i, (ax, (title, density)) in enumerate(zip(axes, densities.items())):
        ax.fill_between(x_values, density, color="lightblue", alpha=0.6)
        ax.plot(x_values, density, linestyle="-")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    bbox = axes[0].get_position()
    for ax in axes[1:]:
        bbox = Bbox.union([bbox, ax.get_position()])

    fig.text(
        bbox.x0 + bbox.width / 2, -0.02, "Kernel Weights", ha="center", va="center"
    )
    fig.text(
        -0.02,
        bbox.y0 + bbox.height / 2,
        "Density",
        ha="center",
        va="center",
        rotation="vertical",
    )
    fig.savefig(output_file.as_posix(), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    plot_density(input_files=INPUT_FILES, output_file=PLOTS_PATH / "density_kernel.png")


if __name__ == "__main__":
    main()
