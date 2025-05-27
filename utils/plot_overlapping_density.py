import json
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
    "A": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/no_args.json"
    ).as_posix(),
    "B": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/random_assign.json"
    ).as_posix(),
    "C": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/synthetic.json"
    ).as_posix(),
    "D": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/merged.json"
    ).as_posix(),
}

pt = 1.2 / 72.27
golden = (1 + 5**0.5) / 2
fig_width = 441.0 * pt
PLOT_FIGSIZE = (fig_width, fig_width / golden)
X_LIMIT_BUFFER = 0.05
DENSITY_POINTS = 500


def load_data(file_path):
    """Load histogram data from a JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[0][2]


def expand_bin_data(bin_data):
    """Expand bin histogram data into raw samples (approximate)."""
    samples = []
    for start, end, count in bin_data:
        center = (start + end) / 2
        samples.extend([center] * int(count))
    return samples


def compute_kde(samples, x_values, normalize=True):
    """Compute the Kernel Density Estimation (KDE) over x_values."""
    kde = gaussian_kde(samples)
    density = kde(x_values)
    if normalize:
        density /= density.max()
    return density


def plot_density(input_files, output_file, normalize=True, logscale=False):
    """Plot overlapping density plots for multiple datasets."""
    all_samples = []
    sample_sets = {}

    # Load and expand data
    for title, file_path in input_files.items():
        bin_data = load_data(file_path)
        samples = expand_bin_data(bin_data)
        sample_sets[title] = np.array(samples)
        all_samples.extend(samples)

    all_samples = np.array(all_samples)
    max_abs_x = np.max(np.abs(all_samples))
    xlim = (-max_abs_x * (1 + X_LIMIT_BUFFER), max_abs_x * (1 + X_LIMIT_BUFFER))
    x_values = np.linspace(*xlim, DENSITY_POINTS)

    # Create plot
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
    colors = sns.color_palette("muted", n_colors=len(input_files))

    for (title, samples), color in zip(sample_sets.items(), colors):
        if samples.size > 1:
            density = compute_kde(samples, x_values, normalize=normalize)
            ax.plot(x_values, density, label=title, linewidth=1.1, color=color)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(xlim)

    # Set y-axis
    if normalize:
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Normalized Density")
    else:
        ax.set_ylabel("Density")
        if logscale:
            ax.set_yscale("log")

    ax.set_xlabel("Kernel Weights")
    ax.legend(title="Experiments", loc="upper left")

    fig.savefig(output_file.as_posix(), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    plot_density(
        input_files=INPUT_FILES,
        output_file=PLOTS_PATH / "overlapping_density_kernel_non_normalized.png",
        normalize=False,
        logscale=False,
    )


if __name__ == "__main__":
    main()
