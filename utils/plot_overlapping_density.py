import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from paths import PLOTS_PATH, RESULT_PATH

# Constants
INPUT_FILES = {
    "Merged": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/merged.json"
    ).as_posix(),
    "No Augment": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/no_args.json"
    ).as_posix(),
    "Random Assign": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/random_assign.json"
    ).as_posix(),
    "Synthetic": RESULT_PATH.joinpath(
        "tensorboard/ar_conditioner_train/histgrams/kernel/synthetic.json"
    ).as_posix(),
}
PLOT_FIGSIZE = (12, 8)
X_LIMIT_BUFFER = 0.05
DENSITY_POINTS = 500

sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})


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
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    colors = sns.color_palette("muted", n_colors=len(input_files))

    for (title, samples), color in zip(sample_sets.items(), colors):
        if samples.size > 1:
            density = compute_kde(samples, x_values, normalize=normalize)
            ax.plot(x_values, density, label=title, linewidth=2, color=color)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(xlim)

    # Set y-axis
    if normalize:
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Normalized Density", fontsize=12)
    else:
        ax.set_ylabel("Density", fontsize=12)
        if logscale:
            ax.set_yscale("log")

    ax.set_xlabel("Kernel Weight Value", fontsize=12)
    title = "Overlapping Density Plots"
    if normalize:
        title = "Overlapping Normalized Density Plots"
    ax.set_title(title, fontsize=14)

    ax.grid(axis="x", zorder=0)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(output_file.as_posix(), dpi=300, bbox_inches="tight")
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
