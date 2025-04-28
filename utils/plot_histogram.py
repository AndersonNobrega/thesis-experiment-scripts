import json
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
INPUT_FILES = {
    "Merged": "results/tensorboard/ar_conditioner_train/histgrams/kernel/merged.json",
    "No Args": "results/tensorboard/ar_conditioner_train/histgrams/kernel/no_args.json",
    "Random Assign": "results/tensorboard/ar_conditioner_train/histgrams/kernel/random_assign.json",
    "Synthetic": "results/tensorboard/ar_conditioner_train/histgrams/kernel/synthetic.json",
}
PLOT_FIGSIZE = (16, 12)
MAX_Y_LIMIT_BUFFER = 0.05

sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})

def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[0][2]

def process_bin_data(bin_data):
    bin_data.sort(key=lambda x: x[0])
    bin_centers = [(start + end) / 2 for start, end, _ in bin_data]
    values = [val for _, _, val in bin_data]
    return bin_centers, values

def plot_histograms(input_files):
    all_values = []
    processed_data = {}

    for title, file_path in input_files.items():
        bin_data = load_data(file_path)
        bin_centers, values = process_bin_data(bin_data)
        processed_data[title] = (bin_centers, values)
        all_values.extend(values)

    max_y_value = max(all_values)
    y_limit = max_y_value + MAX_Y_LIMIT_BUFFER * max_y_value

    fig, axes = plt.subplots(2, 2, figsize=PLOT_FIGSIZE)
    axes = axes.flatten()

    for ax, (title, (bin_centers, values)) in zip(axes, processed_data.items()):
        ax.fill_between(bin_centers, values, color="lightblue", alpha=0.6)
        ax.plot(bin_centers, values, linestyle="-", marker="")
        ax.set_title(title)
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, y_limit)
        ax.grid(True, axis='x', zorder=0)

    plt.tight_layout()
    plt.show()

def main():
    plot_histograms(INPUT_FILES)

if __name__ == "__main__":
    main()
