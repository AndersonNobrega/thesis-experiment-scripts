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
PLOT_FIGSIZE = (12, 8)
MAX_Y_LIMIT_BUFFER = 0.05  # 5% buffer for Y axis

sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})

# Load data from JSON file
def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[0][2]

# Prepare bin centers and normalized values
def process_bin_data(bin_data):
    bin_data.sort(key=lambda x: x[0])
    bin_centers = [(start + end) / 2 for start, end, _ in bin_data]
    values = [val for _, _, val in bin_data]

    max_val = max(values)
    normalized_values = [v / max_val for v in values]
    
    return bin_centers, normalized_values

def plot_overlapped_histograms(input_files):
    processed_data = {}

    for title, file_path in input_files.items():
        bin_data = load_data(file_path)
        bin_centers, values = process_bin_data(bin_data)
        processed_data[title] = (bin_centers, values)

    plt.figure(figsize=PLOT_FIGSIZE)

    # Different line styles to improve comparison
    line_styles = ["-", "--", "-.", ":"]
    colors = sns.color_palette("muted", n_colors=len(input_files))  # Get matching colors

    for (title, (bin_centers, values)), linestyle, color in zip(processed_data.items(), line_styles, colors):
        plt.plot(bin_centers, values, label=title, linestyle=linestyle, linewidth=2, color=color)
        plt.fill_between(bin_centers, values, alpha=0.2, color=color)  # Add transparent fill

    plt.axhline(0, color="black", linewidth=0.8)

    plt.xlabel("Bin Center")
    plt.ylabel("Normalized Frequency")
    plt.ylim(0, 1 + MAX_Y_LIMIT_BUFFER)
    plt.title("Overlapped Normalized Histogram Curves")
    plt.legend()
    plt.grid(True, zorder=0)
    plt.tight_layout()
    plt.show()

def main():
    plot_overlapped_histograms(INPUT_FILES)

if __name__ == "__main__":
    main()
