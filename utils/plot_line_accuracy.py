import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
INPUT_FILES = {
    "Merged": "results/tensorboard/ar_conditioner_train/acc/merged.csv",
    "No Args": "results/tensorboard/ar_conditioner_train/acc/no_args.csv",
    "Random Assign": "results/tensorboard/ar_conditioner_train/acc/random_assign.csv",
    "Synthetic": "results/tensorboard/ar_conditioner_train/acc/synthetic.csv",
}
PLOT_FIGSIZE = (12, 8)

sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})

def plot_overlapped_lines(input_files):
    plt.figure(figsize=PLOT_FIGSIZE)

    colors = sns.color_palette("muted", n_colors=len(input_files))

    for (title, file_path), color in zip(input_files.items(), colors):
        df = pd.read_csv(file_path)
        plt.plot(df["Step"], df["Value"], label=title, color=color, linewidth=2)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1)
    plt.title("Overlapped Line Plot from CSV Files")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    plot_overlapped_lines(INPUT_FILES)

if __name__ == "__main__":
    main()
