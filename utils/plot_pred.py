import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys

def plot_consumption(house, day):
    sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})

    # Load ground truth
    with open(f'/home/anderson/temp/casa_{house}.dat', 'rb') as f:
        arr_gt = pickle.load(f)

    # Load predictions
    with open(f'/home/anderson/parameters_results/experiment_merged_run3/dat/casa_{house}_predicted.dat', 'rb') as f:
        arr1 = pickle.load(f)
    with open(f'/home/anderson/parameters_results/experiment_no_args_run3/dat/casa_{house}_predicted.dat', 'rb') as f:
        arr2 = pickle.load(f)
    with open(f'/home/anderson/parameters_results/experiment_random_assign_run3/dat/casa_{house}_predicted.dat', 'rb') as f:
        arr3 = pickle.load(f)
    with open(f'/home/anderson/parameters_results/experiment_synthetic_modelling_run3/dat/casa_{house}_predicted.dat', 'rb') as f:
        arr4 = pickle.load(f)

    labels = ["Merged", "No Args", "Random Assign", "Synthetic Modelling"]
    arrays = [arr1, arr2, arr3, arr4]

    indexes = [2, 3, 4, 5]
    index_labels = {
        2: "Air Conditioner",
        3: "Shower",
        4: "Fridge",
        5: "Other"
    }

    ground_truth_color = "#181a1c"

    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8), sharex=True)
    axes = axes.flatten()

    for ax, idx in zip(axes, indexes):
        df = []

        for arr, label in zip(arrays, labels):
            values = arr[1440*day:1440*(day+1), idx, 0]

            for i, val in enumerate(values):
                df.append({
                    "Sample": i,
                    "Consumption (W)": val,
                    "Augmentation Technique": label
                })

        if idx == 5:
            values_gt = np.maximum(
                arr_gt[1440*day:1440*(day+1), 1, 0] - 
                (arr_gt[1440*day:1440*(day+1), 2, 0] + arr_gt[1440*day:1440*(day+1), 3, 0] + arr_gt[1440*day:1440*(day+1), 4, 0]),
                0
            )
        else:
            values_gt = arr_gt[1440*day:1440*(day+1), idx, 0]

        df = pd.DataFrame(df)

        sns.lineplot(data=df, x="Sample", y="Consumption (W)", hue="Augmentation Technique", ax=ax, linewidth=1.1)

        sns.lineplot(
            x=np.arange(len(values_gt)), 
            y=values_gt, 
            ax=ax, 
            color=ground_truth_color, 
            label="Ground Truth", 
            linewidth=1.1, 
            legend=True
        )

        ax.set_title(f"{index_labels[idx]}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Consumption (W)")
        ax.tick_params(axis='x', rotation=20)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Augmentation Technique", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.suptitle("Predictions vs Ground Truth per Appliance", fontsize=16, y=1.02)
    plt.show()

parser = argparse.ArgumentParser(description="Plot consumption predictions vs ground truth.")
parser.add_argument('--house', type=str, default="andrey", help="House name (default: 'andrey')")
parser.add_argument('--day', type=int, default=1, help="Day number (default: 1)")

args = parser.parse_args()

plot_consumption(args.house, args.day)
