import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from paths import PLOTS_PATH

sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})

species = (
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
)

weight_counts = {
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

total = 0
for key in weight_counts.keys():
    print(key, sum(weight_counts[key]))
    total += sum(weight_counts[key])

width = 0.7

fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(True, zorder=0)
bottom = np.zeros(len(species))

for boolean, weight_count in weight_counts.items():
    p = ax.bar(
        species,
        weight_count,
        width,
        label=boolean,
        edgecolor="black",
        bottom=bottom,
        zorder=3,
    )
    bottom += weight_count

ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
ax.set_ylabel("Articles")
ax.set_xlabel("Year")
ax.grid(axis="x")
ax.legend(loc="upper left")

plt.savefig(
    PLOTS_PATH.joinpath("./tendencia_papers.png").as_posix(), bbox_inches="tight"
)
