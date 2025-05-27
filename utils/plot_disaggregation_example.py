import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from paths import RESULT_PATH, PLOTS_PATH, CONF_PATH

style_path = CONF_PATH / "paper.mplstyle"
sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})
plt.style.use(style_path)

OUTPUT_FILE = PLOTS_PATH / "disaggregation_example.png"

pt = 1.2 / 72.27
golden = (1 + 5**0.5) / 2
fig_width = 441.0 * pt
PLOT_FIGSIZE = (fig_width, fig_width / golden)

file = open(
    RESULT_PATH.joinpath(
        "./hard_eval/experiment_merged_run3/dat/casa_andrey_predicted.dat"
    ).as_posix(),
    "rb",
)

arr = pickle.load(file)

max_value = arr[9500:10000, 1, 0].max() + (arr[9500:10000, 1, 0].max()) * 0.1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PLOT_FIGSIZE)

ax1.set_ylim(0, max_value)
ax1.plot(arr[9500:10000, 1, 0], label="Total", color="#181a1c")

ax2.set_ylim(0, max_value)
ax2.plot(arr[9500:10000, 5, 0], label="Others")
ax2.plot(arr[9500:10000, 2, 0], label="Air Conditioner")
ax2.plot(arr[9500:10000, 3, 0], label="Electric Shower")
ax2.plot(arr[9500:10000, 4, 0], label="Refrigerator")


plt.subplots_adjust(left=0.08, bottom=0.07, hspace=0.75)


arrowprops = dict(arrowstyle="->", linewidth=2, color="red")

plt.text(
    250,
    int(max_value) + 280,
    "Disaggregation",
    ha="center",
    rotation="vertical",
    size=7,
    bbox={
        "facecolor": "oldlace",
        "alpha": 0.5,
        "boxstyle": "larrow,pad=0.3",
        "ec": "red",
    },
)

plt.tight_layout()
plt.savefig(OUTPUT_FILE.as_posix(), dpi=300, bbox_inches="tight")
plt.close()
