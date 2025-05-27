import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

from paths import RESULT_PATH, PLOTS_PATH, CONF_PATH

style_path = CONF_PATH / "paper.mplstyle"
sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})
plt.style.use(style_path)

file = open(
    RESULT_PATH.joinpath(
        "./hard_eval/experiment_merged_run3/dat/casa_andrey_predicted.dat"
    ).as_posix(),
    "rb",
)

pt = 1.0 / 72.27
golden = (1 + 5**0.5) / 2
fig_width = 441.0 * pt
PLOT_FIGSIZE = (fig_width, fig_width / golden)

OUTPUT_FILE = PLOTS_PATH / "noisy_example.png"

arr = pickle.load(file)
arr_size = arr[9500:10000, 4, 0].shape[0]

max_value = arr[9500:10000, 4, 0].max() + (arr[9500:10000, 4, 0].max()) * 0.1
gaussian_noise = np.random.normal(0, 1, arr_size) * 5
filtered = lowess(
    arr[9500:10000, 4, 0], np.arange(arr_size), is_sorted=True, frac=0.025, it=0
)
filtered_values = filtered[:, 1]
filtered_values[filtered_values < 30] = 0

noisy_signal = arr[9500:10000, 4, 0] + gaussian_noise
noisy_signal[noisy_signal > 200] = 150

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_FIGSIZE)

fig.supxlabel("Minutes")
fig.supylabel("Power Consumption (W)")

ax1.set_ylim(-5, max_value)
ax1.plot(filtered_values, color="#181a1c")

ax2.set_ylim(-5, max_value)
ax2.plot(noisy_signal, color="#181a1c")

plt.subplots_adjust(left=0.09, wspace=0.4)

arrowprops = dict(arrowstyle="->", linewidth=2, color="red")

plt.text(
    -140,
    int(max_value / 2) - 15,
    "Noise Insertion",
    ha="center",
    bbox={
        "facecolor": "oldlace",
        "alpha": 0.5,
        "boxstyle": "rarrow,pad=0.3",
        "ec": "red",
    },
)

plt.savefig(OUTPUT_FILE.as_posix(), dpi=300, bbox_inches="tight")
plt.close()
