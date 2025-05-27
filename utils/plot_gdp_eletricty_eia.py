import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from paths import PLOTS_PATH, CONF_PATH

style_path = CONF_PATH / "paper.mplstyle"
sns.set_theme(style="whitegrid", palette="muted", rc={"axes.edgecolor": "black"})
plt.style.use(style_path)
plt.rcParams.update({"xtick.bottom": False})

OUTPUT_FILE = PLOTS_PATH / "gdp_eletricity_eia_2022.png"

pt = 1.0 / 72.27
golden = (1 + 5**0.5) / 2
fig_width = 441.0 * pt
PLOT_FIGSIZE = (fig_width, fig_width / golden)

data_historic = {
    "GDP": [
        0,
        6.2,
        9.6,
        12.2,
        15.8,
        19.7,
        23.7,
        27.7,
        31.7,
        35.6,
        39.5,
        43.4,
        47.3,
        51.4,
        55.5,
        59.5,
        63.5,
        67.5,
        71.6,
        75.7,
        79.9,
        84.1,
        88.4,
        92.7,
        97.0,
        101.3,
        105.5,
        109.8,
        114.1,
        118.3,
        122.6,
    ],
    "EG": [
        0,
        5.4,
        8.8,
        9.4,
        11.9,
        13.9,
        15.6,
        17.3,
        18.9,
        20.6,
        22.2,
        24.1,
        26.0,
        27.9,
        29.8,
        31.7,
        33.7,
        35.6,
        37.6,
        39.6,
        41.5,
        43.7,
        45.9,
        48.1,
        50.3,
        52.5,
        54.5,
        56.6,
        58.7,
        60.8,
        62.9,
    ],
    "Year": [
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
        "2026",
        "2027",
        "2028",
        "2029",
        "2030",
        "2031",
        "2032",
        "2033",
        "2034",
        "2035",
        "2036",
        "2037",
        "2038",
        "2039",
        "2040",
        "2041",
        "2042",
        "2043",
        "2044",
        "2045",
        "2046",
        "2047",
        "2048",
        "2049",
        "2050",
    ],
}

df = pd.DataFrame.from_dict(data_historic)

plt.figure(figsize=PLOT_FIGSIZE, constrained_layout=True)

line_plot = sns.lineplot(
    data=df[df["Year"] < "2023"],
    x="Year",
    y="GDP",
    color="#3b719f",
    label="GDP (Purchasing Power Parity)",
    lw=1.1,
)
sns.lineplot(
    data=df[df["Year"] >= "2022"],
    x="Year",
    y="GDP",
    linestyle="dashed",
    color="#3b719f",
    lw=1.1,
)
sns.lineplot(
    data=df[df["Year"] < "2023"],
    x="Year",
    y="EG",
    label="Electricity Generation",
    color="#cb4c4e",
    lw=1.1,
)
sns.lineplot(
    data=df[df["Year"] >= "2022"],
    x="Year",
    y="EG",
    linestyle="dashed",
    color="#cb4c4e",
    lw=1.1,
)

line_plot.set_ylabel("Growth")

for ind, label in enumerate(line_plot.get_xticklabels()):
    if ind % 5 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)

line_plot.axvline(x="2022", ymin=0, ymax=1, color="black")
plt.legend(loc="upper center", frameon=True)
plt.savefig(OUTPUT_FILE.as_posix(), dpi=300, bbox_inches="tight")
plt.close()
