import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pairo_butler.utils.tools import UGENT, listdir, pyout


ROOT1 = Path("/media/matt/Expansion/observation_results")
ROOT2 = Path("/media/matt/Expansion/Results")
OUT = Path("/media/matt/Expansion/Analysis")
FIGSIZE = (6, 4)

# # compute relative coverage
# for towel_nr in range(10):
#     with open(
#         ROOT / f"sample_towel_{towel_nr}_ground_truth/observation_result/result.json",
#         "r",
#     ) as f:
#         ground_truth_coverage = json.load(f)["coverage"]
#     for sample in listdir(ROOT):
#         if sample.name.startswith(f"sample_towel_{towel_nr}"):
#             with open(sample / "observation_result/result.json", "r") as f:
#                 sample_result = json.load(f)
#             sample_result["ratio"] = sample_result["coverage"] / ground_truth_coverage
#             with open(sample / "observation_result/result.json", "w") as f:
#                 json.dump(sample_result, f)


# Make boxplot
def load_data():
    try:
        with open(OUT / "boxdata.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"Category": [], "Coverage": [], "Time": [], "Tries": []}
        for sample in listdir(ROOT1):
            if "ground_truth" in sample.name:
                continue
            with open(sample / "observation_result/result.json", "r") as f:
                coverage = json.load(f)["ratio"]
            towel_nr = sample.name.split("_")[2]

            with open(
                ROOT2 / sample.name[7:14] / sample.name[-26:] / "stats.json",
                "r",
            ) as f:
                stats = json.load(f)
            data["Time"].append(stats["time"])
            data["Tries"].append(stats["tries"])

            data["Category"].append(f"{int(towel_nr)}")
            data["Coverage"].append(coverage)
        with open(OUT / "boxdata.json", "w+") as f:
            json.dump(data, f)
    return data


def boxplot(data, measure="Coverage", lim=None):
    df = pd.DataFrame(
        {
            "Category": [
                "All",
            ]
            * len(data["Category"])
            + data["Category"],
            measure: data[measure] + data[measure],
        }
    )

    plt.figure(figsize=FIGSIZE)
    custom_palette = [UGENT.YELLOW] + [UGENT.SKY] * 10
    sns.boxplot(x="Category", y=measure, data=df, palette=custom_palette)
    if lim is not None:
        plt.ylim(*lim)
    plt.title(f"Boxplot of {measure}")
    plt.xlabel("Towel")
    plt.ylabel(measure)
    plt.savefig(OUT / f"box_{measure}.png")


def histogram(data, measure="Coverage", lim=None, bins=None, discrete=False):
    df = pd.DataFrame(data)

    plt.figure(figsize=FIGSIZE)
    if bins is None:
        ax = sns.histplot(
            df[measure], kde=True, color=UGENT.SKY, alpha=1.0, discrete=discrete
        )
    else:
        ax = sns.histplot(
            df[measure],
            bins=bins,
            kde=True,
            color=UGENT.SKY,
            alpha=1.0,
            discrete=discrete,
        )
    ax.lines[0].set_color(UGENT.BLUE)

    if lim is not None:
        plt.xlim(*lim)
    plt.title(f"Histogram of {measure}")
    plt.xlabel(measure)
    plt.ylabel("Frequency")
    plt.savefig(OUT / f"hist_{measure}.png")


def statistics(data, success_threshold=0.8):
    values = np.array(data["Coverage"])
    mean = np.mean(values)
    median = np.median(values)
    stddev = np.std(values)
    success_rate_quantitative = np.sum(values > success_threshold) / values.size

    qualitative = np.array(pd.read_csv(OUT / "qualitative.csv")["Success"])
    success_rate_qualitative = np.sum(qualitative) / qualitative.size

    stats = (
        f"Mean:     {mean:.4f}\n"
        f"Median:   {median:.4f}\n"
        f"Stddev:   {stddev:.4f}\n"
        f"Cov >80%: {success_rate_quantitative:.4f}\n"
        f"Grasp OK: {success_rate_qualitative:.4f}"
    )
    pyout(stats)

    with open(OUT / "results.txt", "w+") as f:
        f.write(stats)


data = load_data()
boxplot(data, "Coverage", lim=[0.0, 1.1])
boxplot(data, "Time")
boxplot(data, "Tries")
histogram(data, "Coverage", lim=[0.0, 1.1], bins=16)
histogram(data, "Time")
histogram(data, "Tries", discrete=True)
statistics(data)
