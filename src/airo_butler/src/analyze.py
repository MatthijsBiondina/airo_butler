import json
from pathlib import Path
import sys

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from pairo_butler.utils.tools import UGENT, listdir, pyout
import cv2
import matplotlib.font_manager as fm

font_path = Path(
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
)  # Update with the correct path

# Check if the font file exists
if font_path.is_file():
    # Add the font
    fm.fontManager.addfont(str(font_path))

    # Set Times New Roman as the default font
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
else:
    print("Times New Roman font not found.")
    sys.exit(0)

ROOT1 = Path("/media/matt/Expansion/samples")
ROOT2 = Path("/media/matt/Expansion/Results/test_trials")
OUT = Path("/media/matt/Expansion/Analysis")

SIZE_MODIFIER = 2
COLUMN_WIDTH = 3.228 * SIZE_MODIFIER
FONTSIZE = 8 * SIZE_MODIFIER
FIGSIZE = (COLUMN_WIDTH, COLUMN_WIDTH / 3 * 2)

# # compute relative coverage
# for towel_nr in range(10):
#     with open(
#         ROOT1 / f"sample_towel_{towel_nr}_ground_truth/observation_result/result.json",
#         "r",
#     ) as f:
#         ground_truth_coverage = json.load(f)["coverage"]
#     for sample in listdir(ROOT1):
#         if sample.name.startswith(f"sample_towel_{towel_nr}"):
#             with open(sample / "observation_result/result.json", "r") as f:
#                 sample_result = json.load(f)
#             sample_result["ratio"] = sample_result["coverage"] / ground_truth_coverage
#             with open(sample / "observation_result/result.json", "w") as f:
#                 json.dump(sample_result, f)


# Make boxplot
def load_data():
    try:
        raise FileNotFoundError
        with open(OUT / "boxdata.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {
            "Category": [],
            "Coverage": [],
            "Time": [],
            "Tries": [],
            "Success": [],
            "Folder": [],
        }

        with open(OUT / "qualitative.csv", "r") as f:
            success_analysis = pd.read_csv(f).to_numpy()[:, 2]

        trial_folders = [s for s in listdir(ROOT1) if "ground_truth" not in s.name]

        for sample, success in zip(trial_folders, success_analysis):
            with open(sample / "observation_result/result.json", "r") as f:
                coverage = json.load(f)["ratio"]
            towel_nr = sample.name.split("_")[2]

            fname = ROOT2 / sample.name[7:14] / sample.name[-26:] / "stats.json"

            with open(
                fname,
                "r",
            ) as f:
                stats = json.load(f)
            data["Time"].append(stats["time"])
            data["Tries"].append(stats["tries"])

            data["Category"].append(f"{int(towel_nr)}")
            data["Coverage"].append(coverage)
            data["Success"].append(success)
            data["Folder"].append(sample.as_posix())

        with open(OUT / "boxdata.json", "w+") as f:
            json.dump(data, f)
    return data


def gimme_plot(title: str, xlabel: str, ylabel: str) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    plt.tight_layout()
    return fig, ax


def get_df(data, independent_variable, dependent_variable, compute_total=True):
    if compute_total:
        return pd.DataFrame(
            {
                independent_variable: [
                    "Total",
                ]
                * len(data[independent_variable])
                + data[independent_variable],
                dependent_variable: data[dependent_variable] + data[dependent_variable],
            }
        )
    else:
        return pd.DataFrame(
            {
                independent_variable: data[independent_variable],
                dependent_variable: data[dependent_variable],
            }
        )


def success_rate():

    # Custom class to hold colors
    class SplitPatch:
        def __init__(self, colors):
            self.colors = colors

    # Custom Handler to create a split patch
    class HandlerSplitPatch:
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            # Get the colors
            colors = orig_handle.colors
            width, height = handlebox.width, handlebox.height

            y = [0, 0, height, height]
            x = [0, height - 1, height - 1, 0]
            x2 = [height, width, width, height]

            patch1 = plt.Polygon(
                np.column_stack([x, y]), edgecolor="none", facecolor=colors[0]
            )
            patch2 = plt.Polygon(
                np.column_stack([x2, y]), edgecolor="none", facecolor=colors[1]
            )

            handlebox.add_artist(patch1)
            handlebox.add_artist(patch2)
            return [patch1, patch2]

    data = load_data()
    df = get_df(data, "Category", "Success", compute_total=True)
    _, ax = gimme_plot(
        "Success Rate of Grasping Adjacent Corners",
        "Cloth Instance",
        "Success Rate (%)",
    )
    grouped = df.groupby("Category").size().reset_index(name="Count")
    grouped["Success"] = df.groupby("Category").sum().reset_index(drop=True)
    grouped["Success Rate"] = grouped["Success"] / grouped["Count"] * 100
    grouped["Failure Rate"] = 100 - grouped["Success Rate"]
    grouped = pd.concat(
        [
            grouped[grouped["Category"] == "Total"],
            grouped[grouped["Category"] != "Total"],
        ]
    ).reset_index(drop=True)

    # Define colors
    colors_success = [
        UGENT.YELLOW if cat == "Total" else UGENT.SKY for cat in grouped["Category"]
    ]
    colors_failure = [
        UGENT.LIGHTGREEN if cat == "Total" else UGENT.AQUA
        for cat in grouped["Category"]
    ]

    # Plot the bars
    ax.bar(
        grouped["Category"],
        grouped["Success Rate"],
        color=colors_success,
        label="Success",
    )
    ax.bar(
        grouped["Category"],
        grouped["Failure Rate"],
        bottom=grouped["Success Rate"],
        color=colors_failure,
        label="Failure",
    )

    # Custom legend handles
    success_handle = SplitPatch(colors=[UGENT.YELLOW, UGENT.SKY])
    failure_handle = SplitPatch(colors=[UGENT.LIGHTGREEN, UGENT.AQUA])

    # Add custom legend
    ax.legend(
        handles=[failure_handle, success_handle],
        labels=["Failure", "Success"],
        loc="lower right",
        handler_map={SplitPatch: HandlerSplitPatch()},
        fontsize=FONTSIZE,
    )

    plt.savefig(OUT / "success_rate.png")


def towels():
    # compute relative coverage
    w, _ = FIGSIZE

    fig, axes = plt.subplots(2, 5, figsize=(w, w / 2))
    axes = axes.flatten()
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
        ax.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
    for towel_nr in range(10):
        path = (
            ROOT1
            / f"sample_towel_{towel_nr}_ground_truth/observation_result/image_left.png"
        )
        img = Image.open(path)

        crop_box = (575, 0, img.width - 500, img.height - 400)
        img = img.crop(crop_box)

        axes[towel_nr].imshow(img)
        axes[towel_nr].set_title(towel_nr, fontsize=FONTSIZE)
        axes[towel_nr].axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "towels.png")
    plt.close(fig)  # Close the figure to release memory

    success_img = Image.open(OUT / "towels.png")
    towels_img = Image.open(OUT / "success_rate.png")

    # Vertically concatenate the images
    total_height = success_img.height + towels_img.height
    max_width = max(success_img.width, towels_img.width)

    concatenated_img = Image.new("RGB", (max_width, total_height))

    concatenated_img.paste(success_img, (0, 0))
    concatenated_img.paste(towels_img, (0, success_img.height))

    concatenated_img.save(OUT / "concatenated_image.png")


def coverage():
    data = pd.DataFrame(load_data())
    data = data[data["Success"] == 1]
    data = {
        "Category": data["Category"].to_list(),
        "Coverage": (data["Coverage"] * 100).to_list(),
    }
    df = get_df(data, independent_variable="Category", dependent_variable="Coverage")

    _, ax = gimme_plot(
        "Coverage of Completed Trials", "Cloth Instance", "% of Surface Visible"
    )
    sns.boxplot(
        x="Category",
        y="Coverage",
        data=df,
        showfliers=True,
        whis=np.inf,
    )
    ax.set_ylim(bottom=0.0, top=105)

    for ii, patch in enumerate(ax.patches):
        face_color, edge_color = (
            (UGENT.YELLOW, UGENT.LIGHTGREEN) if ii == 0 else (UGENT.SKY, UGENT.AQUA)
        )
        patch.set_facecolor(face_color)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(0)
        patch.set_alpha(1)

    for ii, line in enumerate(ax.get_lines()):
        color = UGENT.LIGHTGREEN if ii < 6 else UGENT.AQUA
        line.set_color(color)

        # Adjust the median line length
        if ii % 6 == 4:  # Median lines are the 5th line in each set of 6
            line.set_linewidth(1.5)  # Make the median line thicker
            offset = np.array([0.0, 0.00])
            line.set_xdata(line.get_xdata() - offset)

            # line.set_xdata(line.get_xdata()[1:3])  # Set the x-data to match the box width

    plt.savefig(OUT / "coverage.png")


def timecost():
    data = pd.DataFrame(load_data())
    data = data[data["Success"] == 1]

    fig, axes = plt.subplots(
        2, 1, figsize=FIGSIZE
    )  # Create a figure with 2 subplots vertically
    for ax in axes.flatten():
        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
        ax.tick_params(axis="both", which="minor", labelsize=FONTSIZE)

    # Plot the first histogram on the first subplot
    sns.histplot(
        data["Time"].to_numpy(),
        ax=axes[0],
        color=UGENT.SKY,
        alpha=1,
        linewidth=1,
        bins=16,
        edgecolor=UGENT.AQUA,
    )
    axes[0].set_title("Distribution of Trial Duration", fontsize=FONTSIZE)
    axes[0].set_xlabel("Time (s)", fontsize=FONTSIZE)
    axes[0].set_ylabel("Count", fontsize=FONTSIZE)

    # Plot the second histogram on the second subplot
    sns.histplot(
        data["Tries"].to_numpy(),
        ax=axes[1],
        color=UGENT.SKY,
        alpha=1,
        linewidth=1,
        edgecolor=UGENT.AQUA,
        discrete=True,
        bins=int(data["Tries"].max() - data["Tries"].min() + 1),
    )
    axes[1].set_title("Required Number of Attempts", fontsize=FONTSIZE)
    axes[1].set_xlabel("Attempts", fontsize=FONTSIZE)
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].set_ylabel("Count", fontsize=FONTSIZE)

    plt.tight_layout()  # Adjust the layout to prevent overlap
    plt.savefig(OUT / "compcost.png")


def statistics():
    data = pd.DataFrame(load_data())

    success_rate = np.mean(data["Success"].to_numpy()) * 100
    overall_mean = np.mean(data["Coverage"].to_numpy()) * 100
    overall_median = np.median(data["Coverage"].to_numpy()) * 100
    overall_time = np.mean(data["Time"].to_numpy())
    overall_tries = np.mean(data["Tries"].to_numpy())
    pyout(
        f"OVERALL:\n"
        f"success rate: {success_rate:.2f} %\n"
        f"mu:           {overall_mean:.2f} %\n"
        f"median:       {overall_median:.2f} %\n"
        f"time:         {overall_time:.2f} s\n"
        f"attempts:     {overall_tries:.2f}\n"
    )

    data = data[data["Success"] == 1]
    complete_mean = np.mean(data["Coverage"].to_numpy()) * 100
    complete_median = np.median(data["Coverage"].to_numpy()) * 100
    complete_std = np.std(data["Coverage"].to_numpy()) * 100
    complete_time = np.mean(data["Time"].to_numpy())
    complete_tries = np.mean(data["Tries"].to_numpy())
    pyout(
        f"OVERALL:\n"
        f"mu:       {complete_mean:.2f} %\n"
        f"median:   {complete_median:.2f} %\n"
        f"time:     {complete_time:.2f} s\n"
        f"attempts: {complete_tries:.2f}\n"
    )

    with open(OUT / "stats.txt", "w+") as f:
        f.write(
            f"OVERALL:\n"
            f"success rate: {success_rate:.2f} %\n"
            f"mu:           {overall_mean:.2f} %\n"
            f"median:       {overall_median:.2f} %\n"
            f"time:         {overall_time:.2f} s\n"
            f"attempts:     {overall_tries:.2f}\n"
            "\n"
            f"COMPLETE:\n"
            f"mu:       {complete_mean:.2f} %\n"
            f"median:   {complete_median:.2f} %\n"
            f"std:      {complete_std:.2f} %\n"
            f"time:     {complete_time:.2f} s\n"
            f"attempts: {complete_tries:.2f}\n"
        )


def train_validation_set():
    root = Path("/media/matt/Expansion/Datasets/towels_large")
    pyout(len(listdir(root)) / 945)

    root = Path("/media/matt/Expansion/Datasets/towels_coco_large")
    train_size = len(listdir(Path(root / "train" / "images")))
    valid_size = len(listdir(Path(root / "validation" / "images")))

    pyout(f"{train_size / (train_size + valid_size):.3f}")

    return

    jj = 0
    for trial in listdir(root):
        jj += 1
        if not jj % 5 == 0:
            continue
        video_path = (
            root / trial / "video.mp4"
        )  # Assuming the video is within each trial directory

        if video_path.is_file():
            cap = cv2.VideoCapture(str(video_path))

            ii = 0
            while cap.isOpened():
                ii += 1

                ret, frame = cap.read()
                if not ret:
                    break
                if ii % 5:
                    cv2.imshow("Video", frame)

                # Press 'q' to exit the video display
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            pyout(f"Video file not found: {video_path}")


success_rate()
coverage()
timecost()
towels()
# statistics()
# train_validation_set()
