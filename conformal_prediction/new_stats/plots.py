import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


IMAGE_TYPE_COLORS = {
    "normal":            "#FF9800",  # orange
    "monochrome":        "#9E9E9E",  # grey
    "gaussian_blur":     "#4CAF50",  # green
    "vertical_blur":     "#8BC34A",  # light green
    "horizontal_blur":   "#2E7D32",  # dark green
    "glass_blur":        "#009688",  # teal
    "atmospheric_fog":   "#00BCD4",  # cyan
    "rain":              "#3F51B5",  # indigo
}


def get_base(image_type):
    return next((k for k in IMAGE_TYPE_COLORS if image_type.startswith(k)), image_type)


def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def get_shaded_colors(image_types):
    groups = defaultdict(list)
    for it in image_types:
        base = get_base(it)
        if base != "normal":
            groups[base].append(it)

    color_map = {"normal": hex_to_rgb(IMAGE_TYPE_COLORS["normal"])}
    for base, members in groups.items():
        r, g, b = hex_to_rgb(IMAGE_TYPE_COLORS.get(base, "#607D8B"))
        n = len(members)
        for idx, member in enumerate(members):
            t = 1.0 - 0.5 * (idx / max(n - 1, 1))  # bright → dark
            color_map[member] = (r*t, g*t, b*t)
    return color_map


def make_legend_handles(image_types):
    seen, handles = set(), []
    for it in image_types:
        base = get_base(it)
        if base not in seen:
            seen.add(base)
            handles.append(mpatches.Patch(
                color=hex_to_rgb(IMAGE_TYPE_COLORS.get(base, "#607D8B")),
                label=base
            ))
    return handles


def plot(
    dataset_path: str = "/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone",
    indices: list[int] = [1, 3, 9, 17, 19, 21, 22],
    length=10,
    pickle_path="pred_sets_overall",
    image_types=["normal", "gaussian_blur", "salt_and_pepper"],
):
    dataset_path = Path(dataset_path)
    class_names  = list(pd.read_csv(dataset_path / "classes.csv")["name"].values)
    classes      = [class_names[i] for i in indices]

    # Load all data upfront
    mean_arr = np.zeros((len(classes), len(image_types)))
    std_arr  = np.zeros((len(classes), len(image_types)))
    for i, image_type in enumerate(image_types):
        with open(f"{pickle_path}/{pickle_path}_{length}_type={image_type}.pkl", "rb") as f:
            data = pickle.load(f)
        for j, cls in enumerate(classes):
            mean_arr[j, i] = data[cls]["overall_mean"]
            std_arr[j, i]  = data[cls]["overall_std"]

    # Group non-normal types into subplots
    groups = defaultdict(list)
    for it in image_types:
        base = get_base(it)
        if base != "normal":
            groups[base].append(it)

    color_map  = get_shaded_colors(image_types)
    normal_idx = image_types.index("normal") if "normal" in image_types else None

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 6), sharey=True)
    if n_groups == 1:
        axes = [axes]

    x = np.arange(len(classes))

    for base, members in groups.items():
        bars  = (["normal"] + members) if normal_idx is not None else members
        N     = len(bars)
        width = 0.8 / N

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, member in enumerate(bars):
            col_idx = image_types.index(member)
            val     = member.replace(f"{get_base(member)}_", "") if "_" in member else "default"
            label   = "normal" if member == "normal" else f"{base}={val}"
            ax.bar(
                x + i * width,
                mean_arr[:, col_idx],
                width,
                yerr=std_arr[:, col_idx],
                color=color_map[member],
                capsize=4,
                label=label,
            )

        ax.set_title(f"Prediction set size — {base}")
        ax.set_xticks(x + width * (N - 1) / 2)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylabel("Mean prediction set size")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

        plt.tight_layout()
        save_path = f"plots/prediction_set_{base}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {save_path}")


if __name__ == "__main__":
    image_type_dict = {
        "normal": [],
        "monochrome": [],
        "gaussian_blur": [2,4,6,8],
        "vertical_blur": [11,21,31,41],
        "horizontal_blur": [11,21,31,41],
        "glass_blur": [1,2,3,4],
        "atmospheric_fog": [0.05, 0.10, 0.15, 0.20],
        "rain": [0.85, 0.70, 0.55, 0.40]
    }
    num_images = 10
    
    image_types = []
    for image_type, vals in image_type_dict.items():
        if vals:
            for val in vals:
                image_types.append(f"{image_type}_{val}")
        else:
            image_types.append(image_type)

    plot(
        dataset_path="/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone",
        indices=[1, 3, 9, 17, 19, 21, 22],
        length=num_images,
        pickle_path="pred_sets_overall",
        image_types=image_types,
    )