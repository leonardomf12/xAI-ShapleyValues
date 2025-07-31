import time
from tqdm import tqdm
import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def labels_count(df0):
    df_ = df0.copy()

    # Initial filtering
    try:
        df_ = df_[~(df_["male"] == "male")]  # Remove weirds rows in the dataset
        df_.drop(columns=["path", "label"], inplace=True)
    except ValueError:
        pass


    # Value mapping
    value_map = {"1": "True", "0": "unk", "-1": "False"}
    df_.replace(value_map, inplace=True)

    # Count Labels accounting for unknown values
    label_count_dict = {
        col: df_[col].value_counts(normalize=True).reindex(["unk", "False", "True"], fill_value=0)
        for col in df_.columns
    }

    return label_count_dict, sorted(df_.columns, key=lambda col: label_count_dict[col]["unk"])

def compute_plots(label_dict, order, name):
    fig, axes = plt.subplots(7, 7, figsize=(20, 15))
    axes = axes.flatten()

    for i, col in enumerate(order):  # Unknown order
        ax = axes[i]

        ax.barh(label_dict[col].index.astype(str), label_dict[col] * 100)
        ax.set_title(f"{col} -> {np.round(label_dict[col]['unk'], decimals=3)}", fontsize=8)
        ax.tick_params(labelsize=6)

    for j in range(len(label_dict.keys()), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(ANNOTATIONS_DIR / f"{name}.png")

ANNOTATIONS_DIR = Path(__file__).parent.parent
BUPTBalancedFace_DATASET_PATHS = [
    "maad_dataset_attributes_balancedFace_African.csv",
    "maad_dataset_attributes_balancedFace_Asian.csv",
    "maad_dataset_attributes_balancedFace_Caucasian.csv",
    "maad_dataset_attributes_balancedFace_Indian.csv"
]

df_all = []
for path in tqdm(BUPTBalancedFace_DATASET_PATHS, total=len(BUPTBalancedFace_DATASET_PATHS) + 1):
    name = path.split("_")[-1].split(".")[0]

    # Initial Cleaning
    df = pd.read_csv(ANNOTATIONS_DIR / "maad_balanceface_labels" / path)
    df_all.append(df)

    label_count_dict, col_unk_order = labels_count(df)
    compute_plots(label_count_dict, order = col_unk_order, name=name)

#Also for all together
df_all = pd.concat(df_all, axis=0)
label_count_dict, col_unk_order = labels_count(df_all)
compute_plots(label_count_dict, col_unk_order, name="all")
