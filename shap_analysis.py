import time

import IPython
import numpy as np
import pandas as pd
import shap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from pathlib import Path
from rich.traceback import install
install()
import torch
from torch import nn


# My scripts
from dataloader import MyDataLoader, Data2MTL
from architectures import FACENETModel, CLIPModel, iResNet100Model

def ind_subplot(ax, x_t, x_f, label, tol=30):

    feat = np.arange(x_f.shape[0])
    sort_mask = np.argsort(-x_t)

    # Plots
    ax.plot(feat, x_t[sort_mask] * 1e3, zorder=1,  linewidth=3, label=f"{label} - True - p0.5")
    ax.plot(feat, x_f[sort_mask] * 1e3, zorder=0, label=f"{label} - False - p0.5")

    # Ticks
    cross_idx = np.argmin(np.abs(x_t[sort_mask])) # idx with

    x_ticks = list(ax.get_xticks())
    labels = [str(int(t)) for t in x_ticks]
    # Case where ticks are too close
    min_ticks = np.argmin(np.abs(np.array(x_ticks) - cross_idx))
    if np.abs(x_ticks[min_ticks] - cross_idx) < tol:
        x_ticks[min_ticks] = cross_idx
        labels[min_ticks] = str(cross_idx)
    else:
        x_ticks.append(cross_idx)
        labels.append(str(cross_idx))

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)

    ax.set_xlim(np.min(feat) - 10, np.max(feat) + 10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='x', labelsize=12)

    # Lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=1) # Horizontal
    ax.axvline(cross_idx, color='grey', linestyle='--', linewidth=1) #Vertical

    # Legend
    ax.legend(fontsize=8)

def compute_ranking_plot(**kwargs):
    required_keys = {"facenet_true", "facenet_false", "clip_true", "clip_false", "resnet100_true", "resnet100_false"}
    assert required_keys.issubset(kwargs)
    class_names = ["male", "female", "African", "Asian", "Caucasian", "Indian", "Square Face", "Eyeglasses", "Pointy Nose"]

    for class_idx in range(len(class_names)):
        # Main Figure
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=1, nrows=3, figsize=(6, 6))

        # Figure 1
        x_facenet_true = kwargs["facenet_true"][:, class_idx]
        x_facenet_false = kwargs["facenet_false"][:, class_idx]
        ind_subplot(ax0, x_facenet_true, x_facenet_false, label="FACENET")
        ax0.xaxis.label.set_visible(False)

        # Figure 2
        x_clip_true = kwargs["clip_true"][:, class_idx]
        x_clip_false = kwargs["clip_false"][:, class_idx]
        ind_subplot(ax1, x_clip_true, x_clip_false, label="CLIP")
        ax1.xaxis.label.set_visible(False)
        ax1.set_ylabel(r"$\mathrm{SHAP\ values},\ \phi_j\ \left(\times 10^{-3} \right)$", fontsize=14)

        # Figure 3
        x_resnet100_true = kwargs["resnet100_true"][:, class_idx]
        x_resnet100_false = kwargs["resnet100_false"][:, class_idx]
        ind_subplot(ax2, x_resnet100_true, x_resnet100_false, label="RESNET100", tol=50)
        ax2.set_xlabel("$features, f_i$", fontsize=14)

        fig.suptitle(f"Class: {class_names[class_idx]}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25)
        plt.savefig(SAVE_DIR / f"shap-dist-{class_names[class_idx]}.jpg")

def calculate_spearman_mat(shap_values_: list):
    n = len(shap_values_)
    spearman_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:
                try:
                    coef, _ = spearmanr(shap_values_[i], shap_values_[j])
                    spearman_matrix[i, j] = coef
                    spearman_matrix[j, i] = coef  # symmetric
                except ValueError:
                    pass
    return spearman_matrix

def print_spearman_table(mat0: np.ndarray):
    #mat = mat0.copy()[0]
    mat = np.mean(mat0, axis=0)

    shap_names = ["F_T", "F_F", "C_T", "C_F", "R_T", "R_F"]

    df = pd.DataFrame(mat, columns=shap_names, index=shap_names)
    df = df.round(3)
    print(df)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Save to file (use .tex or .txt based on your workflow)
    with open(SAVE_DIR / "spearman_table.tex", "w") as f:
        f.write(df.to_latex(index=True))

def compute_ranking_table(**kwargs):
    required_keys = {"facenet_true", "facenet_false", "clip_true", "clip_false", "resnet100_true", "resnet100_false"}
    assert required_keys.issubset(kwargs)

    facenet_t = kwargs["facenet_true"]
    facenet_f = kwargs["facenet_false"]
    clip_t = kwargs["clip_true"]
    clip_f = kwargs["clip_false"]
    resnet_t = kwargs["resnet100_true"]
    resnet_f = kwargs["resnet100_false"]

    spearman_mat_class = []
    for class_idx in range(facenet_t.shape[-1]):
        mat_class = calculate_spearman_mat([
            facenet_t[:, class_idx],
            facenet_f[:, class_idx],
            clip_t[:, class_idx],
            clip_f[:, class_idx],
            resnet_t[:, class_idx],
            resnet_f[:, class_idx]
        ])
        spearman_mat_class.append(mat_class)

    spearman_mat = np.concatenate(np.expand_dims(spearman_mat_class, axis=0), axis=0)
    #spearman_mat = np.mean(np.concatenate(np.expand_dims(spearman_mat_class, axis=0), axis = 0), axis=0)

    print_spearman_table(spearman_mat)


def compute_mapping_backwards(**kwargs):
    required_keys = {"facenet_true", "facenet_false", "clip_true", "clip_false", "resnet100_true", "resnet100_false"}
    required_keys_val = {"facenet_true_val", "facenet_false_val", "clip_true_val", "clip_false_val", "resnet100_true_val", "resnet100_false_val"}
    assert required_keys.issubset(kwargs)
    assert required_keys_val.issubset(kwargs)



    #val_labels = kwargs["val_labels"] TODO Delete loader from script
    class_names = ["male", "female", "African", "Asian", "Caucasian", "Indian", "Square Face", "Eyeglasses", "Pointy Nose"]

    # Separate dictionaries for each method
    metrics = {}
    for shap_test, shap_val in zip(sorted(list(required_keys)), sorted(list(required_keys_val))):
        metrics[shap_test] = {}
        shap_ref = kwargs[shap_test].T # (512, 9) -> (9, 512)
        shap_to_compute = kwargs[shap_val].transpose(0, 2, 1).reshape(-1, shap_ref.shape[-1]) # (b, 512, 9) -> (b, 9, 512) -> (b*9, 512)
        y_class = np.array(list(range(len(class_names))) * (len(shap_to_compute) // len(class_names))) # True SHAP values

        # Shap Values Head computing
        epsilon = 1e-10
        shap_pred_list = []
        for shap in tqdm(shap_to_compute, total=len(shap_to_compute)):
            shap = np.expand_dims(shap, axis=0)
            mask = (np.abs(shap[0]) > epsilon)

            # COSINE SIMILARITY
            cos_pred = np.argmax(cosine_similarity(shap_ref[:, mask], shap[:, mask]))

            # Spearman Coefficient
            spearman_arr = np.vstack([shap[:, mask], shap_ref[:, mask]])
            corr_matrix, _ = spearmanr(spearman_arr, axis=1)
            spearman_pred = np.argmax(corr_matrix[0, 1:]) # I take 0 out because shap goes first on vstack!

            # Matmul
            matmul_pred = np.argmax(np.matmul(shap_ref, shap.T))

            shap_pred_list.append(np.array([cos_pred, spearman_pred, matmul_pred]))

        # Computing Confusion Matrix
        shap_pred = np.vstack(shap_pred_list)
        metrics_h = {"mcc": [], "acc": [], "prec": [], "rec": [], "f1": []}

        for class_idx in range(len(class_names)):
            y_shap_pred_bool = (shap_pred == class_idx)
            y_class_bool = (y_class == class_idx)[:, np.newaxis]

            TP = np.sum(np.logical_and(y_shap_pred_bool, y_class_bool), axis=0)
            TN = np.sum(np.logical_and(~y_shap_pred_bool, ~y_class_bool), axis=0)
            FP = np.sum(np.logical_and(y_shap_pred_bool, ~y_class_bool), axis=0)
            FN = np.sum(np.logical_and(~y_shap_pred_bool, y_class_bool), axis=0)

            # Metrics (1, 3)
            mcc_ = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            metrics_h["mcc"].append(mcc_)

            acc = (TP + TN) / (TP + TN + FP + FP)
            metrics_h["acc"].append(acc)

            prec = TP / (TP + FP)
            metrics_h["prec"].append(prec)

            rec = TP / (TP + FN)
            metrics_h["rec"].append(rec)

            f1 = 2 * TP / (2 * TP + FP + FN)
            metrics_h["f1"].append(f1)

        metrics[shap_test]["mcc"] = np.vstack(metrics_h["mcc"])
        metrics[shap_test]["acc"] = np.vstack(metrics_h["acc"])
        metrics[shap_test]["prec"] = np.vstack(metrics_h["prec"])
        metrics[shap_test]["rec"] = np.vstack(metrics_h["rec"])
        metrics[shap_test]["f1"] = np.vstack(metrics_h["f1"])

    process_metrics_to_array(metrics, class_names=class_names)

def process_metrics_to_array(metrics: dict, class_names: list):
    models = list(metrics.keys())
    metrics_name = list(metrics[models[0]].keys())
    approach = ["cosine_similarity", "spearman", "matmul"]

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 10000)

    for metric in metrics_name:
        for app in approach:
            # The outputs of this loop should be each table
            app_idx = {"cosine_similarity": 0, "spearman": 1, "matmul": 2}[app]
            rows = [metrics[model][metric][:, app_idx] for model in models]
            data = np.vstack(rows)

            df = pd.DataFrame(data, columns=class_names, index=models)
            print("*" * 20 + f"\nMetric: {metric} -> Approach: {app}\n")
            print(df)


            SAVE_DIR2 = SAVE_DIR / "shap_backwards_computation" / f"{metric}-{app}-shap-table.tex"
            SAVE_DIR2.parent.mkdir(parents=True, exist_ok=True)
            with open(SAVE_DIR2, "w") as f:
                f.write(df.to_latex(index=True))

# TODO PLOT for all models
def compute_shap_plot(x_orig, x_shap):

    x_s = x_shap[0].transpose((3, 1, 2, 0)) # (9, 112, 112, 3)
    x = x_orig.transpose(1, 2, 0) # (112, 112, 3)


    # Normalization
    x_s = (x_s - np.min(x_s)) / (np.max(x_s) - np.min(x_s))
    #x_shap_norm = np.vstack([np.expand_dims((s_i - np.min(s_i)) / (np.max(s_i) - np.min(s_i)), axis = 0) for s_i in x_s])
    print(x_s.shape)

    # Figure
    fig = plt.figure(constrained_layout=True)

    # GridSpaces
    #grid_parent = GridSpec(1, 2, width_ratios=[1, 3], wspace=0, hspace=0, figure=fig)
    grid_parent = GridSpec(1, 1, wspace=0, hspace=0, figure=fig)

    ## GridSpace: Original Image
    #grid_orig = GridSpecFromSubplotSpec(1, 1, wspace=0, hspace=0, subplot_spec=grid_parent[0])
    #ax_orig = fig.add_subplot(grid_orig[0, 0])
    #ax_orig.imshow((x * 255).astype(np.uint8))
    #ax_orig.axis('off')

    ## GridSpace: Saliency maps
    grid_shap = GridSpecFromSubplotSpec(4, 9, wspace=0, hspace=0, subplot_spec=grid_parent[0])
    alpha = 0.2
    for i in range(4): # TODO Change this to different models
        for j in range(x_s.shape[0]):
            img_saliency = alpha * x + (1 - alpha) * x_s[j]
            img_saliency_int = (img_saliency * 255).astype(np.uint8)


            ax_ = fig.add_subplot(grid_shap[i, j])
            ax_.imshow(img_saliency_int)
            ax_.axis("off")
    plt.show()
    #...

if __name__ == '__main__':
    DIR = Path(__file__).parent
    SHAP_DIR = DIR / "dataset" / "shap"
    SAVE_DIR = DIR / "shap_values" / "plots"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Validation Dataloader
    dataloader = MyDataLoader(
        Data2MTL(state="valid", mode="embeddings", model_emb="facenet", device='cpu',
                 fraction=0.1),
        batch_size=5001
    )
    _, val_labels = next(iter(dataloader))
    val_labels = val_labels.cpu().numpy()
    shap_files = list(SHAP_DIR.rglob("*.npy"))

    # Sorting order: Clip -> Facenet -> Resnet101 | False -> True
    shap_val_list = sorted([f for f in shap_files if "valid" in str(f) and "0.5" in str(f)])
    shap_test_list = sorted([f for f in shap_files if "test" in str(f) and "0.5" in str(f)])

    # Test Shap Values
    x_clip_f = np.mean(np.load(shap_test_list[0]), axis=0)
    x_clip_t = np.mean(np.load(shap_test_list[1]), axis=0)
    x_facenet_f = np.mean(np.load(shap_test_list[2]), axis=0)
    x_facenet_t = np.mean(np.load(shap_test_list[3]), axis=0)
    x_resnet100_f = np.mean(np.load(shap_test_list[4]), axis=0)
    x_resnet100_t = np.mean(np.load(shap_test_list[5]), axis=0)


    #Methodology .1: Shap define a distribution
    compute_ranking_plot(
        clip_false = x_clip_f,
        clip_true = x_clip_t,
        facenet_false = x_facenet_f,
        facenet_true = x_facenet_t,
        resnet100_false= x_resnet100_f,
        resnet100_true = x_resnet100_t,
    )
    #
    # compute_ranking_table(
    #     clip_false = x_clip_f,
    #     clip_true = x_clip_t,
    #     facenet_false = x_facenet_f,
    #     facenet_true = x_facenet_t,
    #     resnet100_false= x_resnet100_f,
    #     resnet100_true = x_resnet100_t,
    # )

    # Methodology .2: Can we track back which head is being classified by looking at shap values?
    # x_clip_f_val = np.load(shap_val_list[0])[:5001]
    # x_clip_t_val = np.load(shap_val_list[1])
    # x_facenet_f_val = np.load(shap_val_list[2])
    # x_facenet_t_val = np.load(shap_val_list[3])
    # x_resnet100_f_val = np.load(shap_val_list[4])
    # x_resnet100_t_val = np.load(shap_val_list[5])
    #
    # compute_mapping_backwards(
    #     val_labels = val_labels,
    #     clip_false = x_clip_f,
    #     clip_true = x_clip_t,
    #     facenet_false = x_facenet_f,
    #     facenet_true = x_facenet_t,
    #     resnet100_false= x_resnet100_f,
    #     resnet100_true = x_resnet100_t,
    #     clip_false_val=x_clip_f_val,
    #     clip_true_val = x_clip_t_val,
    #     facenet_false_val = x_facenet_f_val,
    #     facenet_true_val = x_facenet_t_val,
    #     resnet100_false_val = x_resnet100_f_val,
    #     resnet100_true_val = x_resnet100_t_val,
    # )

    # Methodology .3: Visual Saliency Map
    # train_dataloader = MyDataLoader(
    #     dataset=Data2MTL(
    #         state="train",
    #         device="cpu",
    #         mode="images",
    #     ),
    #     batch_size=10,
    # )
    # x_train, _ = next(iter(train_dataloader))
    # sample_to_compute = Data2MTL(state="test", device="cpu", mode="images")[100][0].cpu()
    #
    # model = FACENETModel(model_name="facenet", mode="images", FFN=True, device="cpu")
    # weights = Path.cwd() / "shap_values" / "models_shap" / "FACENETModel-True-p[0.5]__2025-05-23_11-00__e4.pth"
    # model.load_state_dict(torch.load(str(weights), map_location=torch.device('cpu'))["model_weights"], strict=False)
    # model.eval()
    #
    #
    # class SHAPWrapper(nn.Module):
    #     def __init__(self, model):
    #         super().__init__()
    #         self.model = model
    #
    #     def forward(self, x):
    #         # Return only the tensor used in predict_tensor
    #         return self.model.predict_tensor(x)
    #
    # model_wrapper = SHAPWrapper(model)
    #
    # explainer = shap.GradientExplainer(model_wrapper, x_train)
    # img_shap = explainer.shap_values(torch.unsqueeze(sample_to_compute, dim=0).requires_grad_())
    #
    # compute_shap_plot(x_orig=sample_to_compute.numpy(), x_shap=img_shap)

    #IPython.embed()