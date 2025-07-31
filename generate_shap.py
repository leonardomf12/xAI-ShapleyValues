import argparse
import time

import yaml
from torch import nn, device

from skimage.segmentation import slic
import matplotlib.pyplot as plt
from typing import Dict
import math
from pydantic.experimental.pipeline import transform
from pathlib import Path
from architectures import FACENETModel, iResNet100Model, CLIPModel

import torch
import IPython
import shap
import numpy as np
import random

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# My scripts
from dataloader import MyDataLoader, Data2MTL
import architectures as model_arch_scr

class RepresentativeBackgroundData:
    def __init__(self, model_emb, **kwargs):
        fraction = kwargs.get("fraction", 1.0)
        self.model_emb = model_emb
        device = "cpu"

        # DataLoader
        dataloader = MyDataLoader(
            Data2MTL(state="train", mode="embeddings", model_emb=self.model_emb, device=torch.device(device), fraction=fraction),
            batch_size=5000
        )

        x_train_list = []
        i=0
        for x, _ in tqdm(dataloader, total=len(dataloader)):
            x_train_list.append(x)
            i +=1
            if i > int(len(dataloader)*0.025):
                break
        self.x_train = torch.cat(x_train_list, dim=0)
        self.x_train = torch.squeeze(self.x_train, dim=1).numpy()
        print("DataLoader initialized successfully!")

        self.k_list = [10, 25, 50, 100, 200, 500, 1000]
        for k in tqdm(self.k_list, total=len(self.k_list)):
            bg_data = self.compute_k_representative(k=k)
            self.save_bgdata(bg_data, k)

    def compute_k_representative(self, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        #kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, max_iter=100)
        kmeans.fit(self.x_train)

        k_samples = []
        for k_id in range(k):
            mask = kmeans.labels_ == k_id
            xk = self.x_train[mask]

            dist = np.sqrt(((xk - np.expand_dims(kmeans.cluster_centers_[1], axis=0)) ** 2).sum(axis=1))
            k_samples.append(xk[np.argmin(dist)])

        bg_data = torch.from_numpy(np.vstack(k_samples))

        # Final Verification - Values weren't changed!
        sample_match = []
        for i, sample in enumerate(bg_data):
            sample_match.append(np.any(np.all(np.isclose(self.x_train, sample, atol=1e-6), axis=1)))

        assert np.array(sample_match).all(), f"❌ Some samples after kmeans don't match! {np.sum(~np.array(sample_match))}"
        return bg_data

    def save_bgdata(self, bg_data, k):
        bg_data_name = f"bg_tensor-{k}.pt"

        dataset_path = Path(__file__).parent / "dataset"
        assert dataset_path.exists(), "Dataset path not found!"

        save_path = dataset_path / "shap" / self.model_emb / bg_data_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bg_data, save_path)
        print(f"💾 Background data saved at: {save_path}")

class MySHAP:
    def __init__(self, weights_path, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        assert "mode" in kwargs.keys(), "Must provide mode!"
        assert kwargs["mode"] in ["images", "embeddings"], "Mode selected not supported!"

        # Loading Weight
        self.model_status = torch.load(weights_path)

        self.model_name = self.model_status["model_name"]
        model_arch_name = self.model_name.split("-")[0]
        self.FFN = self.model_status["FFN"]
        self.mode = kwargs["mode"]
        self.model_emb = {"FACENETModel": "facenet", "iResNet100Model": "resnet101", "CLIPModel": "clip"}[model_arch_name]

        # Loading Model Architecture
        self.model = getattr(model_arch_scr, model_arch_name)(
            model_name=self.model_name,
            FFN=self.FFN,
            mode=self.mode,
            device=self.device
        )
        try:
            self.model.load_state_dict(self.model_status["model_weights"], strict=False)
            print("🔄 Model weights loaded successfully! ✅")
        except RuntimeError as e:
            raise ValueError(f"❌ Weights don't fit the model! \n Error: {e}")
        self.model.eval()

        # DataLoader
        bg_size = kwargs["bg_size"]
        # background_dataloader = MyDataLoader(
        #     dataset=Data2MTL(
        #         state="train",
        #         mode=self.mode,
        #         model_emb=self.model_emb,
        #         device="cpu"
        #     ),
        #     batch_size=self.bg_size,
        # )
        #self.bg_data, _ = next(iter(background_dataloader))
        #self.bg_data = torch.squeeze(self.bg_data, dim=1)
        bg_path = Path(__file__).parent / "dataset" / "shap" / self.model_emb / f"bg_tensor-{bg_size}.pt"
        self.bg_data = torch.load(bg_path)

        self.valid_dataloader = MyDataLoader(
            dataset=Data2MTL(
                state="valid",
                mode=self.mode,
                model_emb=self.model_emb,
                device="cpu"
            ),
            batch_size=1,
        )
        #self.img, _ = next(iter(img_dataloader))
        #self.img = torch.squeeze(self.img, dim=1)


class KernelShapEmbedding(MySHAP):
    def __init__(self, weights_path, **kwargs):
        super().__init__(weights_path, **kwargs)
        self.explainer = shap.KernelExplainer(self.predict_fn, self.bg_data.cpu().detach().numpy())

    def predict_fn(self, x_np):
        x = torch.from_numpy(x_np).to(self.device)
        x = self.model.predict_tensor(x)

        return x.cpu().detach().numpy()

    def calculate_shap(self, sset: str, batch_size: int):
        assert sset in ["valid", "test"], "Invalid subset selection!"

        dataloader = MyDataLoader(
            dataset=Data2MTL(
                state=sset,
                mode=self.mode,
                model_emb=self.model_emb,
                device="cpu"
            ),
            batch_size=batch_size,
        )

        shap_list = []
        for x, _ in tqdm(dataloader, total=len(dataloader)):
            x = torch.squeeze(x, dim=1)
            shap_values = self.explainer.shap_values(x.cpu().detach().numpy(), silent=True)
            shap_list.append(shap_values)

        shap_values = np.concatenate(shap_list, axis=0)
        return shap_values

    def save_shap(self, shap_values, sset: str, model_name: str):
        assert sset in ["valid", "test"], "Invalid subset selection!"
        filename = model_name.split("/")[-1].split("_")[0]

        shap_path = Path(__file__).parent / "dataset" / "shap" / self.model_emb / f"shap-{filename}-{sset}.npy"
        shap_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(shap_path, shap_values)


def args_parser():
    parser = argparse.ArgumentParser(description="Run Compute Shap Values!")
    # parser.add_argument(
    #     "-w", "--weights", type=str, required=False,
    #     help="Must indicate the path to the model weights!")
    # parser.add_argument(
    #     "-gbg", "--generate_background", action='store_true', default=False,
    #     help="Specify if you want to compute the background data or not!")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1000,
        help="Number of samples to be calculated at once within calculate_shap()"
    ),
    parser.add_argument(
        "-bs", "--background_size", type=int, default=50,
        help="Must indicate the path to the model weights!"),
    parser.add_argument(
        "-p", "--p_dropout", type=float, default=0.5,
        help="Specify dataloader fraction")
    return parser.parse_args()

def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data

def get_shap_list(p):
    repo_path = Path(__file__).parent
    w_files_list = list((repo_path / "shap_values" / "models_shap").rglob("*.pth"))

    vals_weigh_list = [str(w.relative_to(repo_path)) for w in w_files_list if f"p[{p}]" in str(w)]
    print(f"Weights being used: \n {vals_weigh_list}")
    return vals_weigh_list # Paths from repo

if __name__ == "__main__":
    args = args_parser()
    # Seeds
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    # Checks
    assert args.background_size in [10, 25, 50, 100, 200, 500, 1000], f"Invalid Background size!"

    # Setup
    shap_to_compute_list = get_shap_list(args.p_dropout)
    for shap_to_compute in shap_to_compute_list:
        weight_path = Path(shap_to_compute)
        assert weight_path.exists(), f"Weight path not found! {weight_path}"

        # Background Data
        model_emb = {"FACENETModel": "facenet", "iResNet100Model": "resnet101", "CLIPModel": "clip"}[shap_to_compute.split("/")[-1].split("-")[0]]
        if (Path(__file__).parent / "dataset" / "shap" / model_emb).exists():
            print("Background data already exists! Avoiding the computing...")
        else:
            RepresentativeBackgroundData(model_emb=model_emb)

        # Kernel SHAP
        print(f"Starting computing: {shap_to_compute}")
        kernel_shap = KernelShapEmbedding(weight_path, mode="embeddings", bg_size=args.background_size)

        print(f"Testing Subset Starting ...")
        shap_test = kernel_shap.calculate_shap("test", args.batch_size)
        kernel_shap.save_shap(shap_test, "test", shap_to_compute)

        print(f"Validation Subset Starting ...")
        shap_val = kernel_shap.calculate_shap("valid", args.batch_size)
        kernel_shap.save_shap(shap_val, "valid", shap_to_compute)

    #print(f"SHAP_VALUES SHAPE: {shap_val.shape}")
    #IPython.embed()