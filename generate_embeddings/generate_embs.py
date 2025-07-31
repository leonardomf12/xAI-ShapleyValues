import argparse
from multiprocessing import Pool, set_start_method
from pathlib import Path

import IPython
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
from rich.traceback import install
install()

# ===============================
# Base Embedding Generator Class
# ===============================
class EmbeddingGenerator:
    def __init__(self, model_name, dpath, spath):
        self.model_name = model_name
        self.dpath = dpath
        self.spath = spath
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None

    def setup(self):
        raise NotImplementedError

    def encode(self, image):
        raise NotImplementedError

    def save_embedding(self, path, embedding):
        inner_path = path.relative_to(self.dpath)
        fname = inner_path.stem + ".npy"
        full_spath = self.spath / inner_path.parent / fname
        full_spath.parent.mkdir(parents=True, exist_ok=True)
        np.save(full_spath, embedding)

    def process(self, path):
        image = Image.open(path).convert("RGB")
        tensor = torch.unsqueeze(self.preprocess(image), dim=0).to(self.device)
        with torch.inference_mode():
            emb = self.encode(tensor)
        self.save_embedding(path, emb.cpu().detach().numpy())

# =====================
# Model Implementations
# =====================

class CLIPEmbedding(EmbeddingGenerator):
    def setup(self):
        import clip
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def encode(self, x):
        return torch.squeeze(self.model.encode_image(x))

class ResNetEmbedding(EmbeddingGenerator):
    def setup(self):
        from torchvision.models import resnet101, ResNet101_Weights
        weights = ResNet101_Weights.IMAGENET1K_V2
        self.model = resnet101(weights=weights).to(self.device).eval()
        self.preprocess = weights.transforms()

    def encode(self, x):
        return torch.squeeze(self.model(x))

class FaceNetEmbedding(EmbeddingGenerator):
    def setup(self):
        from facenet_pytorch import InceptionResnetV1
        from torchvision import transforms
        self.model = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
        self.preprocess = transforms.Compose([transforms.ToTensor()])

    def encode(self, x):
        return torch.squeeze(self.model(x))

# ========================
# Multiprocessing Support
# ========================
def init_worker(model_name_, dpath_, spath_):
    global generator
    generators = {
        'clip': CLIPEmbedding,
        'resnet101': ResNetEmbedding,
        'facenet': FaceNetEmbedding
    }
    generator = generators[model_name_](model_name_, dpath_, spath_)
    generator.setup()

def worker(path):
    global generator
    generator.process(path)

# ================
# Argument Parsing
# ================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate image embeddings using various models.")
    parser.add_argument("-m", "--model", type=str, required=True, choices=["clip", "resnet101", "facenet"],
                        help="Model to use for generating embeddings.")
    parser.add_argument("-b", "--batch_size", type=int, default=20, help="Batch size for multiprocessing.")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Number of multiprocessing workers.")
    #parser.add_argument("--dpath", type=str, required=True, help="Path to dataset (images).")
    #parser.add_argument("--spath", type=str, required=True, help="Path to save embeddings.")
    return parser.parse_args()

# ============
# Entry Point
# ============
if __name__ == "__main__":
    args = parse_args()
    assert args.model in ["clip", "resnet101", "facenet"], f"Incorrect Model: {args.model}"

    # Local
    # dpath = Path(__file__).parent.parent / "dataset"
    # spath = Path(__file__).parent.parent / "dataset_"

    # Slurm
    dpath = Path("/nas-ctm01/datasets/public/BIOMETRICS/Face_Recognition/BalancedFace/race_per_7000_aligned")
    spath = Path("/nas-ctm01/datasets/private/XAIBIO/dataset_")
    if not dpath.exists(): raise ValueError("Dataset path doesn't exist:", dpath)
    if not spath.exists(): raise ValueError("Dataset path doesn't exist:", spath)

    # Update paths
    #dpath = dpath
    spath = spath / "embeddings" / args.model
    spath.mkdir(parents=True, exist_ok=True)
    print(f"Dataset images: {dpath}")
    print(f"Saving embeddings at: {spath}")

    # List all images
    imgs_list = list(dpath.rglob("*.jpg"))

    # # Test without multiprocessing
    # init_worker(args.model, dpath, spath)
    # for path in tqdm(imgs_list):
    #     generator.process(path)

    # Multiprocess Strategy
    set_start_method("spawn", force=True)
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.model, dpath, spath)
    ) as pool:
        for _ in tqdm(pool.imap_unordered(worker, imgs_list, chunksize=args.batch_size), total=len(imgs_list)):
            pass
