import IPython

#from my_scripts.pipeline import TrainingPipeline
from abc import ABC, abstractmethod
from pathlib import Path

from pandas.core.window.doc import kwargs_scipy
from sympy.solvers.diophantine.diophantine import reconstruct
from torchinfo import summary
import torch
from torch import nn
from torch.nn import Softmax
from torch.nn.functional import relu
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import precision_score, recall_score, f1_score

from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
import clip

# Base model
class BaseModel(ABC, nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        self.model_name = kwargs['model_name']
        self.file_logger = kwargs.get('file_logger', None)
        if self.file_logger is not None:
            self.model_dir_path = self.file_logger.model_dir_path
            self.experiment_name = self.file_logger.experiment_name

        self.mode = kwargs.get('mode', None)
        self.ffn = kwargs.get('FFN', None)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in the model subclass.")

    def log_model_info(self):
        self.logger("Model Initialized:")
        self.logger(str({
            "model_name": "FACENETModel",
            "type": self.mode,
            "ffn": self.ffn,
        }))

    def save_weights(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_weights": self.state_dict(),
            "model_name": self.experiment_name,
            "mode": self.mode,
            "FFN": self.ffn
        }
        weights_name = self.experiment_name + f"__e{epoch}.pth"

        path_list = [
            Path("/nas-ctm01/datasets/private/XAIBIO/.checkpoints"),
            Path(__file__).parent / "models",
        ]

        for path in path_list:
            if path.exists():
                save_path = path / self.model_dir_path / weights_name
                save_path.parent.mkdir(exist_ok=True)
                torch.save(checkpoint, save_path)
                self.file_logger(f"💾 Model weights saved: {save_path}")
                return
            else:
                self.file_logger(f"❌ Path to save model weights not found: {path}")

        self.file_logger("❌ There is no directory to save the model weights. Aborting training ...")
        raise ValueError("❌ There is no directory to save the model weights. Aborting training ...")

    def predict(self, x):
        with torch.inference_mode():
            logits = self.forward(x)

            return {
                'sex': nn.Softmax(dim=1)(logits['sex']),            # (B, 2)
                'race': nn.Softmax(dim=1)(logits['race']),          # (B, 4)
                'square': nn.Sigmoid()(logits['square']),           # (B, 1)
                'eyeglasses': nn.Sigmoid()(logits['eyeglasses']),   # (B, 1)
                'nose': nn.Sigmoid()(logits['nose'])                # (B, 1)
            }

    def predict_tensor(self, x):
        # with torch.inference_mode():
        #     logits = self.forward(x)
        #
        #     return torch.cat([
        #         nn.Softmax(dim=1)(logits['sex']),
        #         nn.Softmax(dim=1)(logits['race']),
        #         nn.Sigmoid()(logits['square']),
        #         nn.Sigmoid()(logits['eyeglasses']),
        #         nn.Sigmoid()(logits['nose'])
        #     ], dim=1)

        logits = self.forward(x)

        return torch.cat([
            nn.Softmax(dim=1)(logits['sex']),
            nn.Softmax(dim=1)(logits['race']),
            nn.Sigmoid()(logits['square']),
            nn.Sigmoid()(logits['eyeglasses']),
            nn.Sigmoid()(logits['nose'])
        ], dim=1)

    @staticmethod
    def prepoc_targets(targets):
        y_true = {}

        y_true["sex"] = targets[:, :1 + 1]
        y_true["race"] = targets[:, 2:5 + 1]
        y_true["square"] = targets[:, -3]
        y_true["eyeglasses"] = targets[:, -2]
        y_true["nose"] = targets[:, -1]

        return y_true

    def compute_metrics(self, y_pred, y_true, head_names, prefix):
        y_true = self.prepoc_targets(y_true)
        threshold = 0.5

        metrics = {}
        for w_i, head_name in enumerate(head_names):
            yp = (y_pred[head_name] > threshold).to(dtype=torch.float32)
            yt = y_true[head_name].to(dtype=torch.float32)

            if head_name in head_names[-3:]:  # BCE
                # Masking
                unk_mask = ~(yt == -1)
                yp_ = yp[unk_mask].cpu().numpy()
                yt_ = yt[unk_mask].cpu().numpy()

            else:

                # Masking - One-hot vector label
                unk_mask = ~(yt == -1).any(dim=1)
                yp = yp[unk_mask]
                yt = yt[unk_mask]

                # Converting one-hot to class indices
                yt_ = yt.argmax(dim=1).cpu().numpy()
                yp_ = yp.argmax(dim=1).cpu().numpy()

            # Metrics
            precision = precision_score(yt_, yp_, average='macro')
            recall = recall_score(yt_, yp_, average='macro')
            f1 = f1_score(yt_, yp_, average='macro')

            # Logging to dict
            metrics[f'{prefix}/{head_name}/precision'] = precision
            metrics[f'{prefix}/{head_name}/recall'] = recall
            metrics[f'{prefix}/{head_name}/f1'] = f1

        return metrics

# =====================
# Model Architectures
# =====================


class FACENETModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        FACENET_EMB_SIZE = 512
        p = kwargs.get('p', 0.3)

        self.mode = kwargs['mode']
        self.ffn = kwargs['FFN']

        # Feature extractor
        self.feat_extractor = InceptionResnetV1(pretrained='vggface2', device=kwargs['device']).eval()

        # Layers
        self.ffn_block = TransformerFFNBlock(d_model=FACENET_EMB_SIZE)

        # MTL Heads
        hidden_dim = 128 # race: [male, female]; race: [Asi, Afr, Cau, Ind]; Square_face; eyeglasses; pointy_nose
        self.linear = nn.Linear(FACENET_EMB_SIZE, hidden_dim)

        self.out_sex = nn.Linear(hidden_dim, 2)
        self.out_race = nn.Linear(hidden_dim, 4)
        self.out_square_head = nn.Linear(hidden_dim, 1)
        self.out_eyeglasses = nn.Linear(hidden_dim, 1)
        self.out_pointy_nose = nn.Linear(hidden_dim, 1)

        # Forward pass
        self.forward_blocks = nn.Sequential()
        if self.mode == 'images': self.forward_blocks.add_module("feature_extractor", self.feat_extractor)
        if self.ffn_block: self.forward_blocks.add_module("ffn_block", self.ffn_block)
        self.forward_blocks.add_module("linear", self.linear)
        self.forward_blocks.add_module("batchnorm", nn.BatchNorm1d(hidden_dim))
        self.forward_blocks.add_module("relu", nn.ReLU())
        self.forward_blocks.add_module("dropout", nn.Dropout(p=p))

        self.to(kwargs['device'])
        if self.file_logger is not None:
            self.file_logger(f" ✅ Model Initialized Successfully! - Device: {str(kwargs['device'])}") #:check 💾 ✅ ❌


    def forward(self, x):
        try:
            x = self.forward_blocks(x)
        except RuntimeError:
            x = self.forward_blocks(torch.squeeze(x))

        # Logits
        y_sex = self.out_sex(x)
        y_race = self.out_race(x)
        y_square_head = self.out_square_head(x)
        y_eyeglasses = self.out_eyeglasses(x)
        y_pointy_nose = self.out_pointy_nose(x)

        # return torch.cat([
        #     y_sex,
        #     y_race,
        #     y_square_head,
        #     y_eyeglasses,
        #     y_pointy_nose
        # ], dim=1)

        return {
           'sex': y_sex,
           'race': y_race,
           'square': y_square_head,
           'eyeglasses': y_eyeglasses,
           'nose': y_pointy_nose
        }


class iResNet100Model(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        IRESNET100_EMB_SIZE = 1000
        p = kwargs.get('p', 0.3)

        self.mode = kwargs['mode']
        self.ffn = kwargs['FFN']

        # Feature extractor
        weights = ResNet101_Weights.DEFAULT
        self.feat_extractor = resnet101(weights=weights.IMAGENET1K_V2).to(kwargs['device'])
        self.feat_extractor.eval()

        self._preprocess = weights.transforms()

        # Layers
        self.ffn_block = TransformerFFNBlock(d_model=IRESNET100_EMB_SIZE)

        # MTL Heads
        hidden_dim = 128 # race: [male, female]; race: [Asi, Afr, Cau, Ind]; Square_face; eyeglasses; pointy_nose
        self.linear = nn.Linear(IRESNET100_EMB_SIZE, hidden_dim)

        self.out_sex = nn.Linear(hidden_dim, 2)
        self.out_race = nn.Linear(hidden_dim, 4)
        self.out_square_head = nn.Linear(hidden_dim, 1)
        self.out_eyeglasses = nn.Linear(hidden_dim, 1)
        self.out_pointy_nose = nn.Linear(hidden_dim, 1)

        # Forward pass
        self.forward_blocks = nn.Sequential()
        if self.mode == 'images': self.forward_blocks.add_module("feature_extractor", self.feat_extractor)
        if self.ffn_block: self.forward_blocks.add_module("ffn_block", self.ffn_block)
        self.forward_blocks.add_module("linear", self.linear)
        self.forward_blocks.add_module("batchnorm", nn.BatchNorm1d(hidden_dim))
        self.forward_blocks.add_module("relu", nn.ReLU())
        self.forward_blocks.add_module("dropout", nn.Dropout(p=p))

        self.to(kwargs['device'])

        if self.file_logger is not None:
            self.file_logger(f" ✅ Model Initialized Successfully! - Device: {str(kwargs['device'])}") #:check 💾 ✅ ❌


    def forward(self, x):
        try:
            x = self.forward_blocks(x)
        except RuntimeError:
            x = self.forward_blocks(torch.squeeze(x))

        # Logits
        y_sex = self.out_sex(x)
        y_race = self.out_race(x)
        y_square_head = self.out_square_head(x)
        y_eyeglasses = self.out_eyeglasses(x)
        y_pointy_nose = self.out_pointy_nose(x)

        # return torch.cat([
        #     y_sex,
        #     y_race,
        #     y_square_head,
        #     y_eyeglasses,
        #     y_pointy_nose
        # ], dim=1)
        return {
            'sex': y_sex,
            'race': y_race,
            'square': y_square_head,
            'eyeglasses': y_eyeglasses,
            'nose': y_pointy_nose
        }



class CLIPModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        CLIP_EMB_SIZE = 512

        self.mode = kwargs['mode']
        self.ffn = kwargs['FFN']
        p = kwargs.get('p', 0.3)

        # Feature extractor
        self.clip_model, _ = clip.load("ViT-B/32", device=kwargs['device'])
        for param in self.clip_model.parameters():
            param.requires_grad = True

        self.feat_extractor = CLIPImageEncoder(clip_model=self.clip_model)
        self.feat_extractor.eval()
        self.preprocess = self.preprocess = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.Normalize(mean=[0.4815, 0.4578, 0.4082],
                        std=[0.2686, 0.2613, 0.2758])
        ])


        # Layers
        self.ffn_block = TransformerFFNBlock(d_model=CLIP_EMB_SIZE)

        # MTL Heads
        hidden_dim = 128 # race: [male, female]; race: [Asi, Afr, Cau, Ind]; Square_face; eyeglasses; pointy_nose
        self.linear = nn.Linear(CLIP_EMB_SIZE, hidden_dim)

        self.out_sex = nn.Linear(hidden_dim, 2)
        self.out_race = nn.Linear(hidden_dim, 4)
        self.out_square_head = nn.Linear(hidden_dim, 1)
        self.out_eyeglasses = nn.Linear(hidden_dim, 1)
        self.out_pointy_nose = nn.Linear(hidden_dim, 1)

        # Forward pass
        self.forward_blocks = nn.Sequential()
        if self.mode == 'images': self.forward_blocks.add_module("feature_extractor", self.feat_extractor)
        if self.ffn_block: self.forward_blocks.add_module("ffn_block", self.ffn_block)
        self.forward_blocks.add_module("linear", self.linear)
        self.forward_blocks.add_module("batchnorm", nn.BatchNorm1d(hidden_dim))
        self.forward_blocks.add_module("relu", nn.ReLU())
        self.forward_blocks.add_module("dropout", nn.Dropout(p=p))

        self.to(kwargs['device'])
        if self.file_logger is not None:
            self.file_logger(f" ✅ Model Initialized Successfully! - Device: {str(kwargs['device'])}") #:check 💾 ✅ ❌


    def forward(self, x):
        if self.mode == 'images':
            x = self.preprocess(x)

        try:
            x = self.forward_blocks(x.float())
        except RuntimeError:
            x = self.forward_blocks(torch.squeeze(x.float())) # (b, 1, 512) -> (b, 512); float16 -> float32

        # Logits
        y_sex = self.out_sex(x)
        y_race = self.out_race(x)
        y_square_head = self.out_square_head(x)
        y_eyeglasses = self.out_eyeglasses(x)
        y_pointy_nose = self.out_pointy_nose(x)

        return {
            'sex': y_sex,
            'race': y_race,
            'square': y_square_head,
            'eyeglasses': y_eyeglasses,
            'nose': y_pointy_nose
        }

class CLIPImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x):
        return self.clip_model.encode_image(x)
# =====================
# Model Blocks
# =====================

class TransformerFFNBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # FFN
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.activation = nn.ReLU()  # or use nn.GELU() for a smoother activation
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model*4, d_model)

        # Add & Norm
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # FFN
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout_ffn(x)
        x = self.linear2(x)

        # Add & Norm
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x



if __name__ == '__main__':
    x = torch.randn(4, 3, 112, 112)

    model = FACENETModel()
    y = model.predict(x)

    IPython.embed()
