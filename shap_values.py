import time
from torch import nn

from skimage.segmentation import slic
import matplotlib.pyplot as plt
from typing import Dict
import math
from pydantic.experimental.pipeline import transform

from architectures import FACENETModel, iResNet100Model, CLIPModel

import torch
import IPython
import shap
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# My scripts
from dataloader import MyDataLoader, Data2MTL
import architectures as model_arch_scr

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
        # TODO Make the batch size an input parameter
        background_dataloader = MyDataLoader(
            dataset=Data2MTL(
                state="train",
                mode=self.mode,
                model_emb=self.model_emb,
                device=self.device
            ),
            batch_size=200,
        )
        self.bg_data, _ = next(iter(background_dataloader))

        img_dataloader = MyDataLoader(
            dataset=Data2MTL(
                state="test",
                mode=self.mode,
                model_emb=self.model_emb,
                device=self.device
            ),
            batch_size=1,
        )
        self.img, _ = img_dataloader.dataset[200]
        self.img = torch.unsqueeze(self.img, dim=0)

class KernelShapImage(MySHAP):
    def __init__(self, weights_path, **kwargs):
        super().__init__(weights_path, **kwargs)
        self.bg_size = kwargs.get("bg_size", 1000)
        self.num_segm = kwargs.get("num_segm", 50)
        self.bg_state = True

        # SLIC mask
        img_ = np.squeeze(self.img.cpu().detach().numpy().transpose(0, 2, 3, 1))
        self.segment_mask = slic(img_, n_segments=self.num_segm, sigma=5, start_label=0)
        self.num_segm = len(np.unique(self.segment_mask))
        print(f"n_segments in practice = {len(np.unique(self.segment_mask))}")
        #self.slic_masking(img_, masks=np.array([[1, 0, 1, 1]]))


        # Compute background
        #bg_mask = np.ones((self.bg_size, self.num_segm))
        bg_mask = np.random.randint(0, 2, size=(self.bg_size, self.num_segm), dtype=np.int8)
        zero_rows = np.where(np.all(bg_mask == 0, axis=1))[0]
        print(f"Rows with all zeros in bg_mask: {zero_rows}")
        self.explainer = shap.KernelExplainer(self.predict_fn, bg_mask)



    def predict_fn(self, mask):
        print(f"predict_fn called with masks shape: {mask.shape}")
        if self.bg_state: # Background data
            x = self.bg_data
        else: # Img to calculate SHAP values on
            x = self.img
        print(x.shape)
        x = self.slic_masking(x, mask)
        print(x.shape)
        with torch.inference_mode():
            output = self.model.predict(x)
        return self.postprocess(output)

    @staticmethod
    def postprocess(x_out_dict: Dict[str, torch.Tensor]):
        x_head_list = []
        for x_out in x_out_dict.values():
            x_head_list.append(x_out.cpu().detach().numpy())

        x_out = np.concatenate(x_head_list, axis=1)
        return x_out

    def slic_masking(self, imgs: torch.Tensor, masks):
        """
        Applies SLIC superpixels to the images according to the masks' map.
        The mask map is created by the original image that we want to evaluate SHAP values on.

        :param imgs: Expects a 4D tensor (B, C, H, W) = (B, 3, 112, 112)
        :param masks: Expects a 3D tensor (B, H, W) = (B, 112, 112)
        :return: Returns a 4D tensor (B, C, H, W) = (B, 3, 112, 112)
        """
        print(f"slic_masking: {imgs.shape}")
        imgs_masked_list = []
        for img, mask in zip(imgs, masks):
            img_segm = img.cpu().detach().numpy().transpose(1, 2, 0).copy()

            for slic_idx in np.where(mask == 0)[0]:
                mean_value = np.mean(img_segm[self.segment_mask == slic_idx], axis=0)
                img_segm[self.segment_mask == slic_idx] = np.expand_dims(mean_value, axis=0)

            img_ = torch.unsqueeze(torch.from_numpy(img_segm.transpose(2, 0, 1)), dim=0) # (112, 112, 3) -> (3, 112, 112)
            imgs_masked_list.append(img_)

        imgs_masked = torch.cat(imgs_masked_list, dim=0).to(self.device)

        # Plot the image
        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #ax1.imshow(imgs[0])
        #ax2.imshow(imgs_masked[0])
        #plt.show()

        return imgs_masked

    def calculate_shap(self):
        self.bg_state = False
        return self.explainer.shap_values(np.ones((1, self.num_segm)))

class DeepExplainerSHAP(MySHAP):
    def __init__(self, weights_path, **kwargs):
        super().__init__(weights_path, **kwargs)
        self._disable_inplace_relu(self.model.feat_extractor)

        #IPython.embed()
        self.explainer = shap.DeepExplainer(self.model, self.bg_data)

    def calculate_shap(self):
        return self.explainer.shap_values(self.img)

    def _disable_inplace_relu(self, module):
        for child in module.children():
            self._disable_inplace_relu(child)
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

class GradientExplainerSHAP(MySHAP):
    def __init__(self, weights_path, **kwargs):
        super().__init__(weights_path, **kwargs)
        self.shap_values = None

        self.explainer = shap.GradientExplainer(self.model, self.bg_data)

    def calculate_shap(self):
        self.shap_values = np.squeeze(self.explainer.shap_values(self.img, nsamples=500))

        self.plot_shap()

    def plot_shap2(self, index=0):
        shap_vals = self.shap_values  # shape (1, 3, 112, 112, 9)

        # Squeeze batch dim if present
        if shap_vals.shape[0] == 1:
            shap_vals = shap_vals[0]  # now shape (3, 112, 112, 9)

        # Select the index-th output map (0 <= index < 9)
        shap_slice = shap_vals[..., index]  # shape (3, 112, 112)

        n_channels = shap_slice.shape[0]
        fig, axs = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))
        if n_channels == 1:
            axs = [axs]

        for i in range(n_channels):
            data = shap_slice[i]
            im = axs[i].imshow(data, cmap='bwr',
                               vmin=-np.max(np.abs(data)),
                               vmax=np.max(np.abs(data)))
            axs[i].set_title(f"Channel {i}, output {index}")
            axs[i].axis('off')
            fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

        plt.suptitle(f"SHAP values for output index {index}")
        plt.show()

    def plot_shap(self):

        #shap.summary_plot(self.shap_values, self.bg_data, plot_type="bar")
        num_subplots = self.shap_values.shape[-1] + 1
        fig, axes = plt.subplots(2, 5, figsize=(20, 15))
        axes = axes.flatten()
        axes[0].imshow(self.img.cpu().detach().numpy()[0][0])
        for i in range(1, num_subplots):
            axes[i].imshow(self.shap_values[0, ..., i - 1])
        plt.show()

if __name__ == "__main__":
    # mode = "images"
    # model_emb = "facenet"
    # FFN = True
    # device = "cpu"
    weights = "models/iResNet100Model-False-p[0.3]/iResNet100Model-False-p[0.3]__2025-05-23_14-22__e1.pth"
    #
    # # Image
    # img = "dataset/images/African/m.0b0g5k/10-FaceId-0_align.jpg"
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    # img_ = transform(Image.open(img))
    #
    # plt.imshow(img_.permute(1, 2, 0).numpy())
    # plt.show()
    #
    # # DataLoader
    # dataloader = Data2MTL(
    #     mode=mode,
    #     model_emb=model_emb,
    #     FFN=FFN,
    #     device=device
    # )
    #
    # x_train, y_train = dataloader.get_xy(size=1000)
    # x_train = x_train.cpu().detach().numpy() # Mandatory for Kernel Shap input to be numpy
    # x_train = np.reshape(x_train, (x_train.shape[0], -1))
    # print(f"Background Data Shape{x_train.shape}")
    #
    # predict_fn = KernelShap(
    #     weights_path=weights,
    #     mode=mode,
    #     FFN=FFN,
    #     device=device
    # )
    #
    #
    # explainer = shap.KernelExplainer(predict_fn, x_train)
    #
    # img = np.expand_dims(img_.cpu().detach().numpy().flatten(), axis=0)
    # explainer.shap_values(img)

    kernel_shap = GradientExplainerSHAP(weights_path=weights, mode="images")
    shap_values = kernel_shap.calculate_shap()
    print(shap_values.shape)


    #print(shap_values)