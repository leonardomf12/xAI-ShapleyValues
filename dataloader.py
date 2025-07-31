from rich.traceback import install
install()

import os
from abc import abstractclassmethod

import IPython
import pandas as pd
import time
import pickle
from pexpect import pxssh
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import subprocess
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

import paramiko
from scp import SCPClient
from io import BytesIO
from dotenv import load_dotenv


class MyDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size, **kwargs):
        super().__init__(dataset=dataset,
                         shuffle=False,
                         batch_size = batch_size)

class Data2MTL(Dataset):
    def __init__(self, state = "train", **kwargs):
        self.fraction = kwargs.get("fraction", 1)

        # Logging
        self.logger = kwargs.get('file_logger', None)
        self.device = kwargs['device']

        # Load Functions
        self.mode = kwargs["mode"]
        model_emb = None if kwargs['mode'] == 'images' else kwargs['model_emb']
        self.dataset_path = self.get_dataset_path(mode = self.mode, model_emb=model_emb)
        if self.logger is not None:
            self.logger(f"Dataset detected at: {self.dataset_path} -> subset: {state}")
        self.load_path_fn = self.load_path_npy if kwargs['mode'] == "embeddings" else self.load_path_jpg

        self.df_x = None
        self.df_y = None


        self.df = self.load_data()
        if self.mode == "embeddings":
            self.df.path = self.df.path.str.replace('.jpg', '.npy', regex=False) # TODO remove this from here
        self.labels = self.compute_labels()
        self.set_state(state)


        self.transform = transforms.Compose([transforms.ToTensor()]) # Converts PIL image to torch tensor


        # TODO In case the Dataset in not locally -> SHH into Slurm -> Read images through BytesIO
        #self.loader = SlurmLoader()

    def check_missing_files(self):
        #path_list = self.df.path if self.mode == "images" else self.df.path.apply(lambda p: p[:-4])
        #path_list = path_list.apply(Path).apply(lambda p: self.dataset_path / p)

        path_list = self.df.path.apply(
            lambda p: self.dataset_path / Path(p if self.mode == "images" else p[:-4] + ".npy")
        )

        missing_paths = []
        for path in tqdm(path_list, total=len(path_list)):
            if not path.exists():
                missing_paths.append(path)

        # TODO Finish this logic

    def __len__(self):
        return len(self.df_x)

    def len_subset(self, subset):
        assert subset in ["train", "valid", "test"], f"Invalid state: {subset}"
        return len(self.df.loc[self.df.subset == subset])



    def set_state(self, state):
        assert state in ["train", "valid", "test"], f"Invalid state: {state}"
        filter_state = self.df.subset == state
        self.df_x = self.df[filter_state].path.values
        self.df_y = torch.tensor(self.labels[filter_state].astype('float16').values) # .astype('int8')
        self.state = state

        frac_size = int(len(self.df_x) * self.fraction)
        self.df_x = self.df_x[:frac_size]
        self.df_y = self.df_y[:frac_size]

    @staticmethod
    def load_data():
        dir_dataset = "annotations/maad_balanceface_labels/pre_compute_maad_labels.pkl"
        try:
            with open(dir_dataset, "rb") as file:
                df_dict = pickle.load(file)
        except FileNotFoundError:
            raise KeyError(
                f"File: {dir_dataset} not found. Try executing annotations/post_processing.py to generate it.")

        return df_dict

    @staticmethod
    def get_dataset_path(mode = "images", model_emb = None):
        if mode == 'embeddings':
            assert model_emb is not None, "Must provide model_emb"
            mode_path = mode + '/' + model_emb
        elif mode == 'images':
            assert model_emb is None
            mode_path = mode
        else:
            raise ValueError(f"Invalid mode: {mode}")

        print("\nLooking for dataset ...")
        possible_paths = [
            Path(__file__).parent / "dataset",
            Path("/nas-ctm01/datasets/public/BIOMETRICS/Face_Recognition/BalancedFace/race_per_7000_aligned"),
            Path.home() / "dataset",
            # Add more your specific paths if it cannot be found!
        ]
        required_subdirs = ["African", "Asian", "Caucasian", "Indian"]

        for path in possible_paths:
            if all((path / mode_path / subdir).is_dir() for subdir in required_subdirs):
                print(f"Dataset found: {path / mode_path}\n")
                return path / mode_path

        raise ValueError("Dataset not found! Add the dataset location to possible_paths list.")

    @staticmethod
    def load_path_npy(path):
        try:
            return torch.from_numpy(np.load(path))
        except FileNotFoundError: # TODO SSH dataset
            raise FileNotFoundError(f"File not found: {path}")
            # try:
            #     return torch.from_numpy(np.load(self.loader[path]))
            # except FileNotFoundError as e:
            #     raise FileNotFoundError(f"Error Loading this file: {path}\n{e}")

    def load_path_jpg(self, path):
        try:
            return self.transform(Image.open(path))
        except FileNotFoundError: # TODO SSH dataset
            raise FileNotFoundError(f"File not found: {path}")

    def compute_labels(self):
        """
        This function maps the initial pre processed .pkl file into an array of:
        1: True
        0: False
        -1: unknown
        """

        df_labels = pd.DataFrame()
        #df_labels["subset"] = self.df.subset

        # Sex Head
        df_labels["male"] = self.df["male"]
        df_labels["female"] = self.df["male"].replace({1: 0, 0: 1})

        # Race Head -> .astype(pd.Int32Dtype())
        df_labels["African"] = (self.df.race == "African")
        df_labels["Asian"] = (self.df.race == "Asian")
        df_labels["Caucasian"] = (self.df.race == "Caucasian")
        df_labels["Indian"] = (self.df.race == "Indian")

        # Face Head
        df_labels["square_face"] = self.df["square_face"] # Not adding more face shapes because % unk is not negligible

        # Eyeglasses Head
        df_labels["eyeglasses"] = self.df["eyeglasses"]

        # PointyNose Head
        df_labels["pointy_nose"] = self.df["pointy_nose"] # Not adding more nose shapes because % unk is not negligible

        return df_labels


    def old_compute_labels(self):
        heads = {
            "sex": ["male", "~male"],
            "race": ["Asian", "African", "Caucasian", "Indian"],
            "age": ["young", "middle_aged", "senior"],
            "hair_color": ["bald", "black_hair", "blond_hair", "brown_hair", "gray_hair"]
        }

        mapping_func = lambda x: {True: 1, False: 0, None: -1}[x]

        dict_labels = {}
        for subset in self.df_dict.keys():
            dict_labels[subset] = {}

            # Sex Head
            out_sex = self.df_dict[subset][["male"]].copy()
            out_sex.loc[:, "female"] = out_sex.map(lambda x: not x if isinstance(x, bool) else x)
            out_sex = out_sex.applymap(mapping_func)
            dict_labels[subset]["head_sex"] = torch.Tensor(out_sex.values)
            del out_sex

            # Race Head
            out_race = self.df_dict[subset][["race"]].copy()
            out_race = pd.get_dummies(out_race["race"])
            out_race = out_race.applymap(mapping_func)
            dict_labels[subset]["head_race"] = torch.Tensor(out_race.values)
            del out_race

            # Age head
            out_age = self.df_dict[subset][heads["age"]].copy()
            out_age = out_age.applymap(mapping_func)
            dict_labels[subset]["head_age"] = torch.Tensor(out_age.values)
            del out_age

            # Hair Color head
            out_hair = self.df_dict[subset][heads["hair_color"]].copy()
            out_hair = out_hair.applymap(mapping_func)
            dict_labels[subset]["head_color"] = torch.Tensor(out_hair.values)
            del out_hair

        return dict_labels



    def __getitem__(self, idx):
        return self.load_path_fn(self.dataset_path / Path(self.df_x[idx])).to(self.device), self.df_y[idx].to(self.device)

    def get_xy(self, size = None, i = 0):

        x_list = []
        y_list = []

        num_samples = len(self.df_x) if size is None else size
        for idx in tqdm(range(i, i+num_samples), total=num_samples):
            x, y = self.__getitem__(idx)
            x_list.append(x)
            y_list.append(y)

        x = torch.stack(x_list)
        y = torch.stack(y_list)
        return x, y

class SlurmLoader:
    def __init__(self):
        load_dotenv()

        self.hostname = os.environ["slurm_inesc"]
        self.username = os.environ["slurm_username"]
        self.password = os.environ["slurm_password"]

        self.ssh_session = self._connect()

    def _connect(self):
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Automatically trust unknown hosts
        ssh_client.connect(
            hostname=self.hostname,
            username=self.username,
            password=self.password
        )
        return ssh_client


    def __getitem__(self, item):
        file_buffer = BytesIO()
        self.ssh_session.open_sftp().getfo(item, file_buffer)
        file_buffer.seek(0)
        return file_buffer

#class WandbLogger:

if __name__ == "__main__":
    #dataset = Data2MTL(mode="embeddings", model_emb="clip")
    dataset = Data2MTL(mode="images")
    dataset.check_missing_files()

