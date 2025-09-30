import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import random_split
import pandas as pd
from chexpert.dataset import ChexpertSmall
import torchvision.transforms as T


class ChestXRayDataset(Dataset):
    def __init__(self, opts, data_dir, mode="train", transform=None):  # mode="train" or "valid"
        self.opts = opts
        self.mode = mode
        self.target = opts.finetune.target
        self.data_dir = data_dir
        self.transform = transform
        self.csv_path = os.path.join(data_dir, f"{mode}.csv")

        # Read CSV
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df[ChexpertSmall.attr_names + ["Path"]]
        self.df = self.df.replace(-1, 0)
        self.df = self.df.fillna(0)

        # Filter out lateral images
        self.df = self.df[~self.df["Path"].str.contains("lateral", case=False, na=False)]

        self.diseased_image_paths = self.df[self.df[self.target] == 1]["Path"].tolist()
        self.healthy_image_paths = self.df[self.df[ChexpertSmall.attr_names].eq(0).all(axis=1)]["Path"].tolist()

        min_images = min(len(self.diseased_image_paths), len(self.healthy_image_paths))
        print(
            f"Min images: {min_images}, Diseased: {len(self.diseased_image_paths)}, Healthy: {len(self.healthy_image_paths)}"
        )

        self.diseased_image_paths = self.diseased_image_paths[:min_images]
        self.healthy_image_paths = self.healthy_image_paths[:min_images]

    def __len__(self):
        return len(self.diseased_image_paths)

    def __getitem__(self, idxs):
        return self.__fetch_one(idxs)

    def __fetch_one(self, idx):
        diseased_path = self.diseased_image_paths[idx]
        healthy_path = self.healthy_image_paths[idx]

        diseased_path = os.path.join(self.data_dir, diseased_path.replace(self.opts.data.prefix, ""))
        healthy_path = os.path.join(self.data_dir, healthy_path.replace(self.opts.data.prefix, ""))

        diseased_image = Image.open(diseased_path).convert("RGB")
        healthy_image = Image.open(healthy_path).convert("RGB")

        if self.transform:
            diseased_image = self.transform(diseased_image)
            healthy_image = self.transform(healthy_image)

        return diseased_image, healthy_image


class ChestXRayDataModule(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.transform = T.Compose(
            [
                T.Resize(self.opts.data.resize) if self.opts.data.resize else T.Lambda(lambda x: x),
                T.CenterCrop(320 if not self.opts.data.resize else self.opts.data.resize),
                T.ToTensor(),
            ]
        )  # expand to 3 channels

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_train_dataset(self):
        ds = ChestXRayDataset(self.opts, self.opts.data.path, mode="train", transform=self.transform)
        print(f"Train dataset size: {len(ds)}")
        return ds

    def get_val_dataset(self):
        return self.get_test_dataset()

    def get_test_dataset(self):
        ds = ChestXRayDataset(self.opts, self.opts.data.path, mode="valid", transform=self.transform)
        print(f"Test dataset size: {len(ds)}")
        return ds

    def setup(self, stage=None):
        self.train_dataset = self.get_train_dataset()
        self.val_dataset = self.get_val_dataset()
        self.test_dataset = self.get_test_dataset()

    def train_dataloader(self):
        if self.train_dataset is None:
            self.train_dataset = self.get_train_dataset()

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.opts.data.train_batch_size,
            shuffle=False,
            num_workers=self.opts.data.num_workers,
            pin_memory=True,
            prefetch_factor=(
                self.opts.data.prefetch_factor
                if self.opts.data.prefetch_factor and self.opts.data.num_workers > 0
                else None
            ),
        )

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        if self.test_dataset is None:
            self.test_dataset = self.get_test_dataset()

        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.opts.data.test_batch_size,
            shuffle=False,
            num_workers=self.opts.data.num_workers,
            pin_memory=True,
            prefetch_factor=(
                self.opts.data.prefetch_factor
                if self.opts.data.prefetch_factor and self.opts.data.num_workers > 0
                else None
            ),
        )


if __name__ == "__main__":

    from omegaconf import OmegaConf

    opts = OmegaConf.load(
        "/cluster/home/agoekmen/projects/chexray-diffusion/removal-editing/src/config/finetune_config.yaml"
    )
    data_module = ChestXRayDataModule(opts)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    for a, b in train_loader:
        print(a.shape, b.shape)
        break
