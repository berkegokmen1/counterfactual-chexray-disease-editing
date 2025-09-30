import os
import time
import gc
import torch
import numpy as np
import random
import torchvision.transforms as T
import torch.nn.functional as F
import yaml
import shutil


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {args.device}")


def clear_mem(verbose=True):
    res = gc.collect(), torch.cuda.empty_cache()
    if verbose:
        print(f"Cleared memory: {res}")
    return res


def create_experiment_folder(args):
    path = os.path.join(args.experiment.path, args.experiment.prefix + time.ctime().replace(" ", "_").replace(":", "-"))
    os.makedirs(path, exist_ok=True)
    args.out_dir = path
    print(f"Created experiment folder: {path}")
    return path


def create_denormalize():
    mean = [0.5330]  # CheXpert mean
    std = [0.0349]  # CheXpert std
    denorm = T.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])

    return denorm


def create_normalize():
    mean = [0.5330]  # CheXpert mean
    std = [0.0349]  # CheXpert std
    t_norm = T.Normalize(mean=mean, std=std)
    return t_norm


def clip_normalize():
    t = T.Compose([T.Resize((512, 512)), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return t


def create_blur(kernel_size=5, sigma=(2.0, 2.0)):
    return T.GaussianBlur(kernel_size, sigma)


def create_mask_dilate():
    blur = create_blur()

    dilate = lambda mask, kernel_size: blur(
        F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
    ).view(mask.shape)

    return dilate


def save_config_yaml(args):
    shutil.copyfile(args.config_path, os.path.join(args.out_dir, "config.yaml"))


def save_txt(
    path,
    name,
    text,
):
    if not isinstance(text, list):
        text = [text]

    with open(os.path.join(path, f"{name}.txt"), "w") as f:
        f.write("\n".join(text))
