import sys
import argparse
import os
import torch
from tqdm import tqdm
import torchxrayvision as xrv

# Suppress weird tensorflow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppresses INFO, WARNING, and ERROR logs

import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel("ERROR")


from typing import Mapping
import torch
import os
import shutil
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from typing import Any
from chexpert.dataset import ChexpertSmall
from cheff.ldm.models.diffusion.ddim import DDIMSampler
from torchvision.utils import make_grid
import wandb
import torchvision
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from easydict import EasyDict as easydict

import chexzero.chexzero.clip as clip

from src.data import ChestXRayDataModule

from src.loss import CLIPLoss
from src.loss import CLIPDirectionLoss
from src.utils import create_normalize as chexpert_normalize
from src.utils import create_denormalize as chexpert_denormalize
from src.utils import clip_normalize
from src.utils import create_blur
from src.utils import create_mask_dilate
from src.utils import set_device


from models.classifier import define_classifier_model
from models.cheff import get_cheff_ldm, clone_model_for_sample


class DiseaseEditingDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        opts,
        clip_model,
        clip_preprocess,
        classifier_model,
        cheff_ldm,
        cheff_params: Mapping[str, Any],
        segmentor,
        mode: str = "removal",
    ):
        super().__init__()

        self.opts = opts
        self.target = opts.finetune.target
        self.target_index = ChexpertSmall.attr_names.index(opts.finetune.target)

        self.lambdas = opts.finetune.lambdas
        # default weight for anchor loss if the config does not provide one
        if not hasattr(self.lambdas, "anchor_step"):
            self.lambdas.anchor_step = 0.05

        self.clip_model = clip_model.to(self.device)
        self.clip_preprocess = clip_preprocess
        self.classifier_model = classifier_model.to(self.device)

        self.original_cheff_ldm = cheff_ldm
        self.original_cheff_params = cheff_params
        self.original_ldm = cheff_ldm.model.to(self.device)

        self.cheff_ldm = cheff_ldm
        self.cheff_params = cheff_params
        self.ldm = cheff_ldm.model.to(self.device)

        self.segmentor = segmentor.to(self.device)
        self.mode = mode

        if self.mode == "removal":
            pos_prompt, neg_prompt = f"No {self.target}", self.target
        else:  # "sample"
            pos_prompt, neg_prompt = self.target, f"No {self.target}"

        self.clip_loss = CLIPLoss(
            opts,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,  # type: ignore
        ).to(self.device)

        self.clip_direction_loss = CLIPDirectionLoss(
            opts,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,  # type: ignore
        ).to(self.device)

        self._freeze(self.clip_model)

        self.chexpert_normalize = chexpert_normalize()
        self.chexpert_denormalize = chexpert_denormalize()
        self.clip_normalize = clip_normalize()

        self.blur = create_blur()
        self.mask_dilate = create_mask_dilate()

    def _plot_loss_history(self, batch_idx: int):
        """
        Save a line plot of every loss component plus the total loss
        accumulated during _denoising_pass for the current sample.
        """
        if not hasattr(self, "_loss_history") or len(self._loss_history) == 0:
            return

        steps = [r["global_step"] for r in self._loss_history]
        keys = [k for k in self._loss_history[0] if k != "global_step"]

        for k in keys:
            plt.plot(steps, [r[k] for r in self._loss_history], label=k)

        plt.xlabel("global step")
        plt.ylabel("loss value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(self.opts.finetune.log_dir, f"sample_{batch_idx}_loss.jpg")
        plt.savefig(out_path)
        plt.clf()

    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

        model.eval()

    def _unfreeze(self, model):
        for param in model.parameters():
            param.requires_grad = True

        model.train()

    def _get_classification(self, x, normalize=False):
        if normalize:
            x = self.chexpert_normalize(x)

        return self.classifier_model(x)  # [B=1, C]

    def _decode_samples(self, x):
        decoded_samples = self.ldm.decode_first_stage(x)
        decoded_samples.clamp_(-1, 1)
        decoded_samples = (decoded_samples + 1) / 2  # normalize this and use

        return decoded_samples

    def _calculate_clip_loss(self, x):
        # CLIPLoss Does internal transformation
        loss, prob = self.clip_loss(x)
        return loss

    def _calculate_clip_direction_loss(self, src, target):
        direction_loss, distance_loss = self.clip_direction_loss(src, target)
        return -torch.log((2 - direction_loss) / 2), 1 - distance_loss

    @torch.no_grad()
    def _calculate_clip_probs(self, x):
        _, probs = self.clip_loss(x)
        pos_prob = probs[:, 0]  # Positive prompt is first
        neg_prob = probs[:, 1]  # Negative prompt is second

        # Take mean
        pos_prob = pos_prob.mean()
        neg_prob = neg_prob.mean()

        return {f"{self.clip_loss.pos_prompt}": pos_prob, f"{self.clip_loss.neg_prompt}": neg_prob}

    def _soft_kl_loss(self, x, logits=None, target_idx=0, T=1.5):
        if logits is None:
            logits = self._get_classification(x, normalize=True)

        # softmax with temperature
        p = F.softmax(logits / T, dim=1)
        q = p.clone().detach()

        if self.mode == "synthesize":
            q[:, target_idx] = q.max(dim=1).values  # push prob up
        else:
            q[:, target_idx] = q.min(dim=1).values  # push prob down
        q = q / q.sum(dim=1, keepdim=True)

        return F.kl_div(F.log_softmax(logits / T, dim=1), q, reduction="batchmean") * (T**2)

    def _calculate_targeted_classification_loss(self, x, target_index):

        logits = self._get_classification(x, normalize=True)
        # Generate the target vector with the same shape as logits
        target = logits.clone()

        # Set the target index to 1 if synthesizing disease, 0 otherwise
        target[:, target_index] = -1e5 if self.mode == "removal" else 1e5

        # Apply binary cross-entropy with logits only to the target class index
        # loss = F.binary_cross_entropy_with_logits(logits[:, target_index], target[:, target_index])
        loss = F.cross_entropy(  # TODO check cross entropy
            logits,
            F.softmax(target, dim=-1),
        )

        return loss

    def _denoising_pass(self, src_latent, batch_idx, x_t=None, bone_mask=None):
        x_t_orig = x_t.clone()

        num_timesteps = self.opts.finetune.num_timesteps
        ddim = DDIMSampler(self.ldm)
        ddim.make_schedule(num_timesteps, ddim_eta=1, verbose=False)
        timesteps = ddim.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        self._loss_history = []  # reset recorder for this sample

        if x_t is None:
            x_t = torch.randn_like(src_latent)

        src_denoised = self._decode_samples(src_latent)
        self._anchor_logits = self._get_classification(src_denoised, normalize=True)

        print(f"{batch_idx} - Original probs:", self._calculate_clip_probs(src_denoised))
        epoch_bar = tqdm(range(self.opts.finetune.num_epochs), desc=f"Sample {batch_idx} – epochs", leave=True)

        for ep in epoch_bar:
            if x_t_orig is not None:
                x_t = x_t_orig.clone().detach().requires_grad_(True)

            step_bar = tqdm(enumerate(time_range), total=total_steps, desc=f"Ep {ep}", leave=False)
            for i, step in step_bar:
                self.optimizer.zero_grad()

                index = total_steps - i - 1
                ts = torch.full((src_latent.shape[0],), step, device=self.device, dtype=torch.long)

                if bone_mask is not None:
                    img_orig = self.ldm.q_sample(src_latent, ts)
                    x_t = img_orig * bone_mask + (1.0 - bone_mask) * x_t.clone()

                c = 1 - bone_mask

                x_t, x_0 = ddim.prev_step(
                    x=x_t,
                    c=c,
                    t=ts,
                    index=index,
                    return_x0=True,
                )

                dest_denoised = self._decode_samples(x_0)
                dest_denoised.requires_grad_(True)

                loss = torch.tensor(0.0).to(self.device)

                dir_loss, dis_loss = self._calculate_clip_direction_loss(src_denoised, dest_denoised)
                clip_direction_loss = dir_loss * self.lambdas.clip_direction_step
                clip_distance_loss = dis_loss * self.lambdas.clip_distance_step
                clip_loss = self._calculate_clip_loss(dest_denoised) * self.lambdas.clip_step
                l1_loss = torch.nn.L1Loss()(src_latent, x_0) * self.lambdas.l1_step

                logits_current = self._get_classification(dest_denoised, normalize=True)

                # 1) target push (soft‑KL)
                classification_target_loss = (
                    self._soft_kl_loss(dest_denoised, logits_current, self.target_index)
                    * self.lambdas.classification_step
                )

                # 2) anchor loss: keep non‑target logits close to the original ones
                others_mask = torch.arange(logits_current.shape[1], device=self.device) != self.target_index
                anchor_loss = (
                    F.mse_loss(logits_current[:, others_mask], self._anchor_logits[:, others_mask])
                    * self.lambdas.anchor_step
                )

                classification_loss = classification_target_loss + anchor_loss

                loss = clip_direction_loss + l1_loss + classification_loss + clip_distance_loss + clip_loss
                loss.backward(retain_graph=True)
                self.optimizer.step()

                # print(f"{batch_idx} - {ep} - {i} - ", "Clip Direction Step Loss:", clip_direction_loss.item(), "Clip Step Loss:", clip_loss.item(), "Classification Step Loss:", classification_loss.item(), "L1 Step Loss:", l1_loss.item(), "Distance Loss:", clip_distance_loss.item(), "Loss:", loss.item())

                self._loss_history.append(
                    {
                        "global_step": ep * total_steps + i,
                        "clip_direction_loss": clip_direction_loss.item(),
                        "clip_loss": clip_loss.item(),
                        "classification_loss": classification_loss.item(),
                        "anchor_loss": anchor_loss.item(),
                        "l1_loss": l1_loss.item(),
                        "clip_distance_loss": clip_distance_loss.item(),
                        "total_loss": loss.item(),
                    }
                )

                step_bar.set_postfix(
                    CD=clip_direction_loss.item(),
                    CLIP=clip_loss.item(),
                    CLS=classification_loss.item(),
                    L1=l1_loss.item(),
                    DIS=clip_distance_loss.item(),
                    ANC=anchor_loss.item(),
                    TOT=loss.item(),
                )

                x_t = x_t.detach()

            dest_denoised = self._decode_samples(x_t)
            torchvision.utils.save_image(
                dest_denoised,
                os.path.join(self.opts.finetune.log_dir, f"sample_{batch_idx}__epoch_{ep}__denoising.jpg"),
            )

        dest_denoised = self._decode_samples(x_t)

        print(f"{batch_idx} - Edited probs:", self._calculate_clip_probs(dest_denoised))
        print(" ")

        # save images
        self._plot_loss_history(batch_idx)
        grid1 = make_grid(src_denoised)
        grid2 = make_grid(dest_denoised)
        image = torch.cat([grid1, grid2], dim=1)
        torchvision.utils.save_image(
            image, os.path.join(self.opts.finetune.log_dir, f"sample_{batch_idx}_final_denoising.jpg")
        )

    def _get_bone_mask(self, x: torch.Tensor):
        x = x.squeeze(0).cpu().clone().detach().numpy()

        x = ((x * 2) - 1.0) * 1024

        transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512, engine="cv2")]
        )

        x = transform(x)

        x = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.segmentor(x)

        pred = 1 / (1 + torch.exp(-pred))  # sigmoid
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1

        mask = (
            pred[:, 0]
            + pred[:, 1]
            + pred[:, 2]
            + pred[:, 3]
            # + pred[:, 6]
            # + pred[:, 7]
            + pred[:, 9]
            + pred[:, 11]
            + pred[:, 12]
            + pred[:, 13]
        )

        mask[mask > 0] = 1
        mask[mask <= 0] = 0

        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64))])
        mask = transform(mask.unsqueeze(0))

        return mask

    def forward(self, batch, batch_idx):
        self.cheff_ldm, self.cheff_params = clone_model_for_sample(self.original_cheff_ldm, self.opts)
        self.ldm = self.cheff_ldm.model.to(self.device)
        self.define_optimizers()
        print("Cloned model")

        x_diseased, x_neutral = batch
        x_diseased, x_neutral = x_diseased.to(self.device).float(), x_neutral.to(self.device).float()

        x = x_diseased if self.mode == "removal" else x_neutral

        bone_mask = self._get_bone_mask(x)

        torchvision.utils.save_image(x, os.path.join(self.opts.finetune.log_dir, f"sample_{batch_idx}_unedited.jpg"))
        torchvision.utils.save_image(
            bone_mask, os.path.join(self.opts.finetune.log_dir, f"sample_{batch_idx}_bone_mask.jpg")
        )

        x_latent = self.ldm.encode_first_stage(x * 2 - 1).mode()

        num_timesteps = self.opts.finetune.num_timesteps
        ddim = DDIMSampler(self.ldm)
        ddim.make_schedule(num_timesteps, ddim_eta=1, verbose=False)
        timesteps = ddim.ddim_timesteps

        x_t = x_latent.clone()

        with torch.no_grad():
            c = 1 - bone_mask

            for i, step in enumerate(timesteps):
                index = i
                ts = torch.full((x_t.shape[0],), step, device=self.device, dtype=torch.long)

                x_t = ddim.next_step(x_t, c, ts, index)

        self._denoising_pass(x_latent, batch_idx, x_t=x_t, bone_mask=bone_mask)

    def define_optimizers(self):
        self.optimizer = torch.optim.Adam(self.cheff_params, lr=self.opts.finetune.learning_rate)


def main(opts, opts_path, mode):
    os.makedirs(opts.finetune.log_dir, exist_ok=True)
    seed_everything(42)
    set_device(opts)

    opts.out_dir = opts.finetune.log_dir

    shutil.copyfile(opts_path, os.path.join(opts.finetune.log_dir, "config.yaml"))
    shutil.copyfile(__file__, os.path.join(opts.finetune.log_dir, "finetune_sample.py"))

    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    clip_model.load_state_dict(torch.load(opts.model.clip.path, map_location="cpu"))
    clip_model.eval()

    classifier_model = define_classifier_model(opts)

    cheff_ldm = get_cheff_ldm(opts, load_checkpoint=True, load_base_uncond=True, load_image_mask_cond=False)
    cheff_ldm, cheff_params = clone_model_for_sample(cheff_ldm, opts)
    segmentor = xrv.baseline_models.chestx_det.PSPNet()

    datamodule = ChestXRayDataModule(opts)

    model = DiseaseEditingDiffusionModel(
        opts=opts,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        classifier_model=classifier_model,
        cheff_ldm=cheff_ldm,
        cheff_params=cheff_params,
        segmentor=segmentor,
        mode=mode,
    ).to(opts.device)

    train_dataloader = datamodule.train_dataloader()

    for batch_idx, batch in enumerate(train_dataloader):
        model(batch, batch_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sample Generation")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file",
        default="./config.yaml",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Target disease",
        default="Pleural Effusion",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode to run the script in",
        choices=["synthesize", "removal"],
        default="removal",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Experiment name",
        default="experiment",
    )
    args = parser.parse_args()

    opts = OmegaConf.load(args.config)

    # Replace null values
    opts.finetune.target = args.target
    opts.finetune.experiment_name = f"{args.experiment_name}_{args.target.replace(' ', '-')}_{args.mode}"

    main(easydict(opts), args.config, args.mode)
