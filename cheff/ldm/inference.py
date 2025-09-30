"""Functions and classes for loading and handling models conveniently."""

import contextlib
from typing import Union, Dict, Optional

import torch
from torch import Tensor

from cheff.ldm.models.autoencoder import AutoencoderKL
from cheff.ldm.models.diffusion.ddpm import LatentDiffusion
from cheff.ldm.models.diffusion.ddim import DDIMSampler


class CheffAEModel:
    def __init__(self, model_path: str, device: Union[str, int, torch.device] = "cuda") -> None:
        self.device = device

        with contextlib.redirect_stdout(None):
            self.model = AutoencoderKL(
                embed_dim=3,
                ckpt_path=model_path,
                ddconfig={
                    "double_z": True,
                    "z_channels": 3,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": (1, 2, 4),
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                lossconfig={"target": "torch.nn.Identity"},
            )
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        return self.model.encode(x).mode()

    def d_encode(self, x: Tensor) -> Tensor:
        return self.model.encode(x).mode()

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        return self.model.decode(z)
    
    def d_decode(self, z: Tensor) -> Tensor:
        return self.model.decode(z)


class CheffLDM:
    def __init__(
        self, model_path: str, ae_path: Optional[str] = None, device: Union[str, int, torch.device] = "cuda"
    ) -> None:
        self.device = device
        with contextlib.redirect_stdout(None):
            self.model = self._init_checkpoint(model_path, ae_path)

        self.model = self.model.to(self.device)
        self.model.model = self.model.model.to(self.device)
        self.model.eval()

        self.sample_shape = [
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size,
        ]

    @torch.no_grad()
    def sample(
        self, batch_size: int = 1, sampling_steps: int = 100, eta: float = 1.0, decode: bool = True, *args, **kwargs
    ) -> Tensor:
        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(sampling_steps, batch_size=batch_size, shape=self.sample_shape, eta=eta, verbose=False)

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @torch.no_grad()
    def sample_inpaint(
        self,
        target_img: Tensor,
        mask: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            batch_size=target_img.shape[0],
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    def differnetiable_sample_inpaint(
        self,
        target_img: Tensor,
        mask: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            batch_size=target_img.shape[0],
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    def _init_checkpoint(self, model_path: str, ae_path: Optional[str] = None) -> LatentDiffusion:
        config_dict = self._get_config_dict(ae_path)
        model = LatentDiffusion(**config_dict)

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict["state_dict"], strict=False)
        return model

    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "linear_start": 0.0015,
            "linear_end": 0.0295,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "image",
            "image_size": 64,
            "channels": 3,
            "monitor": "val/loss_simple_ema",
            "unet_config": CheffLDM._get_unet_config_dict(),
            "first_stage_config": CheffLDM._get_first_stage_config_dict(ae_path),
            "cond_stage_config": "__is_unconditional__",
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                "in_channels": 3,
                "out_channels": 3,
                "model_channels": 224,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 3, 4],
                "num_head_channels": 32,
            },
        }

    @staticmethod
    def _get_first_stage_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "target": "cheff.ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "embed_dim": 3,
                "ckpt_path": ae_path,
                "ddconfig": {
                    "double_z": True,
                    "z_channels": 3,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": (1, 2, 4),
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                "lossconfig": {"target": "torch.nn.Identity"},
            },
        }


class CheffLDMT2I(CheffLDM):

    @torch.no_grad()
    def sample(
        self,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        conditioning: str = "",
        *args,
        **kwargs,
    ) -> Tensor:
        conditioning = self.model.get_learned_conditioning(conditioning)

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps, conditioning=conditioning, batch_size=1, shape=self.sample_shape, eta=eta, verbose=False
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @torch.no_grad()
    def sample_inpaint(
        self,
        target_img: Tensor,
        mask: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        conditioning: str = "",
        *args,
        **kwargs,
    ) -> Tensor:
        assert target_img.shape[0] == 1, "Method implemented only for batch size = 1."

        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()
        conditioning = self.model.get_learned_conditioning(conditioning)

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            conditioning=conditioning,
            batch_size=1,
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @torch.no_grad()
    def sample_inpaint_without_mask(
        self,
        target_img: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        conditioning: str = "",
        *args,
        **kwargs,
    ) -> Tensor:
        assert target_img.shape[0] == 1, "Method implemented only for batch size = 1."

        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()
        conditioning = self.model.get_learned_conditioning(conditioning)

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            conditioning=conditioning,
            batch_size=1,
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            x0=target_enc,
            # mask=mask,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "linear_start": 0.0015,
            "linear_end": 0.0295,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "image",
            "cond_stage_key": "caption",
            "image_size": 64,
            "channels": 3,
            "cond_stage_trainable": True,
            "conditioning_key": "crossattn",
            "monitor": "val/loss_simple_ema",
            "scale_factor": 0.18215,
            "unet_config": CheffLDMT2I._get_unet_config_dict(),
            "first_stage_config": CheffLDMT2I._get_first_stage_config_dict(ae_path),
            "cond_stage_config": CheffLDMT2I._get_cond_config_dict(),
        }

    @staticmethod
    def _get_cond_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.encoders.modules.BERTEmbedder",
            "params": {
                "n_embed": 1280,
                "n_layer": 32,
            },
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                "in_channels": 3,
                "out_channels": 3,
                "model_channels": 224,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 4, 4],
                "num_heads": 8,
                "use_spatial_transformer": True,
                "transformer_depth": 1,
                "context_dim": 1280,
                "use_checkpoint": True,
                "legacy": False,
            },
        }


class CheffLDMT2IEdit(CheffLDM):

    @torch.no_grad()
    def sample(
        self,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        conditioning: str = "",
        mask: Tensor = None,
        target_img: Tensor = None,
        direct_conditioning: Tensor = None,
        *args,
        **kwargs,
    ) -> Tensor:

        bs = target_img.shape[0]

        target_enc = self.model.encode_first_stage(target_img).mode()

        if (conditioning is None or conditioning == "") and direct_conditioning is None:
            conditioning = None
        else:
            if not isinstance(conditioning, list):
                conditioning = [conditioning]

            conditioning = (
                self.model.get_learned_conditioning(conditioning)
                if direct_conditioning is None
                else direct_conditioning
            )

        ddim = DDIMSampler(self.model)

        samples, intermediates = ddim.sample(
            sampling_steps,
            conditioning=conditioning,
            batch_size=bs,
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
            # x_T=x_T,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples, intermediates

    @torch.no_grad()
    def sample_inpaint(
        self,
        target_img: Tensor,
        mask: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        conditioning: str = "",
        *args,
        **kwargs,
    ) -> Tensor:
        assert target_img.shape[0] == 1, "Method implemented only for batch size = 1."

        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()
        conditioning = self.model.get_learned_conditioning(conditioning)

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            conditioning=conditioning,
            batch_size=1,
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
            # x_T=x_T
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @torch.no_grad()
    def sample_inpaint_without_mask(
        self,
        target_img: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        conditioning: str = "",
        *args,
        **kwargs,
    ) -> Tensor:
        assert target_img.shape[0] == 1, "Method implemented only for batch size = 1."

        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()
        conditioning = self.model.get_learned_conditioning(conditioning)

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            conditioning=conditioning,
            batch_size=1,
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            x0=target_enc,
            # mask=mask,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "linear_start": 0.0015,
            "linear_end": 0.0295,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "image",
            "cond_stage_key": "caption",
            "image_size": 64,
            "channels": 3,
            "cond_stage_trainable": True,
            "conditioning_key": "crossattn",
            "monitor": "val/loss_simple_ema",
            "scale_factor": 0.18215,
            "unet_config": CheffLDMT2I._get_unet_config_dict(),
            "first_stage_config": CheffLDMT2I._get_first_stage_config_dict(ae_path),
            "cond_stage_config": CheffLDMT2I._get_cond_config_dict(),
        }

    @staticmethod
    def _get_cond_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.encoders.modules.BERTEmbedder",
            "params": {
                "n_embed": 1280,
                "n_layer": 32,
            },
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                "in_channels": 3,
                "out_channels": 3,
                "model_channels": 224,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 4, 4],
                "num_heads": 8,
                "use_spatial_transformer": True,
                "transformer_depth": 1,
                "context_dim": 1280,
                "use_checkpoint": True,
                "legacy": False,
            },
        }


class CheffLDMMaskCond:
    def __init__(
        self,
        model_path: str,
        ae_path: Optional[str] = None,
        device: Union[str, int, torch.device] = "cuda",
        load_checkpoint: bool = True,
    ) -> None:
        self.device = device
        self.model = self._init_checkpoint(model_path, ae_path, load_checkpoint=load_checkpoint)

        self.model = self.model.to(self.device)
        self.model.model = self.model.model.to(self.device)
        self.model.eval()

        self.sample_shape = [
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size,
        ]

    @torch.no_grad()
    def sample(
        self, batch_size: int = 1, sampling_steps: int = 100, eta: float = 1.0, decode: bool = True, *args, **kwargs
    ) -> Tensor:
        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(sampling_steps, batch_size=batch_size, shape=self.sample_shape, eta=eta, verbose=False)

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @torch.no_grad()
    def sample_inpaint(
        self,
        target_img: Tensor,
        mask: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            batch_size=target_img.shape[0],
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    def differnetiable_sample_inpaint(
        self,
        target_img: Tensor,
        mask: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            batch_size=target_img.shape[0],
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    # def _init_checkpoint(self, model_path: str, ae_path: Optional[str] = None) -> LatentDiffusion:
    #     config_dict = self._get_config_dict(ae_path)
    #     model = LatentDiffusion(**config_dict)

    #     state_dict = torch.load(model_path, map_location=self.device)
    #     model.load_state_dict(state_dict["state_dict"], strict=False)
    #     return model

    def _init_checkpoint(self, model_path: str, ae_path: Optional[str] = None, load_checkpoint=True) -> LatentDiffusion:
        config_dict = self._get_config_dict(ae_path)
        model = LatentDiffusion(**config_dict)

        if load_checkpoint:
            state_dict = torch.load(model_path, map_location=self.device)

            # Custom loading function
            def load_partial_state_dict(model, state_dict):
                model_dict = model.state_dict()
                for name, param in state_dict["state_dict"].items():
                    if name in model_dict:
                        if "input_blocks.0.0.weight" in name:
                            # For the first conv layer, load the first 3 channels and initialize the 4th
                            existing_weight = param
                            new_weight = model_dict[name]
                            new_weight[:, :3, :, :] = existing_weight
                            # Initialize the 4th channel with xavier uniform... (?)
                            torch.nn.init.xavier_uniform_(new_weight[:, 3:, :, :])
                            model_dict[name] = new_weight
                        elif param.shape == model_dict[name].shape:
                            model_dict[name] = param

                model.load_state_dict(model_dict)

            # Use the custom loading function
            load_partial_state_dict(model, state_dict)
        else:
            print("Not loading checkpoint for CheffLDMMaskCond")

        return model

    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "linear_start": 0.0015,
            "linear_end": 0.0295,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "image",
            "image_size": 64,
            "channels": 3,
            "monitor": "val/loss_simple_ema",
            "conditioning_key": "concat",
            "unet_config": CheffLDMMaskCond._get_unet_config_dict(),
            "first_stage_config": CheffLDMMaskCond._get_first_stage_config_dict(ae_path),
            "cond_stage_config": CheffLDMMaskCond._get_cond_stage_config_dict(),
        }

    @staticmethod
    def _get_cond_stage_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.encoders.modules.SpatialRescaler",
            "params": {"n_stages": 1, "in_channels": 1, "out_channels": 1},
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                # "in_channels": 3 + 3 + 1,  # image + mask + conditioning (image)
                "in_channels": 3 + 1,  # image + mask
                "out_channels": 3,
                "model_channels": 224,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 3, 4],
                "num_head_channels": 32,
            },
        }

    @staticmethod
    def _get_first_stage_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "target": "cheff.ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "embed_dim": 3,
                "ckpt_path": ae_path,
                "ddconfig": {
                    "double_z": True,
                    "z_channels": 3,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": (1, 2, 4),
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                "lossconfig": {"target": "torch.nn.Identity"},
            },
        }


class CheffLDMImageMaskCond:
    def __init__(
        self,
        model_path: str,
        ae_path: Optional[str] = None,
        device: Union[str, int, torch.device] = "cuda",
        load_checkpoint: bool = True,
    ) -> None:
        self.device = device
        self.model = self._init_checkpoint(model_path, ae_path, load_checkpoint=load_checkpoint)

        self.model = self.model.to(self.device)
        self.model.model = self.model.model.to(self.device)
        self.model.eval()

        self.sample_shape = [
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size,
        ]

    @torch.no_grad()
    def sample(
        self, batch_size: int = 1, sampling_steps: int = 100, eta: float = 1.0, decode: bool = True, *args, **kwargs
    ) -> Tensor:
        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(sampling_steps, batch_size=batch_size, shape=self.sample_shape, eta=eta, verbose=False)

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @torch.no_grad()
    def sample_inpaint(
        self,
        target_img: Tensor,
        mask: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            batch_size=target_img.shape[0],
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    def differnetiable_sample_inpaint(
        self,
        target_img: Tensor,
        mask: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            batch_size=target_img.shape[0],
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    # def _init_checkpoint(self, model_path: str, ae_path: Optional[str] = None) -> LatentDiffusion:
    #     config_dict = self._get_config_dict(ae_path)
    #     model = LatentDiffusion(**config_dict)

    #     state_dict = torch.load(model_path, map_location=self.device)
    #     model.load_state_dict(state_dict["state_dict"], strict=False)
    #     return model

    def _init_checkpoint(self, model_path: str, ae_path: Optional[str] = None, load_checkpoint=True) -> LatentDiffusion:
        config_dict = self._get_config_dict(ae_path)
        model = LatentDiffusion(**config_dict)

        if load_checkpoint:
            state_dict = torch.load(model_path, map_location=self.device)

            # Custom loading function
            def load_partial_state_dict(model, state_dict):
                model_dict = model.state_dict()
                for name, param in state_dict["state_dict"].items():
                    if name in model_dict:
                        if "input_blocks.0.0.weight" in name:
                            # For the first conv layer, load the first 3 channels and initialize the 4th
                            existing_weight = param
                            new_weight = model_dict[name]
                            new_weight[:, :3, :, :] = existing_weight
                            # Initialize the 4th channel with xavier uniform... (?)
                            torch.nn.init.xavier_uniform_(new_weight[:, 3:, :, :])
                            model_dict[name] = new_weight
                        elif param.shape == model_dict[name].shape:
                            model_dict[name] = param

                model.load_state_dict(model_dict)

            # Use the custom loading function
            load_partial_state_dict(model, state_dict)
        else:
            print("Not loading checkpoint for CheffLDMMaskCond")

        return model

    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "linear_start": 0.0015,
            "linear_end": 0.0295,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "image",
            "image_size": 64,
            "channels": 3,
            "monitor": "val/loss_simple_ema",
            "conditioning_key": "concat",
            "unet_config": CheffLDMImageMaskCond._get_unet_config_dict(),
            "first_stage_config": CheffLDMImageMaskCond._get_first_stage_config_dict(ae_path),
            "cond_stage_config": CheffLDMImageMaskCond._get_cond_stage_config_dict(),
        }

    @staticmethod
    def _get_cond_stage_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.encoders.modules.SpatialRescaler",
            "params": {"n_stages": 1, "in_channels": 1, "out_channels": 1},
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                "in_channels": 3 + 3 + 1,  # image + mask + conditioning (image)
                # "in_channels": 3 + 1,  # image + mask
                "out_channels": 3,
                "model_channels": 224,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 3, 4],
                "num_head_channels": 32,
            },
        }

    @staticmethod
    def _get_first_stage_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "target": "cheff.ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "embed_dim": 3,
                "ckpt_path": ae_path,
                "ddconfig": {
                    "double_z": True,
                    "z_channels": 3,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": (1, 2, 4),
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                "lossconfig": {"target": "torch.nn.Identity"},
            },
        }


class CheffLDMImageCond:
    def __init__(
        self,
        model_path: str,
        ae_path: Optional[str] = None,
        device: Union[str, int, torch.device] = "cuda",
        load_checkpoint: bool = True,
    ) -> None:
        self.device = device
        self.model = self._init_checkpoint(model_path, ae_path, load_checkpoint=load_checkpoint)

        self.model = self.model.to(self.device)
        self.model.model = self.model.model.to(self.device)
        self.model.eval()

        self.sample_shape = [
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size,
        ]

    @torch.no_grad()
    def sample(
        self, batch_size: int = 1, sampling_steps: int = 100, eta: float = 1.0, decode: bool = True, *args, **kwargs
    ) -> Tensor:
        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(sampling_steps, batch_size=batch_size, shape=self.sample_shape, eta=eta, verbose=False)

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    @torch.no_grad()
    def sample_inpaint(
        self,
        target_img: Tensor,
        mask: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            batch_size=target_img.shape[0],
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    def differnetiable_sample_inpaint(
        self,
        target_img: Tensor,
        mask: Tensor,
        sampling_steps: int = 100,
        eta: float = 1.0,
        decode: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        # Encode original image via AE
        target_enc = self.model.encode_first_stage(target_img).mode()

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps,
            batch_size=target_img.shape[0],
            shape=self.sample_shape,
            eta=eta,
            verbose=False,
            mask=mask,
            x0=target_enc,
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    # def _init_checkpoint(self, model_path: str, ae_path: Optional[str] = None) -> LatentDiffusion:
    #     config_dict = self._get_config_dict(ae_path)
    #     model = LatentDiffusion(**config_dict)

    #     state_dict = torch.load(model_path, map_location=self.device)
    #     model.load_state_dict(state_dict["state_dict"], strict=False)
    #     return model

    def _init_checkpoint(self, model_path: str, ae_path: Optional[str] = None, load_checkpoint=True) -> LatentDiffusion:
        config_dict = self._get_config_dict(ae_path)
        model = LatentDiffusion(**config_dict)

        if load_checkpoint:
            state_dict = torch.load(model_path, map_location=self.device)

            # Custom loading function
            def load_partial_state_dict(model, state_dict):
                model_dict = model.state_dict()
                for name, param in state_dict["state_dict"].items():
                    if name in model_dict:
                        if "input_blocks.0.0.weight" in name:
                            # For the first conv layer, load the first 3 channels and initialize the 4th
                            existing_weight = param
                            new_weight = model_dict[name]
                            new_weight[:, :3, :, :] = existing_weight
                            # Initialize the 4th channel with xavier uniform... (?)
                            torch.nn.init.xavier_uniform_(new_weight[:, 3:, :, :])
                            model_dict[name] = new_weight
                        elif param.shape == model_dict[name].shape:
                            model_dict[name] = param

                model.load_state_dict(model_dict)

            # Use the custom loading function
            load_partial_state_dict(model, state_dict)
        else:
            print("Not loading checkpoint for CheffLDMMaskCond")

        return model

    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "linear_start": 0.0015,
            "linear_end": 0.0295,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "image",
            "image_size": 64,
            "channels": 3,
            "monitor": "val/loss_simple_ema",
            "conditioning_key": "concat",
            "unet_config": CheffLDMImageCond._get_unet_config_dict(),
            "first_stage_config": CheffLDMImageCond._get_first_stage_config_dict(ae_path),
            "cond_stage_config": CheffLDMImageCond._get_cond_stage_config_dict(),
        }

    @staticmethod
    def _get_cond_stage_config_dict() -> Dict:
        return {
            # "target": "cheff.ldm.modules.encoders.modules.SpatialRescaler",
            # "params": {"n_stages": 1, "in_channels": 1, "out_channels": 1},
            "target": "torch.nn.Identity"
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            "target": "cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {
                "image_size": 64,
                "in_channels": 3 + 3,  # image + conditioning (image)
                # "in_channels": 3 + 1,  # image + mask
                "out_channels": 3,
                "model_channels": 224,
                "attention_resolutions": [8, 4, 2],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 3, 4],
                "num_head_channels": 32,
            },
        }

    @staticmethod
    def _get_first_stage_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            "target": "cheff.ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "embed_dim": 3,
                "ckpt_path": ae_path,
                "ddconfig": {
                    "double_z": True,
                    "z_channels": 3,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": (1, 2, 4),
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                "lossconfig": {"target": "torch.nn.Identity"},
            },
        }
