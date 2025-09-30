import copy
import os
import torch

from cheff import CheffLDMImageMaskCond
from cheff import CheffLDMMaskCond
from cheff import CheffLDMImageCond
from cheff import CheffLDM
from cheff.ldm.models.autoencoder import AutoencoderKL
from cheff.ldm.models.diffusion.ddpm import LatentDiffusion
from cheff.ldm.models.diffusion.ddim import DDIMSampler
from cheff.ldm.modules.diffusionmodules.openaimodel import AttentionBlock
from cheff.ldm.modules.diffusionmodules.openaimodel import ResBlock
from src.utils import save_txt


def get_cheff_ldm(
    args, load_checkpoint=True, load_base_uncond=False, load_image_mask_cond=False, load_image_cond=False
):
    print("Loading cheff ldm...")
    cheff_args = args.model.cheff
    model_path, ae_path = cheff_args.ldm_path, cheff_args.ae_path

    if load_base_uncond:
        print("Loading CheffLDM...")
        cheff_ldm = CheffLDM(model_path=model_path, ae_path=ae_path, device=args.device)
        return cheff_ldm

    if not load_image_mask_cond and not load_image_cond:
        print("Loading CheffLDMMaskCond...")
        cheff_ldm = CheffLDMMaskCond(
            model_path=model_path, ae_path=ae_path, device=args.device, load_checkpoint=load_checkpoint
        )
    elif load_image_mask_cond:
        print("Loading CheffLDMImageMaskCond...")
        cheff_ldm = CheffLDMImageMaskCond(
            model_path=model_path, ae_path=ae_path, device=args.device, load_checkpoint=load_checkpoint
        )
    elif load_image_cond:
        print("Loading CheffLDMImageCond...")
        cheff_ldm = CheffLDMImageCond(
            model_path=model_path, ae_path=ae_path, device=args.device, load_checkpoint=load_checkpoint
        )
    else:
        raise ValueError("Invalid configuration for Cheff LDM")

    print("Cheff LDM loaded: ", load_checkpoint)

    if cheff_args.load_external:
        print("Loading external model...")
        cheff_ldm.model.load_state_dict(torch.load(cheff_args.external_path, map_location=args.device))

    return cheff_ldm


"""
The effect of applying the erasure objective (6) depends
on the subset of parameters that is fine-tuned. The main
distinction is between cross-attention parameters and noncross-attention parameters. Cross-attention parameters, illustrated in Figure 3a, serve as a gateway to the prompt, directly
depending on the text of the prompt, while other parameters
(Figure 3b) tend to contribute to a visual concept even if the
concept is not mentioned in the prompt.
Therefore we propose fine tuning the cross attentions,
ESD-x, when the erasure is required to be controlled and
specific to the prompt, such as when a named artistic style
should be erased. Further, we propose fine tuning unconditional layers (non-cross-attention modules), ESD-u, when
the erasure is required to be independent of the text in the
prompt, such as when the global concept of NSFW nudity
should be erased. We refer to cross-attention-only finetuning as ESD-x-η (where η refers to the strength of the
negative guidance), and we refer to the configuration that
tunes only non-cross-attention parameters as ESD-u-η. For
simplicity, we write ESD-x and ESD-u when η = 1
"""

all_train_methods = {
    "attn": [AttentionBlock],
    "res": [ResBlock],
    "attnres": [AttentionBlock, ResBlock],
    "all": None,
    "notime": None,
}


def get_trainable_params(model, args, specific_paths=["model.diffusion_model.input_blocks.0.0"], save_params=False):
    parameters = []
    parameter_names = []

    train_method = args.model.train.method.lower().strip()

    assert train_method in all_train_methods.keys(), "Unsupported train method"
    print("Using train method:", train_method)

    if train_method == "notime":
        # Go through all the parameters and only exclude the time embedding ones
        for name, param in model.named_parameters():
            if "time_embed" not in name and "first_stage_model" not in name:
                param.requires_grad = True
                parameters.append(param)
                parameter_names.append(name)

    elif train_method == "all":
        for name, param in model.named_parameters():
            if "first_stage_model" not in name:
                param.requires_grad = True
                parameters.append(param)
                parameter_names.append(name)

    else:
        train_filters = all_train_methods[train_method]

        def recurse(module, current_path=""):
            for name, child in module.named_children():
                path = f"{current_path}.{name}" if current_path else name

                if any(isinstance(child, klass) for klass in train_filters) or (
                    specific_paths and path in specific_paths
                ):
                    for param_name, param in child.named_parameters():
                        param.requires_grad = True
                        parameters.append(param)
                        parameter_names.append(
                            f"{path} -- {name}.{param_name} -- {child.__class__.__name__} -- {param_name}"
                        )
                else:
                    recurse(child, path)

        recurse(model)

    if save_params:
        save_txt(
            args.finetune.log_dir,
            "trainable_params",
            parameter_names,
        )

    # with open(os.path.join(args.out_dir, "non_trainable_params.txt"), "w") as f:
    #     f.write("\n".join(non_trainable_parameter_names))

    print("Trainable parameters:", len(parameters))

    return parameters


def clone_model_for_sample(original_model, args):
    cloned_model = copy.deepcopy(original_model)

    parameters = get_trainable_params(cloned_model.model, args, save_params=True)

    cloned_model.model.train()
    cloned_model.model.model.diffusion_model.train()
    return cloned_model, parameters
