import chexzero.clip
import torch
from torch import nn
from torch.nn import functional as F
from chexzero.clip import tokenize
from PIL import Image


model, preprocess = chexzero.clip.load("ViT-B/32", device="cpu", jit=False)
model.load_state_dict(
    torch.load("/cluster/work/cvl/agoekmen/chexzero/best_64_5e-05_original_22000_0.864.pt", map_location="cpu")
)


def encode_text(text: str):
    tokenized_text = tokenize(text)
    text_features = model.encode_text(tokenized_text)
    return text_features


def encode_image(image: torch.Tensor):
    image_features = model.encode_image(image)
    return image_features


def calculate_scores(text_features: torch.Tensor, image_features: torch.Tensor):
    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


if __name__ == "__main__":
    text = ["Pleural Effusion", "No Pleural Effusion"]
    image = Image.open(
        "/cluster/home/agoekmen/projects/chexray-diffusion/removal-editing/all_exp/working_experiments/__attn/original.png"
    )
    image = preprocess(image).unsqueeze(0)
    text_features = encode_text(text)
    image_features = encode_image(image)

    print(text_features.shape)
    print(image_features.shape)

    logits_per_image, logits_per_text = calculate_scores(text_features, image_features)
    print(logits_per_image)
    print(logits_per_text)

    probs = logits_per_image.softmax(dim=1)
    print(probs)
