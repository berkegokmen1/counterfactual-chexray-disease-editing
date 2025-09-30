import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from chexzero.chexzero.clip import tokenize
from models.clip import CLIPEvaluator


class CLIPLoss(nn.Module):
    def __init__(self, args, pos_prompt=None, neg_prompt=None, clip_model=None, clip_preprocess=None):
        super().__init__()
        self.clip_model = CLIPEvaluator(args, model=clip_model, preprocess=clip_preprocess)
        self.pos_prompt = pos_prompt
        self.neg_prompt = neg_prompt

        # print(f"CLIPLoss: Positive prompt {pos_prompt}, Negative prompt {neg_prompt}")

    def forward_images(self, original, edited, preprocess_images):
        if preprocess_images:
            original = self.clip_model.clip_transform_for_tensor(original)
            edited = self.clip_model.clip_transform_for_tensor(edited)

        original_features = self.clip_model.model.encode_image(original)
        edited_features = self.clip_model.model.encode_image(edited)

        return original_features, edited_features

    def forward_prompts(self, pos_prompt=None, neg_prompt=None):
        if pos_prompt is None:
            pos_prompt = self.pos_prompt
        if neg_prompt is None:
            neg_prompt = self.neg_prompt

        pos_features = self.clip_model.encode_text(pos_prompt)
        neg_features = self.clip_model.encode_text(neg_prompt)

        return pos_features, neg_features

    def forward(self, generated_images, pos_prompt=None, neg_prompt=None, return_logits_per_image=False):
        if pos_prompt is None:
            pos_prompt = self.pos_prompt
        if neg_prompt is None:
            neg_prompt = self.neg_prompt

        # Get logits
        logits_per_image, _ = self.clip_model.score(generated_images, [pos_prompt, neg_prompt])

        if return_logits_per_image:
            return logits_per_image

        # Apply softmax to get probabilities
        probs = F.softmax(logits_per_image, dim=1)

        # Get probabilities for positive and negative prompts
        pos_prob = probs[:, 0]  # Positive prompt is first
        neg_prob = probs[:, 1]  # Negative prompt is second

        # print("Sample probabilities:")
        # print(probs)

        eps = 1e-6
        # neg_prob = torch.clamp(neg_prob, min=eps)
        # pos_prob = torch.clamp(pos_prob, min=eps)

        # loss = -torch.log(neg_prob + eps) + torch.log(pos_prob + eps)
        probs = torch.clamp(probs, min=eps)
        target = torch.zeros_like(probs)
        target[:, 1] = 1.0
        loss = F.binary_cross_entropy_with_logits(probs, target)

        # print("Sample loss:", loss, loss.mean(), loss.shape, loss.requires_grad, torch.any(torch.isnan(loss)))

        return loss, probs


class DirectionLoss(nn.Module):

    def __init__(self, loss_type="mse"):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {"mse": torch.nn.MSELoss, "cosine": torch.nn.CosineSimilarity, "mae": torch.nn.L1Loss}[
            loss_type
        ]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1.0 - self.loss_func(x, y)

        return self.loss_func(x, y)


class CLIPDirectionLoss(nn.Module):

    def __init__(self, args, pos_prompt=None, neg_prompt=None, clip_model=None, clip_preprocess=None):
        super().__init__()

        self.clip_model = CLIPEvaluator(args, model=clip_model, preprocess=clip_preprocess)
        self.pos_prompt = pos_prompt
        self.neg_prompt = neg_prompt

        self.device = args.device

        self.loss = DirectionLoss(loss_type="cosine")

        self.target_direction = self.compute_text_direction()

        print(f"CLIPLoss: Positive prompt {pos_prompt}, Negative prompt {neg_prompt}")

    def encode_text(self, texts):
        tokenized_text = tokenize(texts).to(self.device)
        text_features = self.clip_model.model.encode_text(tokenized_text)

        return text_features

    def encode_image(self, images):
        images = self.clip_model.clip_transform_for_tensor(images)

        image_features = self.clip_model.model.encode_image(images)

        return image_features

    def get_text_features(self, texts):
        text_features = self.encode_text(texts).detach()

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, images):
        image_features = self.encode_image(images)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class=None, target_class=None) -> torch.Tensor:
        print("Computing text direction")
        if source_class is None:
            source_class = self.pos_prompt

        if target_class is None:
            target_class = self.neg_prompt

        text_features = self.get_text_features([source_class, target_class])
        source_features, target_features = text_features[0], text_features[1]

        text_direction = target_features - source_features  # .mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        print("Text direction", text_direction.shape)

        return text_direction

    def forward(self, src_images, target_images, pos_prompt=None, neg_prompt=None):
        if pos_prompt is None:
            pos_prompt = self.pos_prompt
        if neg_prompt is None:
            neg_prompt = self.neg_prompt

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction()

        src_encoding = self.get_image_features(src_images)
        target_encoding = self.get_image_features(target_images)

        edit_direction = target_encoding - src_encoding
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7

        # calculate the distance between src_encoding and target_encoding and it should be maximized

        return self.loss(edit_direction, self.target_direction).mean(), F.mse_loss(src_encoding, target_encoding)


def custom_cross_entropy(pred_logits, target_logits):
    pred_probs = F.softmax(pred_logits, dim=-1)

    # Create a mask for non-inf values in target_logits
    mask = ~torch.isinf(target_logits)

    # Normalize the non-inf part of target_logits
    target_probs = torch.where(mask, F.softmax(target_logits, dim=-1), torch.zeros_like(target_logits))

    # Compute cross-entropy loss only for non-inf positions
    loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-8), dim=-1)

    # Add a large penalty for predicting non-zero probability where target is -inf
    inf_penalty = torch.where(~mask, pred_probs, torch.zeros_like(pred_probs)).sum(dim=-1) * 1e2

    return (loss + inf_penalty).mean()


def kl_loss(pred_logits, target_logits):
    pred_log_softmax = F.log_softmax(pred_logits, dim=-1)
    target_softmax = F.softmax(target_logits, dim=-1)

    kl_loss = F.kl_div(pred_log_softmax, target_softmax, reduction="batchmean")
    return kl_loss


def masked_mse_loss(pred, target, mask):
    """Take in the mask for the region to be inpainted"""
    if pred.shape[-1] != mask.shape[-1]:
        mask = F.interpolate(mask, size=(pred.shape[-1], pred.shape[-1]), mode="bilinear", align_corners=False)

    return torch.mean(((pred - target) * mask) ** 2)
