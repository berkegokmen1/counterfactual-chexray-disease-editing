import chexzero.chexzero.clip
from chexzero.chexzero.clip import tokenize
import torch
from torch.nn import functional as F
from torchvision import transforms


class CLIPEvaluator:
    """
    https://huggingface.co/StanfordAIMI/XrayCLIP__vit-b-16__laion2b-s34b-b88k?library=transformers
    """

    def __init__(self, args, model=None, preprocess=None):
        self.args = args
        self.model_path = args.model.clip.path
        self.device = args.device

        self.model = model
        self.preprocess = preprocess

        if self.model is None and self.preprocess is None:
            print("Loading CLIP model for CLIPEvaluator...")
            self.model, self.preprocess = chexzero.clip.load("ViT-B/32", device="cpu", jit=False)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()

        self.input_size = self.model.visual.input_resolution

    def clip_transform_for_tensor(self, tensor: torch.Tensor, target_size=None):
        if target_size is None:
            target_size = self.input_size

        resize = transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC)

        resized = resize(tensor)

        # mean = [0.48145466, 0.4578275, 0.40821073]
        # std = [0.26862954, 0.26130258, 0.27577711]

        normalize = transforms.Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
        normalize_2 = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        # return resized
        return normalize_2(resized)

    def score(self, images: torch.Tensor, texts=None, preprocess_images=True):
        assert texts is not None and texts != [], "Texts must be provided for scoring."

        tokenized_text = tokenize(texts).to(self.device)
        text_features = self.model.encode_text(tokenized_text)

        if preprocess_images:
            images = self.clip_transform_for_tensor(images)

        image_features = self.model.encode_image(images)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        tokenized_text = tokenize(texts).to(self.device)
        return self.model.encode_text(tokenized_text)

    def score_single(self, image: torch.Tensor, text: str, preprocess_image=True, text_features=None):
        if text_features is None:
            text_features = self.encode_text(text)

        # Preprocess the image if necessary
        if preprocess_image:
            # image already has batch dimension
            image = self.clip_transform_for_tensor(image)  # add batch dimension

        # Extract image features
        image_features = self.model.encode_image(image)

        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity between the single image and text
        similarity = (image_features @ text_features.T).squeeze()

        # Rescale similarity from [-1, 1] to [0, 1]
        score = (similarity + 1) / 2

        return score.item()
