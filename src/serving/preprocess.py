from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ],
)


def preprocess_image(image: Image) -> torch.Tensor:
    """
    Preprocess the input image for model prediction.

    Args:
        image (Image): The input image to preprocess.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    return transform(image).unsqueeze(0)
