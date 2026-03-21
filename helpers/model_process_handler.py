from __future__ import annotations

import base64
import io

import torch
from PIL import Image
from torchvision import transforms
from ts.handler_utils.timer import timed
from ts.torch_handler.base_handler import BaseHandler


class BirdClassifierHandler(BaseHandler):
    @timed
    def preprocess(self, data):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image = data[0].get('data') or data[0].get('body')
        if isinstance(image, str):
            image = base64.b64decode(image)
        if isinstance(image, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image)).convert('RGB')
            image = transform(image).unsqueeze(0).to(self.device)
        else:
            image = torch.FloatTensor(image).unsqueeze(0).to(self.device)
        return image

    @timed
    def inference(self, data):
        with torch.inference_mode():
            output = self.model(data)
        return output

    @timed
    def postprocess(self, data):
        return data.cpu().numpy().tolist()
