from __future__ import annotations

import numpy as np
from scipy.special import softmax


def post_process(predictions: np.ndarray, idx_2_class: dict) -> dict:
    probabilities = softmax(predictions[0])
    predicted_class = int(probabilities.argmax())

    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3_classes = [
        {
            'class': idx_2_class[idx],
            'probability': float(probabilities[idx]),
        }
        for idx in top_3_indices
    ]

    return {
        'predicted_class': idx_2_class[predicted_class],
        'probability': float(probabilities[predicted_class]),
        'top_3_classes': top_3_classes,
    }
