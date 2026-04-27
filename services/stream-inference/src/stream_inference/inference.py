import httpx
import numpy as np
from scipy.special import softmax

from stream_inference.classes import CLASSES


def build_kserve_payload(img_b64: str) -> dict:
    return {
        "inputs": [
            {
                "name": "input",
                "shape": [1],
                "datatype": "BYTES",
                "data": [img_b64],
            }
        ]
    }


def postprocess(prediction: np.ndarray) -> dict:
    probs = softmax(prediction)
    pred = int(probs.argmax())
    top3 = probs.argsort()[-4:][::-1][1:]
    return {
        "predicted_class": CLASSES[pred],
        "probability": float(probs[pred]),
        "top_3_alternatives": [
            {"class": CLASSES[i], "probability": float(probs[i])} for i in top3
        ],
    }


async def call_kserve(
    http_client: httpx.AsyncClient, kserve_url: str, img_b64: str
) -> dict:
    response = await http_client.post(
        f"{kserve_url}/infer",
        headers={"Content-Type": "application/json"},
        json=build_kserve_payload(img_b64),
    )
    response.raise_for_status()
    body = response.json()
    logits = np.array(body["outputs"][0]["data"])
    return postprocess(logits)
