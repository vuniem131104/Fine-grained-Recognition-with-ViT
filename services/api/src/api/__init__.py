from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from scipy.special import softmax
from torchvision import transforms

# Image preprocessing transform
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ],
)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image to tensor for model inference."""
    return transform(image).unsqueeze(0).numpy()


def postprocess_predictions(predictions: np.ndarray, idx_2_class: dict) -> dict:
    """Postprocess model predictions to human-readable results."""
    probabilities = softmax(predictions[0])
    predicted_class = int(probabilities.argmax())

    top_3_indices = probabilities.argsort()[-4:][::-1][1:]
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
        'top_3_alternatives': top_3_classes,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load class mappings and create HTTP client for KServe."""
    classes_file = os.getenv(
        'CLASSES_FILE',
        '/home/lehoangvu/Project_AIDE1/services/api/src/api/classes.txt',
    )
    with open(classes_file) as f:
        classes = f.readlines()
        classes = [line.strip().split()[1].split('.')[-1] for line in classes]

    app.state.idx_2_class = {idx: cls_name for idx, cls_name in enumerate(classes)}

    # KServe model endpoint (InferenceService URL)
    # Local: http://localhost:8080/v1/models/birds-model:predict
    # K8s:   http://birds-model.<namespace>.svc.cluster.local/v1/models/birds-model:predict
    app.state.kserve_url = os.getenv(
        'KSERVE_MODEL_URL',
        'http://localhost:8080/v1/models/birds-model:predict',
    )

    # HTTP client for calling KServe
    app.state.http_client = httpx.AsyncClient(timeout=30.0)

    yield

    await app.state.http_client.aclose()


app = FastAPI(
    version='1.0.0',
    title='Birds Classification API',
    description='Pre/Post-processing API for bird species classification. Calls KServe for model inference.',
    lifespan=lifespan,
)


@app.post(
    '/predict',
    summary='Predict bird species from an image',
    description='Upload an image file. Preprocessing is done here, inference via KServe, postprocessing returned.',
)
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        tensor = preprocess_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image file: {e}')

    try:
        payload = {'instances': tensor.tolist()}

        response = await app.state.http_client.post(
            app.state.kserve_url,
            json=payload,
        )
        response.raise_for_status()
        kserve_result = response.json()

        predictions = np.array(kserve_result['predictions'])
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail='Model inference timeout. Model may be scaling up from zero.',
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f'KServe model error: {e.response.text}',
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference error: {e}')

    try:
        results = postprocess_predictions(predictions, app.state.idx_2_class)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Postprocessing error: {e}')


@app.post(
    '/predict/base64',
    summary='Predict bird species from base64 encoded image',
    description='Send base64 encoded image for classification.',
)
async def predict_base64(payload: dict) -> JSONResponse:
    """Accept base64 encoded image."""
    try:
        image_b64 = payload.get('image')
        if not image_b64:
            raise HTTPException(status_code=400, detail='Missing "image" field')

        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        tensor = preprocess_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image: {e}')

    try:
        payload = {'instances': tensor.tolist()}
        response = await app.state.http_client.post(
            app.state.kserve_url,
            json=payload,
        )
        response.raise_for_status()
        predictions = np.array(response.json()['predictions'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference error: {e}')

    results = postprocess_predictions(predictions, app.state.idx_2_class)
    return JSONResponse(content=results)


@app.get(
    '/health',
    summary='Health check endpoint',
    description='Check if the API is running.',
)
async def health_check() -> JSONResponse:
    return JSONResponse(content={'status': 'ok'})


@app.get(
    '/ready',
    summary='Readiness check',
    description='Check if KServe model is reachable.',
)
async def readiness_check() -> JSONResponse:
    """Check if KServe model is ready (may be scaled to zero)."""
    try:
        # Just check the model endpoint health
        model_health_url = app.state.kserve_url.replace(':predict', '')
        response = await app.state.http_client.get(model_health_url, timeout=5.0)
        if response.status_code == 200:
            return JSONResponse(content={'status': 'ready', 'model': 'available'})
        else:
            return JSONResponse(
                status_code=503,
                content={'status': 'not_ready', 'model': 'unavailable'},
            )
    except Exception:
        return JSONResponse(
            content={'status': 'ready', 'model': 'may_be_scaled_to_zero'},
        )


def main():
    """Run FastAPI server for pre/post-processing."""
    uvicorn.run(
        'api:app',
        host='0.0.0.0',
        port=int(os.getenv('PORT', 8000)),
        workers=int(os.getenv('WORKERS', 1)),
        reload=os.getenv('RELOAD', 'false').lower() == 'true',
    )
