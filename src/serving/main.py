from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import mlflow
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from postprocess import post_process
from preprocess import preprocess_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))

    app.state.model = mlflow.pyfunc.load_model(
        model_uri='models:/BirdsClassificationModel/1',
    )
    with open('/app/classes.txt') as f:
        classes = f.readlines()
        classes = [line.strip().split()[1].split('.')[-1] for line in classes]

    app.state.idx_2_class = {idx: cls_name for idx, cls_name in enumerate(classes)}

    yield

app = FastAPI(
    version='1.0.0',
    title='Birds Classification API',
    description='An API for classifying bird species using a trained ML model.',
    lifespan=lifespan,
)


@app.post(
    '/predict',
    summary='Predict bird species from an image',
    description='Upload an image file to classify the bird species.',
)
async def predict(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        tensor = preprocess_image(image)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={'error': f"Invalid image file: {str(e)}"},
        )

    try:
        predictions = request.app.state.model.predict(tensor.numpy())
        results = post_process(predictions, request.app.state.idx_2_class)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': f"Prediction error: {str(e)}"},
        )


@app.get(
    '/health',
    summary='Health check endpoint',
    description='Check if the API is running.',
)
async def health_check() -> JSONResponse:
    return JSONResponse(content={'status': 'ok'})


def main():
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)


if __name__ == '__main__':
    main()
