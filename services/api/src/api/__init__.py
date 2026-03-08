from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager

import httpx
import numpy as np
import psycopg2
import structlog
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from psycopg2.extras import Json
from scipy.special import softmax
from torchvision import transforms
from ultralytics import YOLO

logger = structlog.get_logger()

classes = ['Black footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani', 'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird', 'Red winged Blackbird', 'Rusty Blackbird', 'Yellow headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting', 'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow breasted Chat', 'Eastern Towhee', 'Chuck will Widow', 'Brandt Cormorant', 'Red faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow', 'Black billed Cuckoo', 'Mangrove Cuckoo', 'Yellow billed Cuckoo', 'Gray crowned Rosy Finch', 'Purple Finch', 'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive sided Flycatcher', 'Scissor tailed Flycatcher', 'Vermilion Flycatcher', 'Yellow bellied Flycatcher', 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch', 'Boat tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied billed Grebe', 'Western Grebe', 'Blue Grosbeak', 'Evening Grosbeak', 'Pine Grosbeak', 'Rose breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous winged Gull', 'Heermann Gull', 'Herring Gull', 'Ivory Gull', 'Ring billed Gull', 'Slaty backed Gull', 'Western Gull', 'Anna Hummingbird', 'Ruby throated Hummingbird', 'Rufous Hummingbird', 'Green Violetear', 'Long tailed Jaeger', 'Pomarine Jaeger', 'Blue Jay', 'Florida Jay', 'Green Jay', 'Dark eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher', 'Pied Kingfisher', 'Ringed Kingfisher', 'White breasted Kingfisher', 'Red legged Kittiwake', 'Horned Lark', 'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red breasted Merganser', 'Mockingbird', 'Nighthawk', 'Clark Nutcracker', 'White breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole', 'Scott Oriole', 'Ovenbird', 'Brown Pelican', 'White Pelican', 'Western Wood Pewee', 'Sayornis', 'American Pipit', 'Whip poor Will', 'Horned Puffin', 'Common Raven', 'White necked Raven', 'American Redstart', 'Geococcyx', 'Loggerhead Shrike', 'Great Grey Shrike', 'Baird Sparrow', 'Black throated Sparrow', 'Brewer Sparrow', 'Chipping Sparrow', 'Clay colored Sparrow', 'House Sparrow', 'Field Sparrow', 'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow', 'Le Conte Sparrow', 'Lincoln Sparrow', 'Nelson Sharp tailed Sparrow', 'Savannah Sparrow', 'Seaside Sparrow', 'Song Sparrow', 'Tree Sparrow', 'Vesper Sparrow', 'White crowned Sparrow', 'White throated Sparrow', 'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 'Tree Swallow', 'Scarlet Tanager', 'Summer Tanager', 'Artic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 'Forsters Tern', 'Least Tern', 'Green tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Black capped Vireo', 'Blue headed Vireo', 'Philadelphia Vireo', 'Red eyed Vireo', 'Warbling Vireo', 'White eyed Vireo', 'Yellow throated Vireo', 'Bay breasted Warbler', 'Black and white Warbler', 'Black throated Blue Warbler', 'Blue winged Warbler', 'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler', 'Chestnut sided Warbler', 'Golden winged Warbler', 'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 'Myrtle Warbler', 'Nashville Warbler', 'Orange crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler', 'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm eating Warbler', 'Yellow Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush', 'Bohemian Waxwing', 'Cedar Waxwing', 'American Three toed Woodpecker', 'Pileated Woodpecker', 'Red bellied Woodpecker', 'Red cockaded Woodpecker', 'Red headed Woodpecker', 'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren', 'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat']

def insert_prediction(base64_image: str, probability: float, predicted_class: str, alternatives: list, connection: psycopg2.extensions.connection) -> None:
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO model_predictions (base64_image, probability, predicted_class, alternatives)
    VALUES (%s, %s, %s, %s);
    """
    cursor.execute(insert_query, (base64_image, probability, predicted_class, Json(alternatives)))
    connection.commit()

def preprocess_image(image: Image.Image, transform: transforms.Compose) -> np.ndarray:
    """
    Preprocess the input image for model inference. Applies resizing, cropping, normalization, and converts to numpy array.

    Args:
        image (Image.Image): The input image to preprocess.
        transform (transforms.Compose): The preprocessing transformations to apply.

    Returns:
        np.ndarray: The preprocessed image as a numpy array.
    """
    logger.info('Preprocessing image for model inference.')
    try:
        preprocessed_image = transform(image).unsqueeze(0).numpy()
        logger.info('Image preprocessing successful.')
        return preprocessed_image
    except Exception as e:
        logger.exception(
            'Error during image preprocessing.',
            extra={
                'error': str(e),
            },
        )
        raise


def postprocess_predictions(prediction: np.ndarray, idx_2_class: dict) -> dict:
    """
    Postprocess the model predictions to extract the predicted class, its probability, and the top 3 alternative classes with their probabilities.

    Args:
        prediction (np.ndarray): The raw model predictions.
        idx_2_class (dict): A mapping from class indices to class names.

    Returns:
        dict: A dictionary containing the predicted class, its probability, and the top 3 alternative classes with their probabilities.
    """
    try:
        logger.info('Postprocessing model predictions.')
        probabilities = softmax(prediction)
        predicted_class = int(probabilities.argmax())

        top_3_indices = probabilities.argsort()[-4:][::-1][1:]
        top_3_classes = [
            {
                'class': idx_2_class[idx],
                'probability': float(probabilities[idx]),
            }
            for idx in top_3_indices
        ]

        result = {
            'predicted_class': idx_2_class[predicted_class],
            'probability': float(probabilities[predicted_class]),
            'top_3_alternatives': top_3_classes,
        }
        logger.info('Postprocessing successful.', extra={'result': result})
        return result
    except Exception as e:
        logger.exception(
            'Error during postprocessing of predictions.',
            extra={
                'error': str(e),
                'raw_prediction': prediction.tolist(),
            },
        )
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load class mappings and create HTTP client for KServe."""
    app.state.idx_2_class = {idx: cls_name for idx, cls_name in enumerate(classes)}

    app.state.kserve_url = os.getenv(
        'KSERVE_MODEL_URL',
    )

    app.state.yolo = YOLO('/app/models/bird_detection/yolov8n.pt')

    app.state.transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ],
    )

    app.state.http_client = httpx.AsyncClient(timeout=30.0)

    app.state.connection = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        database=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        port=os.getenv('POSTGRES_PORT'),
    )

    yield

    await app.state.http_client.aclose()


app = FastAPI(
    version='1.0.0',
    title='Birds Classification API',
    description='API for bird species classification. Calls KServe for model inference.',
    lifespan=lifespan,
)


@app.post(
    '/predict',
    summary='Predict bird species from an image',
    description='Upload an image file. Preprocessing is done here, inference via KServe, postprocessing returned.',
)
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    try:
        logger.info('Received prediction request.', extra={'filename': file.filename, 'content_type': file.content_type})
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        logger.exception('Error occurred while processing the image file.', extra={'filename': file.filename, 'error': str(e)})
        raise HTTPException(status_code=400, detail=f'Invalid image file: {e}')

    try:
        logger.info('Running object detection on the image.')
        results = app.state.yolo.predict(source=image, classes=[14])
        if len(results[0].boxes) > 0:
            top_box = results[0].boxes[0]
            x1, y1, x2, y2 = top_box.xyxy[0].tolist()
            cropped_image = image.crop((x1, y1, x2, y2))
        else:
            cropped_image = image
    except Exception as e:
        logger.exception('Error during object detection.', extra={'error': str(e)})
        raise HTTPException(status_code=500, detail=f'Object detection error: {e}')

    try:
        data_array = preprocess_image(cropped_image, app.state.transform)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Preprocessing error: {e}')

    try:
        logger.info('Sending inference request to KServe.')
        payload = {
            'inputs': [{
                'name': 'input',
                'shape': list(data_array.shape),
                'datatype': 'FP32',
                'data': data_array.tolist(),
            }],
        }

        response = await app.state.http_client.post(
            app.state.kserve_url + '/infer',
            headers={'Content-Type': 'application/json'},
            json=payload,
        )
        response.raise_for_status()
        kserve_result = response.json()

        predictions = np.array(kserve_result['outputs'][0]['data'])
    except httpx.TimeoutException:
        logger.exception('Model inference timeout. Model may be scaling up from zero.')
        raise HTTPException(
            status_code=504,
            detail='Model inference timeout. Model may be scaling up from zero.',
        )
    except httpx.HTTPStatusError as e:
        logger.exception('KServe model inference error.', extra={'response_text': e.response.text})
        raise HTTPException(
            status_code=502,
            detail=f'KServe model error: {e.response.text}',
        )
    except Exception as e:
        logger.exception('Unexpected error during inference.', extra={'error': str(e)})
        raise HTTPException(status_code=500, detail=f'Inference error: {e}')

    try:
        results = postprocess_predictions(predictions, app.state.idx_2_class)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Postprocessing error: {e}')

    try:
        logger.info('Inserting prediction into database.', extra={'record': results})
        base64_image = base64.b64encode(image_data).decode('utf-8')
        insert_prediction(base64_image, results['probability'], results['predicted_class'], results['top_3_alternatives'], app.state.connection)
        return JSONResponse(content=results)
    except Exception as e:
        logger.exception('Error inserting prediction into database.', extra={'error': str(e)})
        app.state.connection.rollback()
        raise HTTPException(status_code=500, detail=f'Postprocessing error: {e}')

@app.get(
    '/health',
    summary='Health check endpoint',
    description='Check if the API is running.',
)
async def health_check() -> JSONResponse:
    try:
        response = await app.state.http_client.get(app.state.kserve_url + '/ready')
        response.raise_for_status()
        return JSONResponse(content={'status': 'ok'}) if response.status_code == 200 else JSONResponse(content={'status': 'error'}, status_code=503)
    except httpx.HTTPError:
        return JSONResponse(content={'status': 'error'}, status_code=503)

def main():
    """Run FastAPI server for pre/post-processing."""
    uvicorn.run(
        'api:app',
        host='0.0.0.0',
        port=int(os.getenv('HTTP_PORT', 8000)),
        workers=int(os.getenv('WORKERS', 1)),
        reload=True,
    )
