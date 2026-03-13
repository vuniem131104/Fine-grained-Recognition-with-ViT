if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "Error: Cannot find file .env"
    exit 1
fi

if [ -z "$MLFLOW_OBJECT_STORAGE_BUCKET" ] || [ -z "$MODEL_STORAGE_BUCKET" ] || [ -z "$MODEL_NAME" ]; then
    echo "Error: Missing required environment variables"
    exit 1
fi

echo "Creating MAR file for model: $MODEL_NAME"

ENCODED_PASSWORD=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${MLFLOW_SQL_PASSWORD}', safe=''))")

SOURCE=$(psql "postgresql://${MLFLOW_SQL_USER}:${ENCODED_PASSWORD}@${MLFLOW_SQL_HOST}:${MLFLOW_SQL_PORT}/${MLFLOW_SQL_DATABASE}" \
    -t -c "SELECT source FROM model_versions WHERE name='${MODEL_NAME}' AND version=${MODEL_VERSION};" | xargs)

if [ -z "$SOURCE" ]; then
    echo "Error: Could not find model source in database"
    exit 1
fi

SOURCE=${SOURCE//:/}

echo "Model source: $SOURCE"

echo "Downloading model from gs://${MLFLOW_OBJECT_STORAGE_BUCKET}/1/${SOURCE}/artifacts/data/model.pth"

gcloud storage cp "gs://${MLFLOW_OBJECT_STORAGE_BUCKET}/1/${SOURCE}/artifacts/data/model.pth" best.pth

if [ ! -f best.pth ]; then
    echo "Error: Failed to download model"
    exit 1
fi

echo "Creating MAR archive..."

torch-model-archiver \
    --model-name ${MODEL_NAME} \
    --version 1.0 \
    --model-file scripts/model.py \
    --serialized-file best.pth \
    --handler scripts/handler.py \
    --force

if [ $? -eq 0 ]; then
    echo "MAR file created successfully: ${MODEL_NAME}.mar"
else
    echo "Error: Failed to create MAR file"
    rm -f best.pth
    exit 1
fi

rm -f best.pth

if [ ! -d "models/bird_classification/model-store" ]; then
    mkdir -p models/bird_classification/model-store
fi

mv ${MODEL_NAME}.mar models/bird_classification/model-store/ 2>/dev/null || true

echo "Done! MAR file is ready at models/bird_classification/model-store/${MODEL_NAME}.mar"

gcloud storage cp models/bird_classification/config/ "gs://${MODEL_STORAGE_BUCKET}/${MODEL_NAME}/v1/" --recursive
gcloud storage cp models/bird_classification/model-store/ "gs://${MODEL_STORAGE_BUCKET}/${MODEL_NAME}/v1/" --recursive
