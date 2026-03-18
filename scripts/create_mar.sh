if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "Error: Cannot find file .env"
    exit 1
fi

REQUIRED_VARS=(
    MLFLOW_BUCKET_NAME
    MODEL_BUCKET_NAME
    MODEL_NAME
    MODEL_VERSION
    POSTGRES_USER
    POSTGRES_PASSWORD
    POSTGRES_HOST
    POSTGRES_PORT
    MLFLOW_DB
)

MISSING_VARS=()
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        MISSING_VARS+=("$VAR")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "Error: Missing required environment variables:"
    for VAR in "${MISSING_VARS[@]}"; do
        echo "  - $VAR"
    done
    exit 1
fi

echo "Creating MAR file for model: $MODEL_NAME"

ENCODED_PASSWORD=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${POSTGRES_PASSWORD}', safe=''))")

SOURCE=$(psql "postgresql://${POSTGRES_USER}:${ENCODED_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${MLFLOW_DB}" \
    -t -c "SELECT source FROM model_versions WHERE name='${MODEL_NAME}' AND version=${MODEL_VERSION};" | xargs)

if [ -z "$SOURCE" ]; then
    echo "Error: Could not find model source in database"
    exit 1
fi

SOURCE=${SOURCE//:/}

echo "Model source: $SOURCE"

echo "Downloading model from gs://${MLFLOW_BUCKET_NAME}/1/${SOURCE}/artifacts/data/model.pth"

gcloud storage cp "gs://${MLFLOW_BUCKET_NAME}/1/${SOURCE}/artifacts/data/model.pth" models/bird_classification/best.pth

if [ ! -f models/bird_classification/best.pth ]; then
    echo "Error: Failed to download model"
    exit 1
fi

echo "Creating MAR archive..."

torch-model-archiver \
    --model-name ${MODEL_NAME} \
    --version 1.0 \
    --model-file scripts/model.py \
    --serialized-file models/bird_classification/best.pth \
    --handler scripts/handler.py \
    --force

if [ $? -eq 0 ]; then
    echo "MAR file created successfully: ${MODEL_NAME}.mar"
else
    echo "Error: Failed to create MAR file"
    rm -f models/bird_classification/best.pth
    exit 1
fi


if [ ! -d "models/bird_classification/model-store" ]; then
    mkdir -p models/bird_classification/model-store
fi

mv ${MODEL_NAME}.mar models/bird_classification/model-store/ 2>/dev/null || true

echo "Done! MAR file is ready at models/bird_classification/model-store/${MODEL_NAME}.mar"

gcloud storage cp models/bird_classification/config/ "gs://${MODEL_BUCKET_NAME}/${MODEL_NAME}/v1/" --recursive
gcloud storage cp models/bird_classification/model-store/ "gs://${MODEL_BUCKET_NAME}/${MODEL_NAME}/v1/" --recursive