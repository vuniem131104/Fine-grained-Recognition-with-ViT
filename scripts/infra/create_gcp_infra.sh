if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Please create one from .env.example"
    exit 1
fi

PROJECT_ID=$(gcloud config list --format='value(core.project)')

echo """
Enable relevant services ...
"""
gcloud services enable \
  container.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  sqladmin.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com


echo """
Create buckets ...
"""
gsutil mb gs://${LOKI_CHUNKS_BUCKET_NAME}
gsutil mb gs://${LOKI_RULER_BUCKET_NAME}
gsutil mb gs://${MLFLOW_BUCKET_NAME}
gsutil mb gs://${MODEL_BUCKET_NAME}
gsutil mb gs://${TEMPO_TRACES_BUCKET_NAME}


echo """
Create SQL instance, user and databases ...
"""
gcloud sql instances create $SQL_INSTANCE_NAME \
  --database-version=POSTGRES_18 \
  --edition=ENTERPRISE \
  --tier=db-f1-micro \
  --storage-size=10 \
  --region=us-central1 \
  --root-password=$POSTGRES_PASSWORD

gcloud sql users create $POSTGRES_USER \
  --instance=$SQL_INSTANCE_NAME \
  --password=$POSTGRES_PASSWORD

gcloud sql databases create $MLFLOW_DB \
  --instance=$SQL_INSTANCE_NAME

gcloud sql databases create $MAIN_DB \
  --instance=$SQL_INSTANCE_NAME

echo """
Create service accounts and grant permissions ...
"""

LOKI_SA_EMAIL="${LOKI_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create ${LOKI_SA_NAME} \
    --display-name="Loki Storage Service Account"

gcloud storage buckets add-iam-policy-binding gs://${LOKI_CHUNKS_BUCKET_NAME} \
    --member="serviceAccount:${LOKI_SA_EMAIL}" \
    --role="roles/storage.objectUser"

gcloud storage buckets add-iam-policy-binding gs://${LOKI_RULER_BUCKET_NAME} \
    --member="serviceAccount:${LOKI_SA_EMAIL}" \
    --role="roles/storage.objectUser"

gcloud iam service-accounts add-iam-policy-binding ${LOKI_SA_EMAIL} \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${MONITORING_NAMESPACE}/${LOKI_SA_NAME}]"

TEMPO_SA_EMAIL="${TEMPO_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create ${TEMPO_SA_NAME} \
    --display-name="Tempo Storage Service Account"

gcloud storage buckets add-iam-policy-binding gs://${TEMPO_TRACES_BUCKET_NAME} \
    --member="serviceAccount:${TEMPO_SA_EMAIL}" \
    --role="roles/storage.objectUser"

gcloud iam service-accounts add-iam-policy-binding ${TEMPO_SA_EMAIL} \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${MONITORING_NAMESPACE}/${TEMPO_SA_NAME}]"

FRONTEND_SA_EMAIL="${FRONTEND_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create ${FRONTEND_SA_NAME} \
    --display-name="Frontend Service Account"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${FRONTEND_SA_EMAIL}" \
    --role="roles/artifactregistry.reader"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${FRONTEND_SA_EMAIL}" \
    --role="roles/artifactregistry.writer"

gcloud iam service-accounts add-iam-policy-binding ${FRONTEND_SA_EMAIL} \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${PRODUCTION_NAMESPACE}/${FRONTEND_SA_NAME}]"

MLFLOW_SA_EMAIL="${MLFLOW_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create ${MLFLOW_SA_NAME} \
    --display-name="MLflow Service Account"

gcloud storage buckets add-iam-policy-binding gs://${MLFLOW_BUCKET_NAME} \
    --member="serviceAccount:${MLFLOW_SA_EMAIL}" \
    --role="roles/storage.objectUser"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${MLFLOW_SA_EMAIL}" \
    --role="roles/cloudsql.admin"

gcloud iam service-accounts keys create $MLFLOW_CREDENTIALS \
    --iam-account="${MLFLOW_SA_EMAIL}"

KSERVE_SA_EMAIL="${KSERVE_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create ${KSERVE_SA_NAME} \
    --display-name="KSERVE Service Account"

gcloud storage buckets add-iam-policy-binding gs://${MODEL_BUCKET_NAME} \
    --member="serviceAccount:${KSERVE_SA_EMAIL}" \
    --role="roles/storage.objectUser"

gcloud iam service-accounts add-iam-policy-binding ${KSERVE_SA_EMAIL} \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${PRODUCTION_NAMESPACE}/${KSERVE_SA_NAME}]"

API_SA_EMAIL="${API_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create ${API_SA_NAME} \
    --display-name="API Server Service Account"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${API_SA_EMAIL}" \
    --role="roles/artifactregistry.reader"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${API_SA_EMAIL}" \
    --role="roles/artifactregistry.writer"

gcloud iam service-accounts add-iam-policy-binding ${API_SA_EMAIL} \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${PRODUCTION_NAMESPACE}/${API_SA_NAME}]"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${API_SA_EMAIL}" \
    --role="roles/cloudsql.admin"

GITHUB_ACTIONS_SA_EMAIL="${GITHUB_ACTIONS_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create ${GITHUB_ACTIONS_SA_NAME} \
    --display-name="GitHub Actions Service Account"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${GITHUB_ACTIONS_SA_EMAIL}" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${GITHUB_ACTIONS_SA_EMAIL}" \
    --role="roles/container.clusterViewer"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${GITHUB_ACTIONS_SA_EMAIL}" \
    --role="roles/container.developer"


echo """
Create Artifact Registry repository ...
"""
gcloud artifacts repositories create $ARTIFACT_REPO \
    --repository-format=docker \
    --location=$ZONE \
    --description="Docker repository for storing application images"

echo """
Configure Docker to authenticate with Artifact Registry ...
"""
gcloud auth configure-docker ${ZONE}-docker.pkg.dev

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="user:${USER_EMAIL}@gmail.com" \
    --role="roles/artifactregistry.writer"

echo """
Create GKE cluster (Autopilot mode) ...
"""
gcloud container clusters create-auto $GKE_CLUSTER_NAME \
    --region=$ZONE \
    --project=$PROJECT_ID