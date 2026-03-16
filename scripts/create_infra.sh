if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Please create one from .env.example"
    exit 1
fi

echo """
######################################################################
# Create the following resources in Google cloud:                    #
#  - Bucket in cloud storage that will be used as artifact storage   #
#  - PostgreSQL DB in cloud SQL that will be used as a backend db    #
#  - Service account (and json-key) with access to GCS and cloud SQL #
#  - Container registry with the mlflow image                        #
######################################################################
"""
PROJECT_ID=$(gcloud config list --format='value(core.project)')

echo """
Enable relevant services ...
"""
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  sqladmin.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com



echo """
Create bucket ...
"""
gsutil mb gs://$BUCKET_NAME



echo """
Create backend DB ...
"""
gcloud sql instances create $SQL_INSTANCE_NAME \
  --database-version=POSTGRES_18 \
  --edition=ENTERPRISE \
  --tier=db-f1-micro \
  --storage-size=10 \
  --region=us-central1 \
  --root-password=$SQL_PWD

echo "Create mlflow_credentials directory ..."
mkdir -p ./mlflow_credentials

echo """
Create a dedicated service account for Cloud Build ...
"""
CLOUDBUILD_SA_NAME=cloudbuild-sa
CLOUDBUILD_SA_EMAIL="${CLOUDBUILD_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Check if service account already exists
if gcloud iam service-accounts describe $CLOUDBUILD_SA_EMAIL &>/dev/null; then
    echo "Service account $CLOUDBUILD_SA_EMAIL already exists"
else
    gcloud iam service-accounts create $CLOUDBUILD_SA_NAME \
        --display-name="Cloud Build Service Account" \
        --description="Service account for Cloud Build operations"
fi

echo """
Grant necessary permissions to Cloud Build service account ...
"""
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${CLOUDBUILD_SA_EMAIL}" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${CLOUDBUILD_SA_EMAIL}" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${CLOUDBUILD_SA_EMAIL}" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${CLOUDBUILD_SA_EMAIL}" \
    --role="roles/cloudbuild.builds.builder"

echo """
Create a service account with GCS access
Create a service account with cloud SQL access
Create a service account with artifact registry access
Create a key that will be stored locally in ./mlflow_credentials ...
"""
MLFLOW_SA_NAME=mlflow-account
MLFLOW_SA_EMAIL="${MLFLOW_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create ${MLFLOW_SA_NAME}

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:${MLFLOW_SA_EMAIL}" \
  --role "roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:${MLFLOW_SA_EMAIL}" \
  --role "roles/cloudsql.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member "serviceAccount:${MLFLOW_SA_EMAIL}" \
  --role "roles/artifactregistry.reader"

gcloud iam service-accounts keys create $MLFLOW_CREDENTIALS \
      --iam-account=${MLFLOW_SA_EMAIL}



echo """
Add the custom mlflow image in to the artifact registry ...
"""
gcloud artifacts repositories create $MLFLOW_REPO \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for MLflow images"

gcloud builds submit services/mlflow \
    --config=services/mlflow/cloudbuild.yaml \
    --service-account="projects/${PROJECT_ID}/serviceAccounts/${CLOUDBUILD_SA_EMAIL}" \
    --substitutions=_TAG_NAME=$TAG_NAME,_MLFLOW_REPO="$MLFLOW_REPO"
