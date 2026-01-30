# Birds Classification Service

Kiến trúc 2 service:
1. **FastAPI** - Pre/Post-processing API (với HPA autoscale)
2. **KServe** - Model serving (scale-to-zero với Knative)

```
┌─────────────────┐       ┌─────────────────────┐
│                 │       │                     │
│  Client/User    │──────▶│  FastAPI Service    │
│                 │       │  (Pre/Post-process) │
└─────────────────┘       │  - HPA autoscale    │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │                     │
                          │  KServe Model       │
                          │  (birds-model)      │
                          │  - Scale to Zero    │
                          │  - MLflow model     │
                          └─────────────────────┘
```

## Components

### 1. FastAPI Pre/Post-processing Service
- Nhận image từ client
- Preprocess: resize, normalize image → tensor
- Gọi KServe model để inference
- Postprocess: convert logits → class predictions
- Autoscale với HPA (CPU/memory based)

### 2. KServe Model Service
- Load model từ MLflow Model Registry
- Chỉ xử lý inference
- Scale to zero khi không có traffic
- Auto scale up khi có request

## Endpoints

### FastAPI Service (port 8000)
- `POST /predict` - Upload image file
- `POST /predict/base64` - JSON với base64 encoded image
- `GET /health` - Health check
- `GET /ready` - Readiness (check KServe model)
- `GET /docs` - Swagger UI

### KServe Model (port 8080)
- `POST /v1/models/birds-model:predict` - Inference
- `GET /v1/models/birds-model` - Model metadata

## Local Development

```bash
# Install dependencies
uv sync

# Run FastAPI pre/post-processing service
uv run serving

# Run KServe model server (separate terminal)
uv run kserve-model
```

## Docker Build

```bash
# FastAPI preprocessing service
docker build -f src/serving/Dockerfile -t fastapi-preprocessing:latest .

# KServe model service (loaded from MLflow)
docker build -f src/serving/Dockerfile.kserve -t birds-model:latest .
```

## Kubernetes Deployment

```bash
# 1. Deploy KServe InferenceService (scale-to-zero)
kubectl apply -f k8s/inference-service.yaml

# 2. Deploy FastAPI service
kubectl apply -f k8s/fastapi-deployment.yaml

# 3. Apply HPA for FastAPI
kubectl apply -f k8s/hpa.yaml
```

## Environment Variables

### FastAPI Service
| Variable | Description | Default |
|----------|-------------|---------|
| `KSERVE_MODEL_URL` | KServe model endpoint | `http://birds-model.default.svc.cluster.local/v1/models/birds-model:predict` |
| `CLASSES_FILE` | Path to classes.txt | `/app/services/serving/src/serving/classes.txt` |
| `PORT` | Server port | `8000` |
| `WORKERS` | Uvicorn workers | `1` |

### KServe Model Service
| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server | Required |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment | Required |
| `MODEL_URI` | MLflow model URI | `models:/BirdsClassificationModel/1` |
| `HTTP_PORT` | HTTP server port | `8080` |

## Autoscaling

### FastAPI - HPA (Horizontal Pod Autoscaler)
- Min replicas: 1
- Max replicas: 10
- CPU target: 70%
- Memory target: 80%

### KServe - Scale to Zero (Knative)
- Min replicas: 0 (scale to zero)
- Max replicas: 5
- Scale target: 10 concurrent requests
- Scale-down delay: 300 seconds

## Example Usage

```python
import requests

# Upload image file
with open("bird.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
print(response.json())

# Or use base64
import base64
with open("bird.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/predict/base64",
    json={"image": image_b64}
)
print(response.json())
```