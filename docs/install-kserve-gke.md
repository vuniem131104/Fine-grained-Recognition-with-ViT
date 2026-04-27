# Cài đặt KServe trên GKE Autopilot

Hướng dẫn này mô tả cách cài KServe ở chế độ **RawDeployment** trên GKE Autopilot (không Knative, không Istio) và deploy [serving Helm chart](../k8s/helm/serving) để chatbot có thể gọi inference.

## Mục lục

1. [Yêu cầu](#1-yêu-cầu)
2. [Kiểm tra tài nguyên cluster](#2-kiểm-tra-tài-nguyên-cluster)
3. [Cài cert-manager](#3-cài-cert-manager)
4. [Cài KServe](#4-cài-kserve-rawdeployment-mode)
5. [Cài serving chart](#5-cài-serving-chart)
6. [Cấu hình chatbot gọi KServe](#6-cấu-hình-chatbot-gọi-kserve)
7. [Smoke test](#7-smoke-test)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Yêu cầu

- `kubectl` đã trỏ đúng GKE cluster (`kubectl config current-context`).
- `helm` v3.x.
- `gcloud` đăng nhập với quyền đọc/sửa IAM project.
- GSA `bird-serving-sa@<PROJECT_ID>.iam.gserviceaccount.com` đã có:
  - `roles/storage.objectUser` trên bucket model (`gs://bird-classification-model-bucket`).
  - `roles/iam.workloadIdentityUser` cho principal `<PROJECT_ID>.svc.id.goog[ai-engine/bird-serving-sa]`.
- Model đã upload sẵn lên `gs://bird-classification-model-bucket/v1/` (chứa `config/` và `model-store/<name>.mar`).

Verify nhanh:

```bash
PROJECT_ID=$(gcloud config get-value project)
gcloud iam service-accounts get-iam-policy \
  bird-serving-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --format=json | jq '.bindings'

gcloud storage buckets get-iam-policy gs://bird-classification-model-bucket \
  --format=json | grep bird-serving-sa
```

## 2. Kiểm tra tài nguyên cluster

GKE Autopilot tính tiền theo pod request. Trước khi cài, đảm bảo cluster còn chỗ cho:

| Component | CPU req | Mem req |
|---|---|---|
| cert-manager (3 pods) | ~150m | ~384Mi |
| kserve-controller-manager | ~70m | ~192Mi |
| Predictor pod (TorchServe) | 200m+ | 1Gi+ |
| **Tổng tối thiểu** | **~420m** | **~1.6Gi** |

```bash
# Tổng request hiện tại
kubectl get pods -A -o json | jq -r '
  [.items[] | select(.status.phase=="Running") |
   (.spec.containers + (.spec.initContainers // []) |
    map(.resources.requests.cpu // "0m") |
    map(if test("[0-9]+m$") then (sub("m$";"") | tonumber)
        else (tonumber * 1000) end) | add)] | add'

# Allocatable
kubectl get nodes -o custom-columns=NAME:.metadata.name,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory
```

Nếu cluster đầy mà không tăng được node (quota SSD `us-central1` thường là blocker với Autopilot), giảm requests của các app khác trước:

```bash
# Ví dụ: thu nhỏ chatbot
helm upgrade chatbot k8s/helm/chatbot -n core --reuse-values --server-side=false \
  --set chatbot.resources.requests.cpu=200m \
  --set chatbot.resources.requests.memory=384Mi \
  --set cloudSqlProxy.resources.requests.cpu=50m \
  --set cloudSqlProxy.resources.requests.memory=128Mi \
  --set autoscaling.maxReplicas=1
```

> **Lưu ý:** giảm CPU request làm HPA thấy CPU utilization % tăng → tự scale-up replica. Tạm thời cap `maxReplicas=1` cho đến khi cluster ổn định.

## 3. Cài cert-manager

KServe cần cert-manager phát Certificate cho webhook server.

```bash
helm repo add jetstack https://charts.jetstack.io
helm repo update

helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --version v1.16.2 \
  --set crds.enabled=true \
  --set 'global.leaderElection.namespace=cert-manager' \
  --set resources.requests.cpu=50m --set resources.requests.memory=128Mi \
  --set webhook.resources.requests.cpu=50m --set webhook.resources.requests.memory=128Mi \
  --set cainjector.resources.requests.cpu=50m --set cainjector.resources.requests.memory=128Mi \
  --set startupapicheck.resources.requests.cpu=20m --set startupapicheck.resources.requests.memory=64Mi
```

> **Bắt buộc trên Autopilot:** `global.leaderElection.namespace=cert-manager`. Mặc định cainjector lock leader trong `kube-system`, nhưng Autopilot block ghi vào managed namespace → cainjector không inject được CA bundle vào webhook config → mọi cert-manager API call sẽ fail với `x509: certificate signed by unknown authority`.

Verify:

```bash
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/instance=cert-manager \
  -n cert-manager --timeout=120s

# CA bundle phải có (length > 0)
kubectl get validatingwebhookconfiguration cert-manager-webhook \
  -o jsonpath='{.webhooks[0].clientConfig.caBundle}' | wc -c
```

## 4. Cài KServe (RawDeployment mode)

### Vì sao RawDeployment?

| | Serverless | RawDeployment |
|---|---|---|
| Cần Knative | Có | Không |
| Cần Istio/Kourier | Có | Không |
| Scale-to-zero | Có | Không |
| Hoạt động trên Autopilot | Khó (block DaemonSet, kube-system) | OK |

QA cluster → **RawDeployment** là pragmatic.

### 4.1 Cài CRDs

```bash
helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version v0.14.0 \
  --namespace kserve --create-namespace
```

### 4.2 Cài controller (2 phase)

Helm install thẳng sẽ fail vì chart apply `ClusterServingRuntime` *cùng lúc* với controller — webhook chưa kịp ready → admission denied. Workaround: render manifests, split, apply controller trước, đợi ready, rồi apply runtimes.

```bash
# Render
helm template kserve oci://ghcr.io/kserve/charts/kserve --version v0.14.0 \
  --namespace kserve \
  --set kserve.controller.deploymentMode=RawDeployment \
  --set kserve.modelmesh.enabled=false \
  --set kserve.controller.gateway.disableIngressCreation=true \
  --set kserve.controller.resources.requests.cpu=50m \
  --set kserve.controller.resources.requests.memory=128Mi \
  --set kserve.controller.resources.limits.cpu=200m \
  --set kserve.controller.resources.limits.memory=512Mi \
  --set kserve.controller.rbacProxy.resources.requests.cpu=20m \
  --set kserve.controller.rbacProxy.resources.requests.memory=64Mi \
  --set kserve.controller.rbacProxy.resources.limits.cpu=100m \
  --set kserve.controller.rbacProxy.resources.limits.memory=128Mi \
  > /tmp/kserve-all.yaml

# Split
python3 - <<'EOF'
import yaml
docs = list(yaml.safe_load_all(open('/tmp/kserve-all.yaml')))
ctrl = [d for d in docs if d and d.get('kind') not in ('ClusterServingRuntime','ClusterStorageContainer')]
rt   = [d for d in docs if d and d.get('kind')     in ('ClusterServingRuntime','ClusterStorageContainer')]
yaml.safe_dump_all(ctrl, open('/tmp/kserve-controller.yaml','w'))
yaml.safe_dump_all(rt,   open('/tmp/kserve-runtimes.yaml','w'))
print(f"controller={len(ctrl)} runtimes={len(rt)}")
EOF

# Apply controller (chú ý -n kserve để namespaced resources vào đúng namespace)
kubectl apply -f /tmp/kserve-controller.yaml -n kserve

# Đợi controller ready
kubectl wait --for=condition=Ready pod \
  -l control-plane=kserve-controller-manager -n kserve --timeout=180s

# Apply runtimes
kubectl apply -f /tmp/kserve-runtimes.yaml
```

> **Lý do `-n kserve`:** `helm template` không nhúng `metadata.namespace` vào ServiceAccount/Role/etc. Nếu `kubectl apply` không có `-n`, resources sẽ vào `default` (hoặc namespace context hiện tại) → controller pod báo `serviceaccount kserve/kserve-controller-manager not found`.

Verify:

```bash
kubectl get pods -n kserve
# kserve-controller-manager-xxx   2/2   Running

kubectl get clusterservingruntime
# 10 runtimes (kserve-torchserve, kserve-sklearnserver, ...)
```

## 5. Cài serving chart

Chart [`k8s/helm/serving`](../k8s/helm/serving) tạo `InferenceService` cho model bird classifier.

### 5.1 Sửa template để hỗ trợ RawDeployment

[`k8s/helm/serving/templates/inference-service.yaml`](../k8s/helm/serving/templates/inference-service.yaml) hard-code `scaleMetric: concurrency` — đó là metric Knative-only. Trong RawDeployment chỉ chấp nhận `cpu` / `memory`. Đổi sang configurable:

```yaml
scaleMetric: {{ .Values.autoscaling.scaleMetric | default "concurrency" }}
```

### 5.2 Install

```bash
helm install serving k8s/helm/serving -n ai-engine \
  --set autoscaling.scaleMetric=cpu \
  --set autoscaling.scaleTarget=80 \
  --set autoscaling.maxReplicas=1 \
  --set resources.requests.cpu=200m \
  --set resources.requests.memory=1Gi \
  --set resources.limits.cpu=500m \
  --set resources.limits.memory=1500Mi
```

Đợi predictor pull model từ GCS và load vào TorchServe (~2-3 phút):

```bash
kubectl get inferenceservice -n ai-engine -w
# READY=True khi xong

kubectl get pods -n ai-engine
# bird-classification-model-predictor-xxx   1/1   Running
```

## 6. Cấu hình chatbot gọi KServe

Service KServe tạo ra trong RawDeployment có tên `<modelName>-predictor` trong namespace `ai-engine`.

Sửa env `KSERVE_MODEL_URL` trong [`k8s/helm/chatbot/templates/deployment.yaml`](../k8s/helm/chatbot/templates/deployment.yaml):

```yaml
- name: KSERVE_MODEL_URL
  value: http://bird-classification-model-predictor.ai-engine.svc.cluster.local/v2/models/bird_classification
```

Lưu ý quan trọng:

- **Tên model** trong path `/v2/models/<name>` phải khớp tên file `.mar` upload lên GCS (ở đây là `bird_classification`, có dấu `_`), **không phải** tên `InferenceService` (`bird-classification-model`).
- Code [`services/chatbot/src/chatbot/tools/predict.py`](../services/chatbot/src/chatbot/tools/predict.py) đã append `/infer` ở cuối, nên `KSERVE_MODEL_URL` chỉ cần đến `/v2/models/<name>`.

Tên model thực tế:

```bash
kubectl exec -n core deploy/bird-chatbot -c bird-chatbot -- \
  curl -s http://bird-classification-model-predictor.ai-engine.svc.cluster.local/v2/models
# {"models":["bird_classification"]}
```

Apply:

```bash
helm upgrade chatbot k8s/helm/chatbot -n core --reuse-values --server-side=false
```

## 7. Smoke test

Test v2 metadata:

```bash
kubectl exec -n core deploy/bird-chatbot -c bird-chatbot -- \
  curl -s http://bird-classification-model-predictor.ai-engine.svc.cluster.local/v2/models/bird_classification
# {"name":"bird_classification","versions":null,"platform":"","inputs":[],"outputs":[]}
```

Test inference với 1 ảnh PNG nhỏ (base64):

```bash
B64=$(printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa1m\x07\xa6\x00\x00\x00\x00IEND\xaeB`\x82' | base64 -w0)

kubectl exec -n core deploy/bird-chatbot -c bird-chatbot -- \
  curl -s -X POST -H "Content-Type: application/json" \
    -d "{\"inputs\":[{\"name\":\"input\",\"shape\":[1],\"datatype\":\"BYTES\",\"data\":[\"$B64\"]}]}" \
    http://bird-classification-model-predictor.ai-engine.svc.cluster.local/v2/models/bird_classification/infer
# {"model_name":"bird_classification", "outputs":[{"shape":[200],"datatype":"FP64","data":[...]}]}
```

`outputs[0].shape == [200]` → đúng số class CUB-200.

## 8. Troubleshooting

### `x509: certificate signed by unknown authority` khi gọi cert-manager webhook

cainjector chưa populate CA bundle. Check:

```bash
kubectl logs -n cert-manager deploy/cert-manager-cainjector | grep -i "leader\|forbidden"
```

Nếu thấy `cannot create resource "leases" in API group "coordination.k8s.io" in the namespace "kube-system"` → set `global.leaderElection.namespace=cert-manager` (mục 3).

### `[concurrency] is not a supported metric`

Bạn đang ở RawDeployment mode nhưng InferenceService dùng `scaleMetric: concurrency`. Đổi sang `cpu` hoặc `memory` (mục 5.1).

### Predictor pod `Pending` mãi

```bash
kubectl describe pod -n ai-engine -l serving.kserve.io/inferenceservice=bird-classification-model | tail -20
```

- `Insufficient cpu` → giảm requests của các app khác (mục 2).
- `Node scale up ... GCE quota exceeded` → check quota: `gcloud compute regions describe us-central1 --format='value(quotas)' | tr ';' '\n' | grep -i ssd`. Autopilot node mới cần ~100GB SSD.

### `Model with name X does not exist`

Tên model trong URL không khớp tên đã load. List ra:

```bash
kubectl exec -n core deploy/bird-chatbot -c bird-chatbot -- \
  curl -s http://bird-classification-model-predictor.ai-engine.svc.cluster.local/v2/models
```

Tên model = tên file `.mar` (không có đuôi) trong `gs://<bucket>/v1/model-store/`.

### Storage initializer fail (không pull được model)

```bash
kubectl logs -n ai-engine -l serving.kserve.io/inferenceservice=bird-classification-model -c storage-initializer
```

- `403 Forbidden` → GSA chưa có `storage.objectUser` trên bucket.
- `unable to find a service account token` → KSA chưa annotate đúng GSA, hoặc Workload Identity binding thiếu. Check:

```bash
kubectl get sa bird-serving-sa -n ai-engine -o yaml | grep gcp-service-account

gcloud iam service-accounts get-iam-policy \
  bird-serving-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --format=json | jq '.bindings'
```

### Helm install KServe fail "release name still in use"

```bash
helm uninstall kserve -n kserve
helm uninstall kserve-crd -n kserve   # chỉ khi cần xoá CRD (cẩn thận: xoá mọi InferenceService)
```

## Tham khảo

- [KServe RawDeployment docs](https://kserve.github.io/website/master/admin/kubernetes_deployment/)
- [KServe v2 inference protocol](https://kserve.github.io/website/master/modelserving/data_plane/v2_protocol/)
- [GKE Autopilot resource requests](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-resource-requests)
- [GKE Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity)
