```
kubectl create secret generic mlflow-secret \
  --from-literal=postgres_user=vuniem \
  --from-literal=postgres_password=vuniem131104 \
  --from-literal=postgres_db=mlflow_db \
  --from-literal=postgres_host=192.168.100.26 \
  --from-literal=postgres_port=5432 \
  --from-literal=tracking-uri=http://mlflow-server.dev.svc.cluster.local:5000 \
  --from-file=gcs-auth=/home/lehoangvu/Project_AIDE1/mlflow_credentials/mlflow-key.json \
  -n dev \
  --dry-run=client -o yaml | kubectl apply -f -
```

kubectl create secret generic loki-gcs-service-account \
  --from-file=key.json=/home/lehoangvu/Project_AIDE1/project-2a55a756-9eba-4a36-8c6-bda0416f0109.json \
  -n monitoring
