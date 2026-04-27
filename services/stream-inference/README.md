# stream-inference

Kafka/Redpanda consumer that bridges streamed inference requests to the KServe
`InferenceService`.

Flow:
1. Consume `inference-requests` topic.
2. Preprocess + POST to KServe `/v2/models/{name}/infer`.
3. Postprocess (softmax + top-3) and produce to `inference-results`.
4. Produce a structured log record to `inference-logs` for downstream
   monitoring (drift / anomaly detection / re-training storage).
