# Bài giảng: Streaming Inference qua Redpanda + KServe

Tài liệu này giải thích **toàn bộ** flow streaming inference vừa được thêm vào project: vì sao thiết kế thế này, từng dòng code đang làm gì, và các pattern Kafka đứng sau. Đọc xong bạn phải tự giải thích được cho người khác.

---

## 1. Bối cảnh & yêu cầu

### Yêu cầu đề bài
> Hệ thống có khả năng dự đoán trên stream (giả lập bằng cách produce message vào stream engine).
>
> i. Message từ stream engine sẽ được gửi tới Inference API được triển khai bởi KServe, có khả năng autoscale dựa trên metric req/s. Dữ liệu cần tiền xử lý trước khi predict, dữ liệu predict xong cần gửi về cho client ngay lập tức.
>
> ii. Logs tới Inference API sẽ được gửi tới một service phụ (cũng cần có khả năng autoscale) để đánh giá anomaly detection / drift detection và gửi alert về cho developer (Discord/Slack), đồng thời logs cũng sẽ được lưu trữ lại để phục vụ mục đích re-training.

### Trạng thái trước khi thêm streaming
- Frontend gửi `POST /chat` (kèm `img_b64` nếu có) tới chatbot service.
- Chatbot dùng LangGraph: `memory_management → query_enricher → tool_executor → answer_generator`.
- Trong `tool_executor`, nếu request có ảnh, nó gọi **trực tiếp** KServe HTTP (`POST /v2/.../infer`), nhận logits, postprocess (softmax + top-3), trả về.
- Sync, đơn giản, nhưng **không có stream engine**. Không thoả yêu cầu đề.

### Trạng thái sau
- Khi có ảnh, chatbot **không gọi KServe nữa**. Thay vào đó nó **produce** message vào topic `inference-requests` của Redpanda.
- Service mới `stream-inference` consume topic đó, gọi KServe, rồi produce kết quả về 2 topic: `inference-results` (chatbot nhận để trả lời user) và `inference-logs` (cho monitoring/drift sau này).
- Chatbot có 1 background task subscribe `inference-results` để khớp reply với request đang chờ.

Đây là pattern **request-reply qua message broker** — kinh điển khi muốn decouple producer khỏi worker, autoscale worker độc lập, và có audit log tự nhiên.

---

## 2. Sơ đồ tổng thể

```
┌────────────┐                                                          ┌────────────────┐
│  Frontend  │                                                          │  Monitoring    │
│ (React)    │                                                          │  service       │
└─────┬──────┘                                                          │  (sẽ làm sau)  │
      │ POST /chat (SSE)                                                └────────▲───────┘
      ▼                                                                          │
┌─────────────────────────────────────────┐                                      │
│  chatbot service (FastAPI + LangGraph)  │                                      │
│                                         │                                      │
│  memory → enricher → tool_executor      │                                      │
│                       │                 │                                      │
│                       │ produce         │                                      │
│                       ▼                 │      ┌──────────┐                    │
│              ┌────────────────┐         │      │ Redpanda │                    │
│              │ AIOKafka       │ ───────────────►          │                    │
│              │ Producer       │         │  topic:        │                    │
│              └────────────────┘         │  inference-    │                    │
│                                         │  requests       │                    │
│              ┌────────────────┐         │      │          │                    │
│              │ AIOKafka       │ ◄──────────────┤          ├────────────────────┘
│              │ Consumer       │         │      │  topic:  │
│              │ (reply loop)   │         │      │  inference-logs
│              └───────┬────────┘         │      │          │
│                      │ resolve Future   │      │          │
│                      ▼                  │      │          │
│              answer_generator → SSE     │      │          │
└──────────────────────┬──────────────────┘      │          │
                       │ stream tokens           │  topic:  │
                       ▼                         │  inference-results
                 (back to FE)                    │   ▲      │
                                                 │   │      │
                                                 └───┼──────┘
                                                     │ produce result
                                                     │
                                            ┌────────┴──────────┐
                                            │ stream-inference  │
                                            │  consumer (k8s    │
                                            │  HPA / KEDA on    │
                                            │  Kafka lag)       │
                                            └────────┬──────────┘
                                                     │ HTTP /v2/.../infer
                                                     ▼
                                            ┌───────────────────┐
                                            │ KServe Inference  │
                                            │ Service (PyTorch) │
                                            │ Knative autoscale │
                                            │ on rps            │
                                            └───────────────────┘
```

---

## 3. Vì sao chia thành 2 service?

| | chatbot | stream-inference |
|---|---|---|
| Vai trò | Orchestrate hội thoại (LLM, RAG, history) | Bridge Kafka ↔ KServe |
| Stateful? | Có (LangGraph state, conversation memory) | Không (mỗi message độc lập) |
| Scale theo? | Số user concurrent | Kafka lag |
| Crash impact | User mất hội thoại đang stream | Message chưa commit → pod khác xử lý lại |

Tách ra cho phép:
1. **Autoscale độc lập**: chatbot scale theo HTTP traffic, stream-inference scale theo lag. Không bị ràng buộc.
2. **Đa nguồn input**: simulator script (giả lập stream theo đề), hoặc sau này có thêm camera/IoT, đều produce vào cùng topic — chỉ stream-inference phải biết KServe, các producer khác không quan tâm.
3. **Bảo vệ KServe**: chatbot có thể spike hàng trăm request, nhưng stream-inference (với semaphore) ép thành dòng đều → KServe Knative thấy rps thật → autoscale chuẩn.

---

## 4. Kafka 101 (chỉ những gì cần)

Nếu bạn đã rành thì skip. Nếu chưa, đọc kỹ phần này — nó là nền tảng để hiểu code.

### 4.1 Topic, partition, offset

- **Topic** = log append-only, tên như đường dẫn (`inference-requests`).
- **Partition** = 1 topic chia thành N partition để song song hoá. Mỗi partition là 1 sequence message có thứ tự.
- **Offset** = số thứ tự message trong partition (0, 1, 2, …). Tăng monotonic, không bao giờ ghi lại.

```
topic "inference-requests" (3 partitions)
  partition 0: [m0][m1][m2][m3]...
  partition 1: [m0][m1][m2]...
  partition 2: [m0][m1][m2][m3][m4]...
```

Producer chọn partition bằng cách:
- Hash của `key` (cùng key → cùng partition → cùng order).
- Round-robin nếu không có key.

Trong code mình produce với `key=request_id.encode()` → mỗi request_id luôn vào cùng 1 partition → ordering xác định, debug dễ.

### 4.2 Consumer group

Đây là khái niệm **quan trọng nhất** để hiểu khác biệt giữa 2 consumer trong project.

- Mỗi consumer khai báo `group_id`.
- Kafka phân phối partition giữa các consumer **trong cùng group**: mỗi partition chỉ thuộc về **đúng 1** consumer trong group đó tại 1 thời điểm.
- Consumer **khác group** không ảnh hưởng nhau → mỗi group nhận **đầy đủ** mọi message.

Ví dụ topic 6 partition:

**Trường hợp A** — 3 consumer cùng group `worker`:
```
group "worker":
  consumer1  ← partition 0, 1
  consumer2  ← partition 2, 3
  consumer3  ← partition 4, 5
```
Mỗi message đi tới đúng 1 consumer. Đây là **work queue** — load balance.

**Trường hợp B** — 3 consumer, mỗi consumer 1 group riêng:
```
group "consumer1":  partition 0,1,2,3,4,5  ← consumer1 (1 mình 1 group)
group "consumer2":  partition 0,1,2,3,4,5  ← consumer2
group "consumer3":  partition 0,1,2,3,4,5  ← consumer3
```
Mỗi message đi tới **cả 3** consumer. Đây là **broadcast** — fan-out.

Project ta dùng **cả 2 pattern** cho 2 mục đích khác nhau (phần 6 sẽ giải thích).

### 4.3 Offset commit

- Consumer phải nói cho Kafka biết "tôi đã xử lý xong tới offset N rồi" — gọi là **commit**.
- Nếu consumer chết, consumer kế thừa sẽ đọc tiếp từ committed offset.
- `enable_auto_commit=True` → driver tự commit định kỳ. Nguy cơ mất message: commit xong rồi crash trước khi xử lý.
- `enable_auto_commit=False` → bạn tự gọi `consumer.commit()` sau khi xử lý xong → **at-least-once** (không mất, nhưng có thể duplicate).

### 4.4 `auto_offset_reset`

Khi consumer **lần đầu** join group (chưa có committed offset), bắt đầu đọc từ đâu?
- `earliest`: đọc từ đầu topic (mọi message đang còn trong retention).
- `latest`: chỉ đọc message mới phát sinh từ thời điểm join trở đi.

Khi đã có committed offset thì option này **không** áp dụng nữa — luôn tiếp từ committed.

### 4.5 Lag

```
lag (per partition) = log_end_offset − committed_offset
lag (per group)     = sum(lag mỗi partition)
```

Lag = "việc còn tồn đọng". Đó là metric autoscale tự nhiên cho worker:
- Lag tăng → producer nhanh hơn consumer → scale out.
- Lag ≈ 0 → đủ → scale in.

KEDA `kafka` scaler trong Kubernetes chính là query lag rồi điều chỉnh số replica.

---

## 5. Pattern: Request-Reply qua Kafka

Kafka về bản chất là pub-sub, không phải RPC. Nhưng ta có thể giả lập request-reply:

```
Client (chatbot)                         Worker (stream-inference)
─────────────────                        ─────────────────────────
1. tạo request_id (UUID)
2. tạo asyncio.Future
3. lưu pending[request_id] = future
4. produce(requests_topic,                consume(requests_topic)
   key=request_id,                        ↓
   value={request_id, payload})           xử lý
                                          ↓
                                          produce(results_topic,
                                                  key=request_id,
5. await future (timeout)                          value={request_id, result})
6. (background loop) consume(results_topic):
   payload = json.loads(msg.value)
   fut = pending.pop(payload["request_id"])
   fut.set_result(payload["result"])
7. future resolved → step 5 nhận result → return
```

Điểm chốt:
- `request_id` là **correlation id** để khớp reply về request.
- Client phải có **2 mạch Kafka**: producer (cho request) + consumer (cho reply).
- Mỗi instance client phải nhận **mọi** reply (vì reply X chỉ ý nghĩa với pod đang `await` future X) → consumer reply phải có **group_id duy nhất mỗi pod**.
- Worker thì ngược lại — phải **share công việc** → group_id cố định, cùng nhau xử lý.

---

## 6. Đi chi tiết qua code

### 6.1 `tool_executor.py` (chatbot)

File: `services/chatbot/src/chatbot/agents/tool_executor/tool_executor.py`

#### Khởi tạo

```python
def __init__(self, http_client):
    self.bootstrap = os.getenv("REDPANDA_BOOTSTRAP_SERVERS", "redpanda:29092")
    self.requests_topic = os.getenv("REDPANDA_REQUESTS_TOPIC", "inference-requests")
    self.results_topic = os.getenv("REDPANDA_RESULTS_TOPIC", "inference-results")
    self.request_timeout = float(os.getenv("INFERENCE_TIMEOUT", "30"))

    self._producer = None
    self._consumer = None
    self._consumer_task = None
    self._pending: dict[str, asyncio.Future] = {}
```

`_pending` là **bản đồ correlation**: `request_id → Future`. Đây là cốt lõi của pattern. Khi Kafka reply về với `request_id` X, ta tìm Future X trong dict này, gọi `set_result()` → coroutine đang `await` future đó được đánh thức.

#### Start (gọi 1 lần lúc app startup)

```python
async def start(self):
    self._producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap)
    await self._producer.start()

    self._consumer = AIOKafkaConsumer(
        self.results_topic,
        bootstrap_servers=self.bootstrap,
        group_id=f"chatbot-{uuid.uuid4()}",   # ★ unique mỗi pod
        auto_offset_reset="latest",            # ★ chỉ quan tâm reply mới
        enable_auto_commit=True,
    )
    await self._consumer.start()
    self._consumer_task = asyncio.create_task(self._consume_results())
```

Hai tham số đánh dấu ★ là điểm mấu chốt — đã giải thích phần 4 và 5. Nói gọn:
- `group_id=uuid4()`: mỗi pod 1 group riêng → mọi pod đều nhận đầy đủ reply.
- `auto_offset_reset="latest"`: pod mới khởi động không cần replay lịch sử reply (vì không pod nào đang chờ những reply cũ đó).

`asyncio.create_task(self._consume_results())` spawn 1 background coroutine chạy mãi, đọc reply và resolve future. Nó **không block** main event loop.

#### Background loop

```python
async def _consume_results(self):
    try:
        async for msg in self._consumer:
            payload = json.loads(msg.value)
            request_id = payload.get("request_id")
            fut = self._pending.pop(request_id, None)
            if fut and not fut.done():
                if "error" in payload:
                    fut.set_exception(RuntimeError(payload["error"]))
                else:
                    fut.set_result(payload.get("result", {}))
    except asyncio.CancelledError:
        raise
```

- `async for msg in consumer` là async generator của aiokafka — nó tự `await` lúc broker chưa có message, không nuốt CPU.
- `pop` thay vì `get` để đồng thời xoá khỏi dict, tránh memory leak.
- Phòng trường hợp future đã resolve (timeout từ phía request) thì `fut.done()` True, ta bỏ qua → tránh `InvalidStateError`.
- Nếu reply là error, ta `set_exception` → bên `await` sẽ raise.

#### Hàm chính: `_classify_via_stream`

```python
async def _classify_via_stream(self, img_b64: str) -> dict:
    request_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    self._pending[request_id] = future

    message = json.dumps({"request_id": request_id, "img_b64": img_b64}).encode()
    await self._producer.send_and_wait(
        self.requests_topic, value=message, key=request_id.encode()
    )

    try:
        return await asyncio.wait_for(future, timeout=self.request_timeout)
    except asyncio.TimeoutError:
        self._pending.pop(request_id, None)
        raise RuntimeError(...)
```

Chuỗi sự kiện:
1. Sinh `request_id` (UUID v4 → globally unique).
2. Tạo `asyncio.Future` → đối tượng "promise" của asyncio. Coroutine có thể `await` nó; coroutine khác có thể `set_result` để resolve.
3. Đăng ký vào `_pending` **trước khi** produce — đảm bảo nếu reply về siêu nhanh (race), background loop đã thấy entry trong dict.
4. `send_and_wait` produce message và đợi broker ack → có chắc message đã vào Kafka.
5. `asyncio.wait_for(future, timeout=30)` block coroutine này tới khi background loop resolve future, hoặc timeout. Lưu ý: `wait_for` chỉ block 1 coroutine, không block event loop → các request khác vẫn chạy song song.
6. Nếu timeout → pop entry để không leak, raise.

#### Tích hợp với LangGraph

```python
async def process(self, inputs: ToolExecutorInput) -> ToolExecutorOutput:
    if inputs.tool_type == ToolType.CLASSIFICATION:
        result = await self._classify_via_stream(inputs.img_b64)
    elif inputs.tool_type == ToolType.RETRIEVAL:
        result = await context_retrieve(self.http_client, inputs.query)
    ...
```

Từ góc độ LangGraph, `process` vẫn là coroutine async đơn thuần — nó chỉ `await` rồi return dict. LangGraph **không cần biết** dưới mui xe đang đi qua Kafka. Đây là bí mật của pattern: **async/await che giấu hoàn toàn** cơ chế stream.

### 6.2 Lifespan của FastAPI

File: `services/chatbot/src/chatbot/api.py`

```python
@asynccontextmanager
async def lifespan(app):
    ...
    _chatbot_service = ChatbotService(...)
    await _chatbot_service.tool_executor.start()
    yield
    ...
    await _chatbot_service.tool_executor.stop()
    await _http_client.aclose()
```

`lifespan` là hook của FastAPI: code trước `yield` chạy lúc startup, code sau `yield` chạy lúc shutdown.

`start()` mở producer + consumer + background task. `stop()` cancel task, đóng connection, cancel mọi future còn pending (tránh coroutine khác kẹt mãi).

### 6.3 `stream-inference/main.py` (worker)

File: `services/stream-inference/src/stream_inference/main.py`

#### Setup

```python
consumer = AIOKafkaConsumer(
    requests_topic,
    bootstrap_servers=bootstrap,
    group_id=group_id,                 # ★ "stream-inference" cố định
    auto_offset_reset="earliest",       # ★ không mất việc cũ
    enable_auto_commit=False,           # ★ commit thủ công sau khi xong
)
producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
```

3 dấu ★:
- `group_id` cố định → các pod cùng group → Kafka chia partition giữa pod → **work queue**, mỗi message chỉ 1 pod xử lý.
- `auto_offset_reset="earliest"` → pod mới start (chưa có committed offset) sẽ đọc từ đầu, không bỏ sót request đang chờ.
- `enable_auto_commit=False` → ta tự gọi `consumer.commit()` sau khi đã produce result + log → at-least-once.

#### Vòng main

```python
async for msg in consumer:
    await handle_message(msg, http_client, producer, ...)
    await consumer.commit()
```

Đơn giản: đọc, xử lý, commit. Tuần tự (mỗi pod tại 1 thời điểm chỉ xử lý 1 message). Nếu cần throughput cao hơn trong 1 pod, có thể chuyển sang `create_task` (xem phần 9 — limitation).

#### `handle_message`

```python
async with semaphore:
    payload = json.loads(msg.value)
    request_id = payload["request_id"]
    img_b64 = payload["img_b64"]

    started = time.time()
    try:
        result = await call_kserve(http_client, kserve_url, img_b64)
        latency_ms = (time.time() - started) * 1000

        # 1) reply cho chatbot
        await producer.send_and_wait(
            results_topic,
            value=json.dumps({"request_id": request_id, "result": result}).encode(),
            key=request_id.encode(),
        )
        # 2) log cho monitoring
        await producer.send_and_wait(
            logs_topic,
            value=json.dumps({...}).encode(),
            key=request_id.encode(),
        )
    except Exception as e:
        # gửi error về để chatbot không phải đợi timeout
        await producer.send_and_wait(
            results_topic,
            value=json.dumps({"request_id": request_id, "error": str(e)}).encode(),
            ...
        )
```

Mỗi message thành công sinh **2 produce**:
- Vào `inference-results`: cho chatbot khớp Future.
- Vào `inference-logs`: cho monitoring (sẽ làm sau) — kèm `latency_ms`, `model_version`, top-3, … đủ thông tin tính drift / anomaly / dataset retraining.

Lỗi → produce error vào `inference-results` để client không treo tới timeout.

#### `inference.py`

```python
def build_kserve_payload(img_b64):
    return {"inputs": [{"name": "input", "shape": [1],
                        "datatype": "BYTES", "data": [img_b64]}]}

def postprocess(prediction):
    probs = softmax(prediction)
    pred = int(probs.argmax())
    top3 = probs.argsort()[-4:][::-1][1:]
    return {
        "predicted_class": CLASSES[pred],
        "probability": float(probs[pred]),
        "top_3_alternatives": [...]
    }

async def call_kserve(http_client, kserve_url, img_b64):
    resp = await http_client.post(f"{kserve_url}/infer",
                                   json=build_kserve_payload(img_b64))
    resp.raise_for_status()
    logits = np.array(resp.json()["outputs"][0]["data"])
    return postprocess(logits)
```

KServe v2 protocol: input `BYTES` chứa base64 image, output là tensor logits chưa softmax. TorchServe handler bên KServe lo decode + preprocess + forward + return logits. Worker lo postprocess (softmax, top-k) — đó là phần "tiền/hậu xử lý" trong yêu cầu đề (thực tế tiền xử lý ảnh nằm trong handler của TorchServe; worker chỉ hậu xử lý).

---

## 7. Vì sao 2 consumer khác nhau? — phần đối chiếu

| Tham số | chatbot reply consumer | stream-inference worker |
|---|---|---|
| Topic | `inference-results` | `inference-requests` |
| `group_id` | `f"chatbot-{uuid4()}"` | `"stream-inference"` |
| `auto_offset_reset` | `latest` | `earliest` |
| `enable_auto_commit` | `True` | `False` |
| Mục đích | Mỗi pod nhận **mọi** reply → tự lọc theo `request_id` | Pod **chia nhau** request → mỗi message chỉ 1 pod xử lý |
| Ngữ nghĩa | Broadcast (fan-out) | Work queue |
| Khi pod restart | Bỏ qua reply cũ (vô nghĩa) | Xử lý tiếp việc chưa commit |

Đây là kiến thức **bắt buộc phải nhớ** khi làm việc với Kafka. Sai pattern là Future không bao giờ resolve hoặc message bị xử lý duplicate vô tội vạ.

---

## 8. Concurrency: semaphore và `max_inflight`

```python
semaphore = asyncio.Semaphore(max_inflight)

async with semaphore:
    result = await call_kserve(...)
```

`asyncio.Semaphore(N)` cho phép tối đa N coroutine vào critical section. Coroutine thứ N+1 chờ. Mục đích:
- **Bảo vệ KServe** khỏi flood (nếu Kafka có 1000 message dồn lại).
- **Bảo vệ memory** pod stream-inference (mỗi message giữ image base64 trong RAM).
- **Phối hợp với connection pool httpx**: `max_connections=max_inflight*2`.

Quy tắc chọn:
```
max_inflight ≈ latency_kserve_seconds × rps_target_per_pod
```
KServe ~200ms, target 40 rps/pod → `max_inflight ≈ 8` (đó là default).

---

## 9. Limitation hiện tại & cách khắc phục

### 9.1 Vòng main đang tuần tự

```python
async for msg in consumer:
    await handle_message(...)   # ← await chặn vòng lặp
    await consumer.commit()
```

Mỗi pod tại 1 thời điểm chỉ xử lý **1 message**, dù `max_inflight=8`. Semaphore hiện tại gần như không phát huy tác dụng.

**Khắc phục** (khi cần throughput cao):
```python
tasks = set()
async for msg in consumer:
    task = asyncio.create_task(handle_message(...))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    if len(tasks) >= max_inflight:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
```

Lúc này commit phải logic hơn (chỉ commit offset nhỏ hơn message đã hoàn thành liên tiếp), vì các message có thể hoàn thành không theo thứ tự.

### 9.2 Reply không mất nhưng có thể duplicate

Vì at-least-once ở phía worker: nếu produce result xong nhưng commit chưa kịp thì pod chết, pod kế sẽ xử lý lại → produce result lần 2 → chatbot pod nhận 2 reply cùng `request_id`.

Giải quyết: `pop` của ta đã idempotent — reply thứ 2 không tìm thấy entry, bỏ qua. Vô hại.

### 9.3 Reply về sai pod nếu pod chatbot scale ngang

KHÔNG xảy ra với thiết kế này — vì `group_id=uuid4()` mỗi pod, **mọi pod đều nhận đầy đủ reply**, pod nào đang chờ thì resolve, pod khác bỏ qua. Bù lại: traffic Kafka tăng tuyến tính theo số pod chatbot. Với hệ thống nhỏ thì OK; với scale lớn cần chuyển sang dùng 1 reply topic per-pod (tên topic = pod hostname) để Kafka chỉ giao reply tới đúng pod đó.

### 9.4 Future leak khi produce lỗi

Nếu `send_and_wait` raise (ví dụ broker down), code hiện tại không pop `_pending[request_id]` → leak entry. Nên bọc try/finally:

```python
self._pending[request_id] = future
try:
    await self._producer.send_and_wait(...)
    return await asyncio.wait_for(future, timeout=...)
except Exception:
    self._pending.pop(request_id, None)
    raise
```

Cải thiện cho version sau.

---

## 10. Vận hành & debug

### 10.1 Khởi động local

```bash
docker compose -f docker-compose.application.yml up -d redpanda redpanda-console
docker exec -it redpanda rpk topic create \
    inference-requests inference-results inference-logs -p 6
docker compose -f docker-compose.application.yml up -d stream-inference chatbot
```

`-p 6` = 6 partition mỗi topic → cho phép 6 pod stream-inference chạy song song.

### 10.2 Dashboard

Mở `http://localhost:8080` (Redpanda Console):
- Tab **Topics** → chọn topic → xem message vừa produce.
- Tab **Consumer Groups** → xem lag, partition assignment, committed offset.

### 10.3 CLI

```bash
docker exec -it redpanda rpk topic list
docker exec -it redpanda rpk topic consume inference-requests --num 5
docker exec -it redpanda rpk group describe stream-inference
```

`group describe` cho ra mỗi partition: `CURRENT-OFFSET`, `LOG-END-OFFSET`, `LAG`. Đó là metric mà KEDA query để autoscale.

### 10.4 Test path

1. POST ảnh tới chatbot → bật log:
   - chatbot log "Inference request produced" với `request_id`.
   - stream-inference log "Inference handled" cùng `request_id`.
   - chatbot stream answer → có nghĩa Future đã resolve.
2. Kill stream-inference giữa lúc đang xử lý → khởi động lại → message chưa commit được pick up lại (nhờ `enable_auto_commit=False`).
3. Tăng số pod stream-inference → xem partition được rebalance (Console → Consumer Groups).

---

## 11. Mapping về yêu cầu đề bài

| Yêu cầu | Phần thoả |
|---|---|
| "Predict on stream, simulate by producing into stream engine" | Redpanda + chatbot produce + simulator script (đã có spam_requests.py) |
| "Message từ stream → KServe Inference API" | stream-inference consume → call KServe |
| "KServe autoscale theo req/s" | InferenceService Knative annotations: `scaleMetric: rps` (sẽ chỉnh trong Helm) |
| "Tiền/hậu xử lý" | Tiền: TorchServe handler (decode b64, resize, normalize). Hậu: `postprocess()` softmax + top-3 trong stream-inference |
| "Gửi về client ngay lập tức" | `inference-results` topic → chatbot future → SSE về FE |
| "Logs gửi service phụ, autoscale, drift/anomaly, alert Discord/Slack, lưu re-training" | `inference-logs` topic đã có. Service `monitoring` consume topic này — **chưa làm**, là bước tiếp theo |

---

## 12. Bước tiếp theo (chưa làm trong vòng này)

1. **Service `monitoring`** consume `inference-logs`:
   - Drift: PSI / KS-test trên phân phối confidence và class so với baseline training.
   - Anomaly: spike low-confidence, latency p99 vượt ngưỡng, class collapse.
   - Alert: Discord/Slack webhook khi vi phạm threshold.
   - Persist: Postgres `inference_logs` table + raw input ref → MinIO/S3 → dataset re-training.
2. **Helm chart** cho `stream-inference` + KEDA `ScaledObject` autoscale theo Kafka lag.
3. **Sửa Helm KServe** sang `scaleMetric: rps`, set `scaleTarget` phù hợp.
4. Refactor vòng main stream-inference sang fire-and-track để semaphore thật sự hoạt động.

---

## 13. Checklist tự kiểm tra hiểu bài

Trả lời được hết là OK:

1. Vì sao chatbot dùng `group_id=uuid4()` còn stream-inference dùng group_id cố định?
2. Vì sao `auto_offset_reset` của 2 consumer khác nhau?
3. Future trong `_pending` để làm gì? Resolve bởi ai?
4. Ý nghĩa của `key=request_id.encode()` khi produce?
5. `enable_auto_commit=False` đem lại đảm bảo gì? Đánh đổi gì?
6. Lag là gì và tại sao là metric autoscale tốt cho stream-inference?
7. `max_inflight` semaphore bảo vệ ai? Vì sao trong implementation hiện tại nó gần như không có tác dụng?
8. Chatbot scale 5 pod, mỗi message reply sẽ đi vào mấy pod?
9. Nếu stream-inference produce result xong rồi crash trước khi commit, điều gì xảy ra ở chatbot?
10. Vì sao tách stream-inference khỏi chatbot thay vì gọi KServe trực tiếp như cũ?
