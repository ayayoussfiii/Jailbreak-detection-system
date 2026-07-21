# Real-Time Jailbreak Detection System

A production-style pipeline that classifies incoming LLM prompts as **benign** or
**jailbreak attempt** in real time, using a fine-tuned DistilBERT model streamed
through Apache Pulsar, with hot-swap model reloading and a live monitoring
dashboard.

```
┌────────────┐     ┌───────────────┐     ┌──────────────────────┐     ┌─────────────┐
│  Prompt     │────▶│  Pulsar topic  │────▶│  Consumer worker      │────▶│  Pulsar      │
│  Producer   │     │  prompts.in    │     │  (DistilBERT classifier)   │  topic       │
└────────────┘     └───────────────┘     │  + hot-swap registry   │     │  results.out │
                                          └──────────┬────────────┘     └──────┬──────┘
                                                     │                          │
                                                     ▼                          ▼
                                          ┌───────────────────┐      ┌───────────────────┐
                                          │ metrics / drift     │──────▶│  Streamlit         │
                                          │ store (SQLite)      │      │  dashboard         │
                                          └───────────────────┘      └───────────────────┘
```

## Highlights

- **Model**: DistilBERT (`distilbert-base-uncased`) fine-tuned for binary
  sequence classification (`benign` vs `jailbreak`) on adversarial prompts
  patterned after **JailbreakBench** (`src/model/data_prep.py` documents how to
  swap in the real dataset via `datasets.load_dataset("JailbreakBench/JBB-Behaviors")`).
- **Streaming**: Apache Pulsar producer/consumer. The consumer batches
  messages, runs inference, and republishes verdicts + latency to a results
  topic. End-to-end inference is designed to stay **under 50ms** per prompt on
  GPU and typically 20-40ms on CPU with ONNX export (see `classifier.py`).
- **Hot-swap reloading**: `model_registry.py` watches a `MODEL_DIR` for a new
  version marker file. When a new model is promoted, the consumer atomically
  swaps the in-memory model reference between message batches — **no restart,
  no dropped messages, no downtime**.
- **Dashboard**: Streamlit app showing detection rate, false-positive rate,
  p50/p95/p99 latency, and a live "attack vector evolution" view (cluster of
  jailbreak prompts by technique, drift score over time).
- **Drift monitoring**: `drift.py` computes population stability index (PSI)
  and KS-statistics between the embedding distribution of a rolling window of
  live traffic vs. the training distribution, flags drift, and can trigger a
  retraining job.

## Quickstart

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Start Pulsar (standalone, via Docker) + the app stack
cd docker && docker compose up -d

# 3. Seed some sample traffic (benign + jailbreak prompts)
python scripts/seed_data.py

# 4. Train / fine-tune the classifier (uses synthetic data by default;
#    swap in real JailbreakBench data as noted in data_prep.py)
python src/model/train.py --epochs 3 --output-dir models/v1

# 5. Promote the model into the hot-swappable registry
python -c "from src.streaming.model_registry import promote; promote('models/v1')"

# 6. Run the streaming consumer (the real-time classifier service)
python -m src.streaming.consumer

# 7. In another terminal, run the producer to simulate incoming traffic
python -m src.streaming.producer --rate 20

# 8. Launch the live dashboard
streamlit run src/dashboard/app.py
```

## Repo layout

```
src/
  model/
    data_prep.py       # dataset loading + preprocessing (JailbreakBench-style)
    train.py           # fine-tuning script (HF Trainer)
    classifier.py       # inference wrapper, ONNX export, hot-swappable
  streaming/
    producer.py         # Pulsar producer (simulated or real traffic ingestion)
    consumer.py          # Pulsar consumer: classify + publish verdicts, <50ms budget
    model_registry.py    # hot-swap logic (file-watch + atomic pointer swap)
  monitoring/
    metrics.py           # rolling detection-rate / FP / latency stats -> SQLite
    drift.py             # PSI / KS drift scoring on embeddings
    logger.py            # structured logging helper
  dashboard/
    app.py               # Streamlit dashboard
    components.py         # chart helpers
data/
  sample_prompts.jsonl   # labeled benign/jailbreak examples for smoke testing
docker/
  docker-compose.yml     # Pulsar standalone + app services
tests/
  test_classifier.py
scripts/
  seed_data.py
  run_pulsar_standalone.sh
```

## Notes on the real JailbreakBench dataset

This repo ships a small synthetic dataset (`data/sample_prompts.jsonl`) so the
pipeline runs end-to-end offline. For production fine-tuning, swap in the real
corpus:

```python
from datasets import load_dataset
ds = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison")
```

`src/model/data_prep.py` has a `load_jailbreakbench()` function stubbed out for
this — just uncomment the `datasets` call and point it at your local cache or
the Hub.

## License

MIT — for internal security tooling / research use.
