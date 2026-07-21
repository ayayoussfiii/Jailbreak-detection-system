"""
Real-time jailbreak-detection consumer.

Reads prompts off a Pulsar topic, classifies them with the hot-swappable
DistilBERT classifier, republishes verdicts to a results topic, and records
latency / detection metrics for the dashboard and drift monitor.

Run:
    python -m src.streaming.consumer
"""
import json
import time
import yaml
from pathlib import Path

import pulsar

from src.model.classifier import JailbreakClassifier
from src.streaming.model_registry import HotSwapWatcher, get_active_model_dir, DEFAULT_POINTER_FILE
from src.monitoring.metrics import MetricsStore
from src.monitoring.logger import get_logger

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
log = get_logger("consumer")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run():
    cfg = load_config()
    pulsar_cfg = cfg["pulsar"]
    model_cfg = cfg["model"]

    active_dir = get_active_model_dir(
        Path(model_cfg["active_pointer_file"]),
        fallback=None,
    )
    log.info(f"Loading initial model from {active_dir}")
    classifier = JailbreakClassifier(active_dir, max_length=model_cfg["max_length"])

    metrics = MetricsStore(cfg["monitoring"]["sqlite_path"])

    client = pulsar.Client(pulsar_cfg["service_url"])
    consumer = client.subscribe(
        pulsar_cfg["topic_in"],
        subscription_name=pulsar_cfg["subscription"],
        consumer_type=(pulsar.ConsumerType.Shared
                        if pulsar_cfg["consumer_type"] == "shared"
                        else pulsar.ConsumerType.Exclusive),
    )
    producer = client.create_producer(pulsar_cfg["topic_out"])

    latency_budget_ms = model_cfg["latency_budget_ms"]
    batch_max = pulsar_cfg["batch_max_messages"]
    batch_wait_s = pulsar_cfg["batch_max_wait_ms"] / 1000.0

    with HotSwapWatcher(classifier, pointer_file=Path(model_cfg["active_pointer_file"])):
        log.info("Consumer started. Waiting for messages...")
        while True:
            batch_msgs, batch_texts = [], []
            deadline = time.monotonic() + batch_wait_s

            # Micro-batch to amortize model forward-pass cost while staying
            # well inside the sub-50ms per-message latency budget.
            while len(batch_msgs) < batch_max and time.monotonic() < deadline:
                try:
                    msg = consumer.receive(timeout_millis=max(
                        1, int((deadline - time.monotonic()) * 1000)))
                except Exception:
                    break
                payload = json.loads(msg.data())
                batch_msgs.append(msg)
                batch_texts.append(payload["text"])

            if not batch_texts:
                continue

            verdicts = classifier.predict(batch_texts)

            for msg, verdict in zip(batch_msgs, verdicts):
                result = {
                    "text": verdict.text,
                    "label": verdict.label,
                    "score": verdict.score,
                    "latency_ms": verdict.latency_ms,
                    "model_version": verdict.model_version,
                    "over_budget": verdict.latency_ms > latency_budget_ms,
                    "ts": time.time(),
                }
                producer.send(json.dumps(result).encode("utf-8"))
                metrics.record_prediction(result)
                consumer.acknowledge(msg)

                if result["over_budget"]:
                    log.warning(
                        f"Latency budget exceeded: {verdict.latency_ms:.1f}ms "
                        f"(budget={latency_budget_ms}ms) model={verdict.model_version}"
                    )


if __name__ == "__main__":
    run()
