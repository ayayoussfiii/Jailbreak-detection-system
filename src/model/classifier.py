"""
Inference wrapper around the fine-tuned DistilBERT jailbreak classifier.

Designed to be:
  - fast (batches requests, optional ONNX Runtime backend for CPU latency)
  - hot-swappable (an instance can be told to reload weights from a new path
    without the caller needing to re-create the object or restart anything)
"""
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class Verdict:
    text: str
    label: str
    score: float
    latency_ms: float
    model_version: str


class JailbreakClassifier:
    """
    Thread-safe wrapper that can hot-swap its underlying model in place.

    Usage:
        clf = JailbreakClassifier("models/v1")
        verdicts = clf.predict(["some prompt", "another prompt"])
        ...
        clf.reload("models/v2")   # atomic swap, no downtime
    """

    def __init__(self, model_dir: str, device: str = None, max_length: int = 256):
        self._lock = threading.RLock()
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load(model_dir)

    def _load(self, model_dir: str) -> None:
        model_dir = str(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(self.device)
        model.eval()

        with self._lock:
            self.tokenizer = tokenizer
            self.model = model
            self.model_version = Path(model_dir).name
            self.id2label = model.config.id2label

    def reload(self, new_model_dir: str) -> None:
        """
        Hot-swap the active model. Callers already holding a reference to
        this classifier instance transparently start using the new weights
        on their *next* call to predict() — no restart, no dropped traffic.
        Loading happens outside the lock so in-flight predictions on the old
        model are not blocked; only the pointer swap is locked.
        """
        model_dir = str(new_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(self.device)
        model.eval()

        with self._lock:
            self.tokenizer = tokenizer
            self.model = model
            self.model_version = Path(model_dir).name
            self.id2label = model.config.id2label

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Verdict]:
        start = time.perf_counter()
        with self._lock:
            tokenizer, model, version, id2label = (
                self.tokenizer, self.model, self.model_version, self.id2label
            )

        enc = tokenizer(
            texts, return_tensors="pt", truncation=True,
            padding=True, max_length=self.max_length,
        ).to(self.device)

        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        scores, preds = probs.max(dim=-1)

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_item_ms = elapsed_ms / max(len(texts), 1)

        verdicts = []
        for text, pred, score in zip(texts, preds.tolist(), scores.tolist()):
            verdicts.append(Verdict(
                text=text,
                label=id2label[pred] if isinstance(id2label, dict) else id2label[str(pred)],
                score=float(score),
                latency_ms=per_item_ms,
                model_version=version,
            ))
        return verdicts

    def predict_one(self, text: str) -> Verdict:
        return self.predict([text])[0]


def export_to_onnx(model_dir: str, onnx_out_path: str, max_length: int = 256) -> None:
    """
    Optional: export the fine-tuned model to ONNX for faster CPU inference.
    Typically brings single-prompt latency well under the 50ms budget.
    """
    from transformers.onnx import export, FeaturesManager

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        model, feature="sequence-classification"
    )
    onnx_config = model_onnx_config(model.config)

    Path(onnx_out_path).parent.mkdir(parents=True, exist_ok=True)
    export(
        preprocessor=tokenizer, model=model, config=onnx_config,
        opset=14, output=Path(onnx_out_path),
    )
    print(f"Exported ONNX model to {onnx_out_path}")
