"""
Fine-tunes DistilBERT for binary jailbreak-vs-benign classification.

Usage:
    python src/model/train.py --epochs 3 --output-dir models/v1
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset

from src.model.data_prep import load_local_jsonl, train_val_split


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall_detection_rate": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "false_positive_rate": false_positive_rate,
    }


def build_datasets(tokenizer, max_length: int):
    rows = load_local_jsonl()
    train_rows, val_rows = train_val_split(rows)

    def to_hf(rows):
        return Dataset.from_dict({
            "text": [r["text"] for r in rows],
            "label": [r["label"] for r in rows],
        })

    train_ds, val_ds = to_hf(train_rows), to_hf(val_rows)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", default="models/v1")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2,
        id2label={0: "benign", 1: "jailbreak"},
        label2id={"benign": 0, "jailbreak": 1},
    )

    train_ds, val_ds = build_datasets(tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}_ckpts",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    with (out_dir / "eval_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Clean up intermediate checkpoint dir to save disk space
    ckpt_dir = Path(f"{args.output_dir}_ckpts")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    print(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()
