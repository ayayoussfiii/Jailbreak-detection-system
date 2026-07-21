"""
Lightweight metrics store backed by SQLite so the dashboard and the
streaming consumer can share state without extra infrastructure.

Tracks, per prediction:
  - text, predicted label, confidence score
  - latency (ms)
  - model version that served the prediction
  - (optional) ground-truth label, if a human reviewer later labels it —
    used to compute true false-positive rate over time, versus the
    "confidence-based" proxy used for real-time estimates.
"""
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pandas as pd


class MetricsStore:
    def __init__(self, db_path: str = "monitoring/metrics.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    text TEXT NOT NULL,
                    label TEXT NOT NULL,
                    score REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    over_budget INTEGER NOT NULL DEFAULT 0,
                    ground_truth TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    psi REAL NOT NULL,
                    ks_stat REAL NOT NULL,
                    ks_pvalue REAL NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            conn.commit()

    def record_prediction(self, result: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (ts, text, label, score, latency_ms, model_version, over_budget)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (result["ts"], result["text"], result["label"], result["score"],
                 result["latency_ms"], result["model_version"], int(result["over_budget"])),
            )
            conn.commit()

    def record_drift(self, psi: float, ks_stat: float, ks_pvalue: float, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO drift_scores (ts, psi, ks_stat, ks_pvalue, status)
                   VALUES (?, ?, ?, ?, ?)""",
                (time.time(), psi, ks_stat, ks_pvalue, status),
            )
            conn.commit()

    def label_feedback(self, prediction_id: int, ground_truth: str) -> None:
        """Allows a human reviewer / labeling queue to attach ground truth
        after the fact, enabling a true (not confidence-proxy) false
        positive rate calculation."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE predictions SET ground_truth = ? WHERE id = ?",
                (ground_truth, prediction_id),
            )
            conn.commit()

    def recent_predictions(self, limit: int = 5000) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY ts DESC LIMIT ?",
                conn, params=(limit,),
            )

    def recent_drift_scores(self, limit: int = 500) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM drift_scores ORDER BY ts DESC LIMIT ?",
                conn, params=(limit,),
            )

    def summary_stats(self, window_s: Optional[float] = 3600) -> dict:
        df = self.recent_predictions(limit=20000)
        if df.empty:
            return {
                "count": 0, "detection_rate": 0.0, "false_positive_rate_proxy": 0.0,
                "p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p99_latency_ms": 0.0,
                "over_budget_rate": 0.0,
            }
        if window_s:
            cutoff = time.time() - window_s
            df = df[df["ts"] >= cutoff]
        if df.empty:
            return {
                "count": 0, "detection_rate": 0.0, "false_positive_rate_proxy": 0.0,
                "p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p99_latency_ms": 0.0,
                "over_budget_rate": 0.0,
            }

        detection_rate = (df["label"] == "jailbreak").mean()

        # Ground-truth-based FPR where labels exist, else a confidence-based
        # proxy (flags predicted-jailbreak calls made with low confidence as
        # "likely false positives" for an early warning signal).
        labeled = df.dropna(subset=["ground_truth"])
        if not labeled.empty:
            fp = ((labeled["label"] == "jailbreak") & (labeled["ground_truth"] == "benign")).sum()
            neg = (labeled["ground_truth"] == "benign").sum()
            fpr = fp / neg if neg else 0.0
        else:
            flagged = df[df["label"] == "jailbreak"]
            fpr = (flagged["score"] < 0.6).mean() if not flagged.empty else 0.0

        return {
            "count": int(len(df)),
            "detection_rate": float(detection_rate),
            "false_positive_rate_proxy": float(fpr),
            "p50_latency_ms": float(df["latency_ms"].quantile(0.50)),
            "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
            "p99_latency_ms": float(df["latency_ms"].quantile(0.99)),
            "over_budget_rate": float(df["over_budget"].mean()),
        }
