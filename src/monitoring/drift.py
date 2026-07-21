"""
Distribution-drift monitoring for the jailbreak classifier.

Two complementary signals:
  1. Population Stability Index (PSI) on a simple, cheap-to-compute feature
     (prediction confidence score histogram) — a fast smoke alarm.
  2. Kolmogorov-Smirnov test on sentence-embedding-projected features
     (here: a lightweight TF-IDF + SVD projection standing in for a full
     embedding model, so this runs without extra GPU dependencies) between a
     reference window (training-time distribution) and the current live
     traffic window — a more sensitive, structural signal that catches new
     *attack vectors* even when overall confidence looks stable.

Run standalone as a periodic job:
    python -m src.monitoring.drift --loop
"""
import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from src.model.data_prep import load_local_jsonl
from src.monitoring.metrics import MetricsStore
from src.monitoring.logger import get_logger

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
log = get_logger("drift")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D score distributions."""
    edges = np.histogram_bin_edges(reference, bins=bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class DriftMonitor:
    def __init__(self, metrics: MetricsStore, cfg: dict):
        self.metrics = metrics
        self.cfg = cfg["monitoring"]
        self._fit_reference()

    def _fit_reference(self):
        """Fits a reference TF-IDF+SVD embedding space on the original
        training corpus, and stores reference-projection coordinates and
        reference confidence scores to compare live traffic against."""
        rows = load_local_jsonl()
        texts = [r["text"] for r in rows]

        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        tfidf = self.vectorizer.fit_transform(texts)

        n_components = min(32, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=max(n_components, 2), random_state=42)
        self.reference_embedding = self.svd.fit_transform(tfidf)

        # Use the first principal projection as a 1-D structural summary for
        # the KS test (cheap and stable; extend to multi-dimensional energy
        # distance if you need more sensitivity).
        self.reference_projection = self.reference_embedding[:, 0]

    def check(self, window_size: int) -> dict:
        df = self.metrics.recent_predictions(limit=window_size)
        if len(df) < max(30, window_size // 10):
            log.info(f"Not enough recent traffic ({len(df)} rows) to compute drift yet.")
            return {"status": "insufficient_data"}

        # Signal 1: PSI on confidence scores
        # (reference = held-out synthetic eval scores are unavailable here,
        #  so we use the first half of the window as a rolling reference —
        #  in production, freeze this at training-time validation scores.)
        scores = df["score"].values
        half = len(scores) // 2
        psi = compute_psi(scores[half:], scores[:half])

        # Signal 2: KS test on the structural embedding projection of the
        # live text vs. the fitted training-time reference.
        current_texts = df["text"].tolist()
        current_tfidf = self.vectorizer.transform(current_texts)
        current_embedding = self.svd.transform(current_tfidf)
        current_projection = current_embedding[:, 0]

        ks_stat, ks_pvalue = ks_2samp(self.reference_projection, current_projection)

        if psi >= self.cfg["psi_critical_threshold"] or ks_pvalue < self.cfg["ks_alpha"]:
            status = "critical"
        elif psi >= self.cfg["psi_warning_threshold"]:
            status = "warning"
        else:
            status = "stable"

        self.metrics.record_drift(psi=psi, ks_stat=float(ks_stat),
                                   ks_pvalue=float(ks_pvalue), status=status)

        result = {"psi": psi, "ks_stat": float(ks_stat), "ks_pvalue": float(ks_pvalue),
                  "status": status}
        log.info(f"Drift check: {result}")

        if status == "critical":
            log.warning(
                "CRITICAL DRIFT DETECTED — live traffic distribution has "
                "shifted significantly from the training distribution. "
                "This likely indicates a new attack vector/technique in the "
                "wild. Consider: (1) sampling recent low-confidence "
                "'jailbreak' predictions for human review, (2) adding "
                "reviewed examples to the training set, (3) retraining and "
                "promoting a new model version via model_registry.promote()."
            )
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="run continuously on an interval")
    args = parser.parse_args()

    cfg = load_config()
    metrics = MetricsStore(cfg["monitoring"]["sqlite_path"])
    monitor = DriftMonitor(metrics, cfg)
    window_size = cfg["monitoring"]["rolling_window_size"]
    interval = cfg["monitoring"]["drift_check_interval_s"]

    if not args.loop:
        monitor.check(window_size)
        return

    log.info(f"Starting drift monitor loop (every {interval}s)...")
    while True:
        monitor.check(window_size)
        time.sleep(interval)


if __name__ == "__main__":
    main()
