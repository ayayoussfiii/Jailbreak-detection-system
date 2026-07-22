"""
Real-time supervision dashboard for the jailbreak 

Run:
    streamlit run src/dashboard/app.py
"""
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.monitoring.metrics import MetricsStore
from src.dashboard import components as C

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@st.cache_resource
def get_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_metrics_store(db_path: str):
    return MetricsStore(db_path)


def main():
    cfg = get_config()
    st.set_page_config(page_title=cfg["dashboard"]["title"], layout="wide")
    st.title("🛡️ " + cfg["dashboard"]["title"])

    metrics = get_metrics_store(cfg["monitoring"]["sqlite_path"])

    with st.sidebar:
        st.header("Settings")
        window_minutes = st.slider("Rolling window (minutes)", 1, 240, 60)
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        st.caption(f"Latency budget: {cfg['model']['latency_budget_ms']}ms")
        st.caption(f"PSI warning / critical: "
                   f"{cfg['monitoring']['psi_warning_threshold']} / "
                   f"{cfg['monitoring']['psi_critical_threshold']}")

    stats = metrics.summary_stats(window_s=window_minutes * 60)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Prompts processed", f"{stats['count']:,}")
    col2.metric("Detection rate", f"{stats['detection_rate']*100:.1f}%")
    col3.metric("False positive rate*", f"{stats['false_positive_rate_proxy']*100:.1f}%")
    col4.metric("p95 latency", f"{stats['p95_latency_ms']:.1f} ms")
    col5.metric("Over-budget rate", f"{stats['over_budget_rate']*100:.1f}%")
    st.caption("*FPR uses reviewer ground truth where available, otherwise a "
               "low-confidence-flag proxy on predicted jailbreaks.")

    df = metrics.recent_predictions(limit=5000)
    if df.empty:
        st.info("No predictions recorded yet. Start the producer + consumer "
                "to see live data here.")
    else:
        cutoff = time.time() - window_minutes * 60
        df = df[df["ts"] >= cutoff]

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Detection & Latency", "Attack Vector Evolution", "Drift Monitoring", "Raw Feed"]
        )

        with tab1:
            c1, c2 = st.columns(2)
            c1.plotly_chart(C.label_breakdown(df), use_container_width=True)
            c2.plotly_chart(C.detection_rate_over_time(df), use_container_width=True)
            c3, c4 = st.columns(2)
            c3.plotly_chart(C.latency_histogram(df, cfg["model"]["latency_budget_ms"]),
                             use_container_width=True)
            c4.plotly_chart(C.latency_over_time(df, cfg["model"]["latency_budget_ms"]),
                             use_container_width=True)

        with tab2:
            st.subheader("Recent flagged jailbreak attempts")
            flagged = df[df["label"] == "jailbreak"].sort_values("ts", ascending=False)
            st.dataframe(
                flagged[["text", "score", "latency_ms", "model_version"]].head(50),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Watch this table over time: clusters of similarly-phrased "
                "new prompts here (not matching earlier flagged patterns) "
                "often signal an emerging attack technique before drift "
                "metrics fully confirm it."
            )
            st.plotly_chart(C.model_version_timeline(df), use_container_width=True)

        with tab3:
            drift_df = metrics.recent_drift_scores(limit=500)
            if drift_df.empty:
                st.info("No drift checks recorded yet. Run "
                        "`python -m src.monitoring.drift --loop` alongside the consumer.")
            else:
                latest = drift_df.iloc[0]
                status_color = {"stable": "green", "warning": "orange", "critical": "red"}
                st.markdown(
                    f"**Latest status:** :{status_color.get(latest['status'], 'gray')}"
                    f"[{latest['status'].upper()}]  "
                    f"(PSI={latest['psi']:.3f}, KS p-value={latest['ks_pvalue']:.4f})"
                )
                st.plotly_chart(
                    C.drift_score_chart(
                        drift_df,
                        cfg["monitoring"]["psi_warning_threshold"],
                        cfg["monitoring"]["psi_critical_threshold"],
                    ),
                    use_container_width=True,
                )

        with tab4:
            st.dataframe(
                df.sort_values("ts", ascending=False)
                  [["text", "label", "score", "latency_ms", "model_version", "over_budget"]]
                  .head(200),
                use_container_width=True, hide_index=True,
            )

    if auto_refresh:
        time.sleep(cfg["dashboard"]["refresh_interval_s"])
        st.rerun()


if __name__ == "__main__":
    main()
