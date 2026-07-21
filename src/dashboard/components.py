import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def latency_histogram(df: pd.DataFrame, budget_ms: float):
    fig = px.histogram(df, x="latency_ms", nbins=40, title="Prediction Latency Distribution")
    fig.add_vline(x=budget_ms, line_dash="dash", line_color="red",
                  annotation_text=f"{budget_ms}ms budget", annotation_position="top right")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def latency_over_time(df: pd.DataFrame, budget_ms: float):
    df = df.sort_values("ts").copy()
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["latency_ms"], mode="markers",
                              marker=dict(size=4, opacity=0.5), name="latency (ms)"))
    fig.add_hline(y=budget_ms, line_dash="dash", line_color="red",
                  annotation_text="latency budget")
    fig.update_layout(title="Latency Over Time", height=320,
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


def label_breakdown(df: pd.DataFrame):
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    fig = px.pie(counts, names="label", values="count", hole=0.45,
                 title="Traffic Breakdown: Benign vs Jailbreak",
                 color="label",
                 color_discrete_map={"benign": "#2ca02c", "jailbreak": "#d62728"})
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def detection_rate_over_time(df: pd.DataFrame, bucket_s: int = 30):
    df = df.sort_values("ts").copy()
    df["bucket"] = (df["ts"] // bucket_s) * bucket_s
    df["time"] = pd.to_datetime(df["bucket"], unit="s")
    grouped = df.groupby("time").apply(
        lambda g: (g["label"] == "jailbreak").mean()
    ).reset_index(name="detection_rate")
    fig = px.line(grouped, x="time", y="detection_rate", markers=True,
                  title="Detection Rate Over Time (rolling buckets)")
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def drift_score_chart(drift_df: pd.DataFrame, psi_warn: float, psi_crit: float):
    drift_df = drift_df.sort_values("ts").copy()
    drift_df["time"] = pd.to_datetime(drift_df["ts"], unit="s")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drift_df["time"], y=drift_df["psi"], mode="lines+markers",
                              name="PSI (score distribution)"))
    fig.add_hline(y=psi_warn, line_dash="dot", line_color="orange", annotation_text="warning")
    fig.add_hline(y=psi_crit, line_dash="dash", line_color="red", annotation_text="critical")
    fig.update_layout(title="Distribution Drift (PSI) Over Time", height=320,
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


def model_version_timeline(df: pd.DataFrame):
    df = df.sort_values("ts").copy()
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    fig = px.scatter(df, x="time", y="model_version", color="model_version",
                      title="Model Version Serving Traffic (hot-swap events visible as new rows)")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig
