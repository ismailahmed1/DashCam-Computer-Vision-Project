import tempfile
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from danger_scorer import DangerScorer
from detector import DashcamDetector
from overlay import annotate_frame

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashcam Danger Analyzer",
    page_icon="🚗",
    layout="wide",
)

st.title("Dashcam Danger Analyzer")
st.caption(
    "YOLOv8 object detection · real-time danger scoring · road safety analysis"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model_option = st.selectbox(
        "YOLOv8 model",
        ["n — Nano (fastest)", "s — Small", "m — Medium (best accuracy)"],
        index=0,
        help="Larger model = better detection, slower processing",
    )
    model_key = model_option[0]

    conf_thresh = st.slider(
        "Detection confidence threshold", 0.20, 0.80, 0.40, 0.05,
        help="Lower = more detections, higher = fewer false positives",
    )

    skip_n = st.slider(
        "Process every Nth frame", 1, 8, 2,
        help="Higher = faster processing, lower temporal resolution",
    )

    st.divider()
    st.markdown("**Danger Level Key**")
    for score_range, label, hex_color in [
        ("75 – 100", "CRITICAL", "#FF2020"),
        ("50 – 74",  "HIGH",     "#FF7800"),
        ("25 – 49",  "CAUTION",  "#FFD700"),
        ("0  – 24",  "LOW",      "#3CC83C"),
    ]:
        st.markdown(
            f'<span style="color:{hex_color};font-weight:700">{label}</span>'
            f'<span style="color:#888"> — {score_range}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Scoring factors**")
    st.markdown(
        "- Object **proximity** (box size relative to frame)\n"
        "- Object **class** (pedestrians weighted 2.5×)\n"
        "- **Lateral position** (center corridor = higher risk)\n"
        "- **Vertical position** (lower = on-road = higher risk)\n"
        "- Exponential **smoothing** across frames"
    )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload dashcam footage",
    type=["mp4", "mov", "avi", "mkv"],
    help="Tip: use real dashcam near-miss footage for best demo effect",
)

if not uploaded:
    st.info("Upload a dashcam clip to get started.")
    st.stop()

# Write to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    tmp.write(uploaded.read())
    src_path = tmp.name

cap = cv2.VideoCapture(src_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

duration_s = total_frames / fps
st.info(
    f"**{uploaded.name}** · {total_frames:,} frames · "
    f"{fps:.1f} FPS · {width}×{height} · {duration_s:.1f}s"
)

run = st.button("▶  Analyze", type="primary", use_container_width=True)
if not run:
    st.stop()

# ── Processing ────────────────────────────────────────────────────────────────
detector = DashcamDetector(model_size=model_key, conf=conf_thresh)
scorer   = DangerScorer(smoothing_window=6)

out_path = src_path.replace(".mp4", "_analyzed.mp4")
fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
out_fps  = max(fps / skip_n, 1.0)
writer   = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))

progress_bar  = st.progress(0, text="Loading model…")
preview_slot  = st.empty()
status_slot   = st.empty()

cap = cv2.VideoCapture(src_path)
scores: list[float]     = []
timestamps: list[float] = []
frame_idx = 0
processed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % skip_n != 0:
        frame_idx += 1
        continue

    detections = detector.detect(frame)
    score      = scorer.score(detections, width, height)
    label, color = scorer.get_level(score)

    annotate_frame(frame, detections, score, label, color)
    writer.write(frame)

    scores.append(score)
    timestamps.append(frame_idx / fps)
    processed += 1

    # Live preview every 4 processed frames
    if processed % 4 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview_slot.image(rgb, use_container_width=True)

    pct = min(frame_idx / max(total_frames - 1, 1), 1.0)
    progress_bar.progress(pct, text=f"Frame {frame_idx}/{total_frames}  —  {label}  ({score:.0f}/100)")
    frame_idx += 1

cap.release()
writer.release()
progress_bar.progress(1.0, text="Done!")
preview_slot.empty()
status_slot.empty()

# ── Results ───────────────────────────────────────────────────────────────────
st.success("Analysis complete!")

peak   = max(scores) if scores else 0
avg    = float(np.mean(scores)) if scores else 0
crit_s = sum(1 for s in scores if s >= 75) / max(len(scores), 1) * (duration_s)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Peak Danger Score",    f"{peak:.0f} / 100")
col2.metric("Average Danger Score", f"{avg:.0f} / 100")
col3.metric("Time at CRITICAL",     f"{crit_s:.1f}s")
col4.metric("Objects Detected",     f"{processed} frames processed")

# ── Danger timeline chart ─────────────────────────────────────────────────────
fig = go.Figure()

# Shaded zones
zone_colors = [
    (75, 100, "rgba(220,30,30,0.08)"),
    (50,  75, "rgba(255,120,0,0.08)"),
    (25,  50, "rgba(255,215,0,0.08)"),
    (0,   25, "rgba(60,200,60,0.06)"),
]
for y0, y1, fill in zone_colors:
    fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, layer="below", line_width=0)

# Threshold lines
for y, dash, label_text, lc in [
    (75, "dash", "CRITICAL", "#FF2020"),
    (50, "dot",  "HIGH",     "#FF7800"),
    (25, "dot",  "CAUTION",  "#FFD700"),
]:
    fig.add_hline(
        y=y, line_dash=dash, line_color=lc, line_width=1,
        annotation_text=label_text,
        annotation_position="right",
        annotation_font_color=lc,
        annotation_font_size=11,
    )

fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=scores,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(255,75,75,0.15)",
        line=dict(color="#FF4B4B", width=2),
        name="Danger Score",
        hovertemplate="t=%{x:.1f}s<br>score=%{y:.1f}<extra></extra>",
    )
)

fig.update_layout(
    title="Danger Score Timeline",
    xaxis_title="Time (s)",
    yaxis_title="Danger Score",
    yaxis=dict(range=[0, 105]),
    height=320,
    template="plotly_dark",
    margin=dict(l=10, r=80, t=40, b=10),
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
with open(out_path, "rb") as f:
    st.download_button(
        "⬇  Download analyzed video",
        f,
        file_name="dashcam_analyzed.mp4",
        mime="video/mp4",
        use_container_width=True,
    )
