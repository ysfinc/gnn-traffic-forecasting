"""
Trafik Kontrol Merkezi — Professional Dashboard
================================================

Sıfırdan yeniden tasarım. Linear/Stripe estetiği:
  - Açık ana tema, tek vurgu rengi (mavi)
  - Geniş whitespace, net hiyerarşi
  - Tab-tabanlı görünüm: 4 görünüm (Baseline, Müdahale, Fark, 3'lü Karşılaştırma)
  - Tek konsolide kontrol paneli
  - Detaylar collapsible

Çalıştırma:
    streamlit run simulator/app.py
"""

import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.nn.recurrent import BatchedDCRNN
from src.normalization import load_stats


# =============================================================================
# Konfig
# =============================================================================
DATA_DIR    = "./data/metr-la"
RESULTS_DIR = "./results"

TRAFFIC_STATES = {
    "🟢 Akıyor":      dict(mph=60, kmh=97),
    "🟡 Yavaşlamış":  dict(mph=45, kmh=72),
    "🟠 Çok Yavaş":   dict(mph=25, kmh=40),
    "🔴 Durmuş":      dict(mph=8,  kmh=13),
}

RED_THRESH, GREEN_THRESH = 30, 45
JUNCTION_LABELS = "ABCDEFGHIJ"

# Modern dark palet (Linear-dark / Vercel-dark vari)
C = dict(
    bg="#0a0e1a",                  # Ana arka plan (lacivert-siyah)
    card="#131825",                # Kart arka plan
    card_hover="#1a2030",
    border="#2a3045",
    border_strong="#3a4060",
    text="#e2e8f0",                # Açık metin
    text_muted="#94a3b8",
    text_subtle="#64748b",
    primary="#3b82f6",             # Mavi vurgu
    primary_dark="#2563eb",
    primary_glow="rgba(59,130,246,0.25)",
    success="#22c55e",             # Parlak yeşil (akış)
    warning="#f59e0b",             # Amber (yavaş)
    danger="#ef4444",              # Kırmızı (tıkanma)
    orange="#f97316",
    neutral="#64748b",
)

# Çizgi grafikler için: modebar gizli, sade görünüm
PLOTLY_CFG = {"displayModeBar": False, "displaylogo": False}

# Harita için: zoom kontrolü + scroll-zoom + reset
MAP_CFG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["toImage"],   # screenshot butonu lazım değil
}


# =============================================================================
# Sayfa & CSS
# =============================================================================
st.set_page_config(
    page_title="Trafik Kontrol Merkezi",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
.stApp {{ background: {C['bg']}; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stMain"] > div:first-child {{ padding: 0.5rem 1.5rem 1.5rem 1.5rem; }}
[data-testid="stSidebar"] {{ display: none; }}

h1, h2, h3, h4 {{ color: {C['text']}; font-weight: 700; }}

/* === Header bar === */
.app-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 22px; background: {C['card']};
    border: 1px solid {C['border']}; border-radius: 12px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.app-title {{
    display: flex; align-items: center; gap: 14px;
}}
.app-title-icon {{
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, {C['primary']} 0%, #6366f1 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; color: white;
}}
.app-title-text {{
    font-size: 1.25rem; font-weight: 700; color: {C['text']};
    line-height: 1.2;
}}
.app-title-sub {{
    font-size: 0.78rem; color: {C['text_muted']}; margin-top: 2px;
}}
.app-header-right {{
    display: flex; align-items: center; gap: 14px;
}}
.status-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 100px;
    background: rgba(16, 185, 129, 0.1);
    color: {C['success']}; font-size: 0.75rem; font-weight: 600;
    border: 1px solid rgba(16, 185, 129, 0.3);
}}
.status-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: {C['success']};
    box-shadow: 0 0 0 4px rgba(16,185,129,0.2);
    animation: pulse 2s infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.6; }}
}}
.tech-meta {{
    font-size: 0.72rem; color: {C['text_subtle']};
    font-family: 'SF Mono', Menlo, monospace; line-height: 1.3;
}}

/* === Card === */
.card {{
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 12px; padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.card-title {{
    font-size: 0.72rem; font-weight: 700;
    color: {C['text_muted']};
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 14px;
}}

/* === KPI strip === */
.kpi-strip {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
}}
.kpi-card {{
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 10px; padding: 14px 18px;
    transition: all 0.2s;
}}
.kpi-card:hover {{
    border-color: {C['border_strong']};
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}}
.kpi-label {{
    font-size: 0.68rem; font-weight: 600;
    color: {C['text_muted']};
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-size: 1.5rem; font-weight: 700;
    color: {C['text']};
    line-height: 1; margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
}}
.kpi-sub {{
    font-size: 0.75rem; color: {C['text_subtle']};
}}

/* === Buttons === */
.stButton > button {{
    background: {C['card']}; color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 8px; font-weight: 500;
    padding: 8px 14px; transition: all 0.15s;
    width: 100%; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.stButton > button:hover {{
    border-color: {C['primary']};
    color: {C['primary']};
    box-shadow: 0 4px 12px rgba(59,130,246,0.15);
    transform: translateY(-1px);
}}
.stButton[data-testid="baseButton-primary"] > button {{
    background: {C['primary']}; color: white;
    border-color: {C['primary']};
}}
.stButton[data-testid="baseButton-primary"] > button:hover {{
    background: {C['primary_dark']}; color: white;
}}

/* === Tabs === */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 4px; background: {C['card']};
    border: 1px solid {C['border']}; border-radius: 10px;
    padding: 4px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent; border-radius: 7px;
    color: {C['text_muted']}; font-weight: 600;
    padding: 8px 18px; border: none;
}}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {{
    background: {C['primary']}; color: white;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ display: none; }}

/* === Junction tile === */
.j-tile {{
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 10px; padding: 12px 14px;
    transition: all 0.15s;
}}
.j-tile.modified {{
    border-color: {C['primary']};
    background: linear-gradient(135deg, {C['card']} 0%, rgba(59,130,246,0.04) 100%);
    box-shadow: 0 2px 8px rgba(59,130,246,0.15);
}}
.j-letter {{
    font-size: 1.4rem; font-weight: 800;
    color: {C['primary']};
    font-family: 'SF Mono', Menlo, monospace;
    line-height: 1;
}}
.j-id {{ color: {C['text_subtle']}; font-size: 0.7rem; font-family: monospace; }}

/* === Selectbox === */
[data-testid="stSelectbox"] label {{ display: none; }}
[data-testid="stSelectbox"] > div {{ border-radius: 8px; }}

/* === Slider === */
[data-testid="stSlider"] {{ padding: 10px 0; }}

/* === Radio buttons === */
[data-testid="stRadio"] label {{ font-weight: 600; color: {C['text']}; }}

/* === Caption === */
.stCaption {{ color: {C['text_subtle']}; }}

/* === Expander === */
[data-testid="stExpander"] {{
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 12px; margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
[data-testid="stExpander"] summary {{
    color: {C['text']}; font-weight: 600;
}}
hr {{ border-color: {C['border']}; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Setup (cache)
# =============================================================================
@st.cache_resource
def load_setup():
    loader = METRLADatasetLoader(raw_data_dir=DATA_DIR)
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
    snapshots = list(dataset)
    loc_df = pd.read_csv(os.path.join(DATA_DIR, "sensor_graph", "sensor_locations.csv"))
    with open(os.path.join(DATA_DIR, "sensor_graph", "adj_mx_mapping.json")) as f:
        mapping = json.load(f)
    sensor_ids = mapping["sensor_ids"]
    loc_df["sid_str"] = loc_df["sensor_id"].astype(str)
    locs = (
        loc_df.set_index("sid_str").reindex(sensor_ids).reset_index()
        .rename(columns={"sid_str": "sensor_id_str"})
    )
    locs["adj_idx"] = range(len(locs))
    with open(os.path.join(RESULTS_DIR, "metrics_dcrnn_baseline.json")) as f:
        metrics = json.load(f)
    class DCRNNForecaster(nn.Module):
        def __init__(self, hidden_dim, K):
            super().__init__()
            self.dcrnn = BatchedDCRNN(in_channels=2, out_channels=hidden_dim, K=K)
            self.head = nn.Linear(hidden_dim, 12)
        def forward(self, x, edge_index, edge_weight):
            x_seq = x.permute(0, 3, 1, 2).contiguous()
            h_seq = self.dcrnn(x_seq, edge_index, edge_weight)
            return self.head(F.relu(h_seq[:, -1, :, :]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = metrics.get("K_diffusion", 2)
    model = DCRNNForecaster(metrics["hidden_dim"], K).to(device)
    model.load_state_dict(torch.load(
        os.path.join(RESULTS_DIR, "model_dcrnn_baseline.pt"),
        map_location=device))
    model.eval()
    means, stds = load_stats(DATA_DIR)
    return snapshots, locs, model, device, means, stds, metrics


snapshots, locs, model, device, means, stds, model_metrics = load_setup()
SPEED_MEAN = float(means[0])
SPEED_STD  = float(stds[0])


# =============================================================================
# Constants / config
# =============================================================================
PRESET_REGIONS = {
    "🏙 Şehir Merkezi": dict(lat=34.05, lon=-118.25, radius=2.5, n=5),
    "🌴 Hollywood":     dict(lat=34.10, lon=-118.32, radius=2.5, n=5),
    "🛣 I-405":          dict(lat=34.05, lon=-118.45, radius=3.0, n=5),
}


# =============================================================================
# Init session state
# =============================================================================
def init_state():
    if "region" not in st.session_state:
        st.session_state["region"] = "🏙 Şehir Merkezi"
    if "horizon" not in st.session_state:
        st.session_state["horizon"] = 30
    if "t_offset" not in st.session_state:
        st.session_state["t_offset"] = 0
    if "snapshot_idx" not in st.session_state:
        st.session_state["snapshot_idx"] = 15000

init_state()


# Compute focus area
region_name = st.session_state["region"]
region = PRESET_REGIONS[region_name]
dlat = (locs["latitude"]  - region["lat"]) * 111
dlon = (locs["longitude"] - region["lon"]) * 92
dist = np.sqrt(dlat**2 + dlon**2)
focus_indices = dist[dist < region["radius"]].sort_values().index.tolist()[:region["n"]]
focus_locs = locs.iloc[focus_indices].reset_index(drop=True)

# Reset junction states if region changed
if st.session_state.get("_last_region") != region_name:
    for idx in focus_indices:
        st.session_state[f"j_state_{idx}"] = "🟢 Akıyor"
    st.session_state["t_offset"] = 0
    st.session_state["_last_region"] = region_name

for idx in focus_indices:
    if f"j_state_{idx}" not in st.session_state:
        st.session_state[f"j_state_{idx}"] = "🟢 Akıyor"


# Scenario callbacks
def _apply(state_map):
    for i, idx in enumerate(focus_indices):
        st.session_state[f"j_state_{idx}"] = state_map.get(i, "🟢 Akıyor")
    st.session_state["t_offset"] = 0

def cb_calm():    _apply({})
def cb_rush():    _apply({i: "🟠 Çok Yavaş" for i in range(len(focus_indices))})
def cb_kaza():    _apply({0: "🔴 Durmuş"})
def cb_coklu():
    n = len(focus_indices)
    _apply({0: "🔴 Durmuş", n // 2: "🔴 Durmuş"})
def cb_reset():   _apply({})


# =============================================================================
# Tahmin
# =============================================================================
T_OFFSET_MIN = st.session_state["t_offset"]
t_offset_steps = T_OFFSET_MIN // 5
HORIZON_MIN = st.session_state["horizon"]
horizon_step = HORIZON_MIN // 5 - 1
SNAPSHOT_IDX = st.session_state["snapshot_idx"]

obs_snap_idx = min(SNAPSHOT_IDX + t_offset_steps, len(snapshots) - 1)
obs_snap = snapshots[obs_snap_idx]

x_base = obs_snap.x.clone()
x_mod  = obs_snap.x.clone()
edge_index = obs_snap.edge_index
edge_weight = obs_snap.edge_attr

for idx in focus_indices:
    state = st.session_state[f"j_state_{idx}"]
    if state == "🟢 Akıyor": continue
    target_mph = TRAFFIC_STATES[state]["mph"]
    target_z = (target_mph - SPEED_MEAN) / SPEED_STD
    for pos_anchor in [9, 10, 11]:
        pos_obs = pos_anchor - t_offset_steps
        if 0 <= pos_obs < 12:
            x_mod[idx, 0, pos_obs] = target_z

with torch.no_grad():
    ei = edge_index.to(device); ew = edge_weight.to(device)
    pred_base = model(x_base.unsqueeze(0).to(device), ei, ew)[0].cpu().numpy()
    pred_mod  = model(x_mod.unsqueeze(0).to(device),  ei, ew)[0].cpu().numpy()

pred_base_mph = pred_base * SPEED_STD + SPEED_MEAN
pred_mod_mph  = pred_mod  * SPEED_STD + SPEED_MEAN
diff_mph = pred_mod_mph - pred_base_mph

# Metrics
n_mod = sum(1 for idx in focus_indices
            if st.session_state[f"j_state_{idx}"] != "🟢 Akıyor")
predicted_red = sum(1 for idx in focus_indices
                    if pred_mod_mph[idx, horizon_step] < RED_THRESH)
predicted_red_base = sum(1 for idx in focus_indices
                          if pred_base_mph[idx, horizon_step] < RED_THRESH)
affected = sum(
    1 for idx in focus_indices
    if st.session_state[f"j_state_{idx}"] == "🟢 Akıyor"
    and abs(diff_mph[idx, horizon_step]) >= 1.0
)


# =============================================================================
# Map builder
# =============================================================================
A_full = np.zeros((207, 207))
ei_np = edge_index.numpy(); ew_np = edge_weight.numpy()
for k in range(ei_np.shape[1]):
    s, d = int(ei_np[0, k]), int(ei_np[1, k])
    A_full[s, d] = float(ew_np[k])


def signal_color(mph):
    if mph < RED_THRESH:   return C['danger']
    if mph < GREEN_THRESH: return C['warning']
    return C['success']


def diff_color(delta):
    a = abs(delta)
    if a < 0.5:    return "#475569"   # neutral gri (dark için)
    if delta < -3: return C['danger']
    if delta < -1: return C['orange']
    if delta < 0:  return "#64748b"
    if delta < 1:  return "#64748b"
    if delta < 3:  return "#4ade80"
    return C['success']


def make_map(values, color_fn, height=500, show_intervention_ring=False):
    fig = go.Figure()

    # Edges
    for fi in focus_indices:
        for fj in focus_indices:
            if fi == fj: continue
            if max(A_full[fi, fj], A_full[fj, fi]) > 0.1:
                lat_i = float(locs.iloc[fi]["latitude"]); lon_i = float(locs.iloc[fi]["longitude"])
                lat_j = float(locs.iloc[fj]["latitude"]); lon_j = float(locs.iloc[fj]["longitude"])
                fig.add_trace(go.Scattermapbox(
                    lat=[lat_i, lat_j], lon=[lon_i, lon_j],
                    mode="lines",
                    line=dict(width=2.5, color="rgba(148,163,184,0.45)"),
                    hoverinfo="skip", showlegend=False,
                ))

    # Nodes
    for i, fi in enumerate(focus_indices):
        lat = float(locs.iloc[fi]["latitude"])
        lon = float(locs.iloc[fi]["longitude"])
        label = JUNCTION_LABELS[i]
        val = values[fi]
        color = color_fn(val)
        is_mod = (show_intervention_ring
                  and st.session_state.get(f"j_state_{fi}", "🟢 Akıyor") != "🟢 Akıyor"
                  and t_offset_steps < 12)
        size = 38 if is_mod else 28
        sid = int(locs.iloc[fi]["sensor_id"])
        hover = f"<b>Kavşak {label}</b>  ·  #{sid}<br>Değer: <b>{val:+.2f}</b>"

        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon], mode="markers+text",
            marker=dict(size=size, color=color),
            text=[label],
            textfont=dict(size=15, color="white", family="Arial Black"),
            textposition="middle center",
            hoverinfo="text", hovertext=hover, showlegend=False,
        ))

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_zoom=11.5,
        mapbox_center=dict(lat=focus_locs["latitude"].mean(),
                            lon=focus_locs["longitude"].mean()),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor=C['card'], plot_bgcolor=C['card'],
    )
    return fig


# =============================================================================
# RENDER
# =============================================================================

# --- HEADER ---
st.markdown(f"""
<div class="app-header">
  <div class="app-title">
    <div class="app-title-icon">🛰</div>
    <div>
      <div class="app-title-text">Trafik Kontrol Merkezi</div>
      <div class="app-title-sub">DCRNN ile yapay zekâ tabanlı 60 dk öne trafik tahmini</div>
    </div>
  </div>
  <div class="app-header-right">
    <div class="status-pill">
      <span class="status-dot"></span> ONLINE
    </div>
    <div class="tech-meta">
      DCRNN v2 · {model_metrics['num_params']:,} params<br>
      {device.type.upper()} · METR-LA
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# --- KPI STRIP — Önce metrikler (bir bakışta sistem durumu) ---
red_delta = predicted_red - predicted_red_base
red_color = C['danger'] if predicted_red else C['success']
spread_color = C['warning'] if affected else C['success']

st.markdown(f"""
<div class="kpi-strip" style="margin-bottom: 14px;">
  <div class="kpi-card">
    <div class="kpi-label">Bölge · Sensör</div>
    <div class="kpi-value">{len(focus_indices)}</div>
    <div class="kpi-sub">{region_name}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Aktif Müdahale</div>
    <div class="kpi-value" style="color:{C['primary'] if n_mod else C['text_subtle']};">{n_mod}</div>
    <div class="kpi-sub">toplam {len(focus_indices)} kavşaktan</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">⚠ Tıkanma +{HORIZON_MIN}dk</div>
    <div class="kpi-value" style="color:{red_color};">{predicted_red}</div>
    <div class="kpi-sub">{'baseline ' + ('%+d' % red_delta) if red_delta else 'baseline ile aynı'}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">🌊 Dalga Etkisi</div>
    <div class="kpi-value" style="color:{spread_color};">{affected}</div>
    <div class="kpi-sub">etkilenen komşu</div>
  </div>
</div>
""", unsafe_allow_html=True)


# --- 3 HARİTA YAN YANA — herşey görünür, hiçbir şey gizli değil ---
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center;
            margin: 18px 0 10px 0;">
  <div style="font-size: 1.05rem; font-weight: 700; color: {C['text']};">
    🗺 Tahmin Karşılaştırması · +{HORIZON_MIN} dk öne
  </div>
  <div style="display:flex; gap:18px; align-items:center; flex-wrap:wrap;">
    <span style="display:inline-flex; align-items:center; gap:5px; font-size:0.78em; color:{C['text_muted']};">
      <span style="width:9px; height:9px; border-radius:50%; background:{C['success']}; display:inline-block;"></span> Akış
    </span>
    <span style="display:inline-flex; align-items:center; gap:5px; font-size:0.78em; color:{C['text_muted']};">
      <span style="width:9px; height:9px; border-radius:50%; background:{C['warning']}; display:inline-block;"></span> Yavaş
    </span>
    <span style="display:inline-flex; align-items:center; gap:5px; font-size:0.78em; color:{C['text_muted']};">
      <span style="width:9px; height:9px; border-radius:50%; background:{C['danger']}; display:inline-block;"></span> Tıkanma
    </span>
    <span style="color:{C['text_subtle']}; font-size:0.78em;">|</span>
    <span style="font-size:0.78em; color:{C['text_muted']};">
      Büyük halka = müdahalen
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

map_cols = st.columns(3, gap="small")

with map_cols[0]:
    st.markdown(f"""
    <div style='padding: 8px 14px; background: {C['card']}; border: 1px solid {C['border']};
                border-radius: 10px 10px 0 0; border-bottom: none;
                font-weight: 700; color: {C['text']};'>
        📊 BASELINE
        <span style="color:{C['text_subtle']}; font-weight:400; font-size:0.82em; margin-left:6px;">
            kontrol grubu (müdahalesiz)
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(
        make_map(pred_base_mph[:, horizon_step], signal_color, height=400),
        use_container_width=True, config=MAP_CFG, key="map_baseline",
    )

with map_cols[1]:
    st.markdown(f"""
    <div style='padding: 8px 14px; background: {C['card']}; border: 1px solid {C['primary']};
                border-radius: 10px 10px 0 0; border-bottom: none;
                font-weight: 700; color: {C['primary']};'>
        🎯 MÜDAHALE
        <span style="color:{C['text_subtle']}; font-weight:400; font-size:0.82em; margin-left:6px;">
            senin senaryon · +{T_OFFSET_MIN} dk sonra
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(
        make_map(pred_mod_mph[:, horizon_step], signal_color, height=400,
                 show_intervention_ring=True),
        use_container_width=True, config=MAP_CFG, key="map_intervened",
    )

with map_cols[2]:
    st.markdown(f"""
    <div style='padding: 8px 14px; background: {C['card']}; border: 1px solid {C['border']};
                border-radius: 10px 10px 0 0; border-bottom: none;
                font-weight: 700; color: {C['text']};'>
        🌊 FARK
        <span style="color:{C['text_subtle']}; font-weight:400; font-size:0.82em; margin-left:6px;">
            yayılma şiddeti (baseline ↔ müdahale)
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(
        make_map(diff_mph[:, horizon_step], diff_color, height=400,
                 show_intervention_ring=True),
        use_container_width=True, config=MAP_CFG, key="map_diff",
    )


# --- KONTROL PANELİ ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(f'<div class="card-title">KONTROL PANELİ</div>', unsafe_allow_html=True)

# Row 1: Region, Horizon, Time
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.markdown(f"<div style='font-size:0.75em; color:{C['text_muted']}; font-weight:600; "
                f"text-transform:uppercase; letter-spacing:1px;'>Bölge</div>",
                unsafe_allow_html=True)
    st.radio("Bölge", list(PRESET_REGIONS.keys()), key="region",
             label_visibility="collapsed", horizontal=True)
with c2:
    st.markdown(f"<div style='font-size:0.75em; color:{C['text_muted']}; font-weight:600; "
                f"text-transform:uppercase; letter-spacing:1px;'>Tahmin Ufku</div>",
                unsafe_allow_html=True)
    st.slider("Tahmin Ufku", 5, 60, step=5, key="horizon",
              label_visibility="collapsed")
    st.caption(f"AI **+{HORIZON_MIN} dk** sonrasını tahmin ediyor")
with c3:
    st.markdown(f"<div style='font-size:0.75em; color:{C['text_muted']}; font-weight:600; "
                f"text-transform:uppercase; letter-spacing:1px;'>Zaman İlerletme</div>",
                unsafe_allow_html=True)
    st.slider("Zaman İlerletme", 0, 60, step=5, key="t_offset",
              label_visibility="collapsed")
    if t_offset_steps == 0:
        st.caption(f"🔴 Müdahale anı (+0 dk) — tam etki")
    elif t_offset_steps < 3:
        st.caption(f"🟡 Müdahale sonrası +{T_OFFSET_MIN} dk geçti — etki azaldı")
    elif t_offset_steps < 12:
        st.caption(f"🟢 +{T_OFFSET_MIN} dk geçti — etki kaybolmak üzere")
    else:
        st.caption(f"✅ +{T_OFFSET_MIN} dk geçti — etki TAMAMEN kaybolmuş")

# Senaryolar
st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown(f"<div style='font-size:0.75em; color:{C['text_muted']}; font-weight:600; "
            f"text-transform:uppercase; letter-spacing:1px; margin-top:8px;'>Senaryo Şablonları</div>",
            unsafe_allow_html=True)
sc = st.columns(5)
sc[0].button("🌿 Sakin",          on_click=cb_calm,  use_container_width=True)
sc[1].button("🚗 Sabah Rush",     on_click=cb_rush,  use_container_width=True)
sc[2].button(f"💥 Kaza ({JUNCTION_LABELS[0]})", on_click=cb_kaza,  use_container_width=True)
sc[3].button("🚧 Çoklu Tıkanıklık", on_click=cb_coklu, use_container_width=True)
sc[4].button("↺ Reset",           on_click=cb_reset, use_container_width=True, type="primary")

# Kavşak ayarları
st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown(f"<div style='font-size:0.75em; color:{C['text_muted']}; font-weight:600; "
            f"text-transform:uppercase; letter-spacing:1px; margin-top:8px;'>"
            f"Tek Tek Kavşak Ayarı ({len(focus_indices)} kavşak)</div>",
            unsafe_allow_html=True)
jc = st.columns(len(focus_indices))
for i, idx in enumerate(focus_indices):
    label = JUNCTION_LABELS[i]
    sid = int(locs.iloc[idx]["sensor_id"])
    is_mod = st.session_state[f"j_state_{idx}"] != "🟢 Akıyor"
    with jc[i]:
        tile_cls = "j-tile modified" if is_mod else "j-tile"
        st.markdown(f"""
        <div class="{tile_cls}" style="margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="j-letter">{label}</span>
                <span class="j-id">#{sid}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.selectbox(f"j_{label}", options=list(TRAFFIC_STATES.keys()),
                     key=f"j_state_{idx}", label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)


# --- DETAYLI ANALİZ — Her zaman görünür (tek expander değil, normal blok) ---
# Region profile
horizons = [(h + 1) * 5 for h in range(12)]
if True:
    mean_base = np.mean([pred_base_mph[idx, :] for idx in focus_indices], axis=0)
    mean_mod  = np.mean([pred_mod_mph[idx, :]  for idx in focus_indices], axis=0)

    st.markdown(f"<div style='margin-top:18px; font-size: 1.05rem; font-weight: 700; "
                f"color: {C['text']};'>📈 Bölge Ortalama Hız Trajektorisi (5-60 dk)</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='color:{C['text_muted']}; font-size:0.82em; margin-bottom:8px;'>"
                f"Gri = baseline · Mavi = senin senaryon · Sarı dikey çizgi = şu an gösterilen +{HORIZON_MIN} dk</div>",
                unsafe_allow_html=True)

    fp = go.Figure()
    fp.add_trace(go.Scatter(x=horizons, y=mean_base, name="Baseline",
        line=dict(color=C['text_subtle'], width=3),
        mode="lines+markers", marker=dict(size=8, color=C['text_subtle'])))
    fp.add_trace(go.Scatter(x=horizons, y=mean_mod, name="Müdahaleli",
        line=dict(color=C['primary'], width=3),
        mode="lines+markers", marker=dict(size=8, color=C['primary'])))
    fp.add_hline(y=RED_THRESH, line_dash="dash", line_color=C['danger'])
    fp.add_hline(y=GREEN_THRESH, line_dash="dash", line_color=C['success'])
    fp.add_vline(x=HORIZON_MIN, line_dash="dot", line_color=C['warning'])
    fp.update_layout(
        paper_bgcolor=C['card'], plot_bgcolor=C['card'],
        xaxis=dict(title="Tahmin horizonu (dk)", gridcolor=C['border'],
                   color=C['text_muted'], tickmode="array", tickvals=horizons),
        yaxis=dict(title="Ort. hız (mph)", gridcolor=C['border'],
                   color=C['text_muted']),
        margin=dict(l=10, r=10, t=20, b=40), height=280,
        legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C['text'])),
        font=dict(color=C['text']),
    )
    st.plotly_chart(fp, use_container_width=True, config=PLOTLY_CFG,
                    key="region_profile_chart")

    # Sparklines
    st.markdown(f"<div style='margin-top:24px; font-size: 1.05rem; font-weight: 700; "
                f"color: {C['text']};'>📊 Kavşak Bazlı Tahmin Trajektorisi</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='color:{C['text_muted']}; font-size:0.82em; margin-bottom:8px;'>"
                f"Her kavşak için 12 horizon (5-60 dk) tahmin eğrileri</div>",
                unsafe_allow_html=True)
    fs = make_subplots(
        rows=1, cols=len(focus_indices),
        subplot_titles=[f"Kavşak {JUNCTION_LABELS[i]}" for i in range(len(focus_indices))],
        horizontal_spacing=0.04,
    )
    y_min = min(pred_base_mph[focus_indices, :].min(),
                pred_mod_mph[focus_indices, :].min()) - 3
    y_max = max(pred_base_mph[focus_indices, :].max(),
                pred_mod_mph[focus_indices, :].max()) + 3
    for i, idx in enumerate(focus_indices):
        is_mod = (st.session_state.get(f"j_state_{idx}", "🟢 Akıyor") != "🟢 Akıyor"
                  and t_offset_steps < 12)
        fs.add_trace(go.Scatter(
            x=horizons, y=pred_base_mph[idx, :], mode="lines",
            line=dict(color=C['text_subtle'], width=2),
            showlegend=(i == 0), name="Baseline",
        ), row=1, col=i+1)
        fs.add_trace(go.Scatter(
            x=horizons, y=pred_mod_mph[idx, :], mode="lines+markers",
            line=dict(color=C['primary'] if is_mod else C['success'], width=2.5),
            marker=dict(size=5),
            showlegend=(i == 0),
            name="Müdahale" if is_mod else "Etkilenen",
        ), row=1, col=i+1)
        fs.add_hline(y=RED_THRESH, line_dash="dot", line_color=C['danger'],
                     line_width=1, row=1, col=i+1)
        fs.add_hline(y=GREEN_THRESH, line_dash="dot", line_color=C['success'],
                     line_width=1, row=1, col=i+1)
    fs.update_xaxes(gridcolor=C['border'], color=C['text_muted'],
                    tickmode="array", tickvals=[5, 30, 60], tickfont=dict(size=9))
    fs.update_yaxes(gridcolor=C['border'], color=C['text_muted'],
                    range=[y_min, y_max])
    fs.update_layout(
        paper_bgcolor=C['card'], plot_bgcolor=C['card'],
        margin=dict(l=10, r=10, t=40, b=30), height=240,
        legend=dict(orientation="h", y=1.18, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=C['text'])),
        font=dict(color=C['text']),
    )
    for ann in fs.layout.annotations:
        ann.font.color = C['primary']; ann.font.size = 11; ann.font.family = "Arial Black"
    st.plotly_chart(fs, use_container_width=True, config=PLOTLY_CFG,
                    key="sparklines_chart")

    # System Analysis table
    st.markdown(f"<div style='margin-top:24px; font-size: 1.05rem; font-weight: 700; "
                f"color: {C['text']};'>📋 Sistem Analizi · +{HORIZON_MIN} dk Detay Tablosu</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='color:{C['text_muted']}; font-size:0.82em; margin-bottom:8px;'>"
                f"Müdahale ettiğin ve yayılmadan etkilenen kavşakların tahmin değişimi</div>",
                unsafe_allow_html=True)

    modified, strong, mild = [], [], []
    for i, fi in enumerate(focus_indices):
        label = JUNCTION_LABELS[i]
        bp = pred_base_mph[fi, horizon_step]
        mp = pred_mod_mph[fi, horizon_step]
        d = mp - bp
        is_mod = (st.session_state.get(f"j_state_{fi}", "🟢 Akıyor") != "🟢 Akıyor"
                  and t_offset_steps < 12)
        if is_mod: modified.append((label, bp, mp, d, "müdahale"))
        elif abs(d) >= 3.0: strong.append((label, bp, mp, d, "güçlü"))
        elif abs(d) >= 1.0: mild.append((label, bp, mp, d, "hafif"))

    if not (modified or strong or mild):
        st.info("Müdahale yok veya etki yok. Bir senaryo seç ya da kavşak durumunu değiştir.")
    else:
        rows = modified + strong + mild
        rows_html = ""
        for label, bp, mp, d, kind in rows:
            color = (C['danger'] if d < -2 else C['orange']
                     if d < 0 else C['success'])
            arrow = "↓" if d < 0 else "↑" if d > 0 else "="
            kind_color = (C['primary'] if kind == "müdahale"
                          else C['warning'] if kind == "güçlü" else C['text_muted'])
            rows_html += f"""
            <tr>
                <td style="padding:10px; font-weight:700; color:{C['primary']};">{label}</td>
                <td style="padding:10px; text-align:right; font-family:monospace;">{bp:.1f}</td>
                <td style="padding:10px; text-align:right; font-family:monospace; color:{color}; font-weight:700;">{mp:.1f}</td>
                <td style="padding:10px; text-align:right; font-family:monospace; color:{color};">{arrow} {abs(d):.2f}</td>
                <td style="padding:10px; text-align:right; color:{kind_color}; font-size:0.75em; text-transform:uppercase; letter-spacing:1px; font-weight:600;">{kind}</td>
            </tr>"""
        st.markdown(f"""
        <table style="width:100%; border-collapse:collapse;">
            <thead>
                <tr style="border-bottom: 2px solid {C['border']};">
                    <th style="text-align:left; padding:10px; color:{C['text_muted']}; font-size:0.75em; text-transform:uppercase; letter-spacing:1px;">Kavşak</th>
                    <th style="text-align:right; padding:10px; color:{C['text_muted']}; font-size:0.75em; text-transform:uppercase; letter-spacing:1px;">Baseline (mph)</th>
                    <th style="text-align:right; padding:10px; color:{C['text_muted']}; font-size:0.75em; text-transform:uppercase; letter-spacing:1px;">Müdahaleli (mph)</th>
                    <th style="text-align:right; padding:10px; color:{C['text_muted']}; font-size:0.75em; text-transform:uppercase; letter-spacing:1px;">Fark</th>
                    <th style="text-align:right; padding:10px; color:{C['text_muted']}; font-size:0.75em; text-transform:uppercase; letter-spacing:1px;">Tip</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)


with st.expander("🎓 Bu Dashboard Tezimde Ne Anlatıyor?"):
    st.markdown("""
**Üç görünüm** projenin temel mesajını özetler:

- **📊 Baseline**: AI'nın senin müdahalen olmadan ne tahmin ettiği (kontrol grubu)
- **🎯 Müdahaleli**: Senin trafik senaryona AI'nın tepkisi
- **🌊 Fark**: İkisi arasındaki delta = **DCRNN'in mekansal yayılma kanıtı**

**Zaman İlerletme** ise zamanla doğal sönmeyi gösterir: müdahale 60 dakikada AI'nın
input penceresinden tamamen çıkar, sistem baseline'a döner.

**Tezdeki sayısal sonuç:** DCRNN, 55+ dk horizon'da klasik LSTM'i geçer ve RMSE'de
hep önde — bu üç haritada gözlemlediğin yayılma deseni, o sayıların görsel kanıtıdır.
""")

with st.expander("📖 Terimler Sözlüğü"):
    st.markdown("""
- **Sensör/Kavşak**: LA otoyolundaki gerçek hız ölçer (207 adet)
- **Müdahale**: "Şu kavşakta son 15 dk şu trafik vardı" senaryosu
- **Baseline**: Hiç dokunmasaydın AI ne tahmin ederdi (kontrol grubu)
- **Fark**: Müdahaleli tahmin − Baseline tahmini
- **Tahmin Ufku**: Kaç dk sonrasına bakıyoruz (5-60)
- **Zaman İlerletme**: Müdahaleden sonra geçen gerçek zaman (0-60)
- **Dalga Etkisi**: Tek kavşaktaki olayın komşulara yayılması
""")

st.markdown(f"""
<div style="text-align:center; color:{C['text_subtle']}; font-size:0.78em;
            margin-top:20px; padding-top:14px; border-top: 1px solid {C['border']};">
    🛰 Trafik Kontrol Merkezi · DCRNN · METR-LA · <b>Yusuf İnce</b> · 2026
</div>
""", unsafe_allow_html=True)
