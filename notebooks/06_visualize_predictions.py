"""
Çoklu Model Tahmin Görselleştirme
==================================

Eğitilmiş tüm modelleri (A3T-GCN, DCRNN, LSTM) yükler, test setinde tahmin yapar,
karşılaştırmalı görseller üretir:

  1. Sensör başına zaman serisi (3 modelin tahmini + gerçek aynı eksende)
  2. Horizon-bazlı MAE (mph) — 3 model
  3. En iyi modelin scatter'ı (pred vs true)

Çalıştırma (eğitim sonrası):
    .\\venv\\Scripts\\python.exe -u notebooks\\06_visualize_predictions.py
"""

import sys, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.nn.recurrent import A3TGCN2, BatchedDCRNN

from src.normalization import load_stats, inverse_transform


DATA_DIR    = "./data/metr-la"
RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"

NUM_TIMESTEPS_IN  = 12
NUM_TIMESTEPS_OUT = 12

os.makedirs(FIGURES_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# -----------------------------------------------------------------------------
# Model mimarileri (training script'leriyle aynı, sadece inference için)
# -----------------------------------------------------------------------------
def build_a3tgcn(hidden_dim, batch_size):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.tgnn = A3TGCN2(in_channels=2, out_channels=hidden_dim,
                                periods=NUM_TIMESTEPS_IN, batch_size=batch_size)
            self.head = nn.Linear(hidden_dim, NUM_TIMESTEPS_OUT)
        def forward(self, x, edge_index, edge_weight):
            h = self.tgnn(x, edge_index, edge_weight)
            return self.head(F.relu(h))
    return M().to(device)


def build_dcrnn(hidden_dim, k):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.dcrnn = BatchedDCRNN(in_channels=2, out_channels=hidden_dim, K=k)
            self.head  = nn.Linear(hidden_dim, NUM_TIMESTEPS_OUT)
        def forward(self, x, edge_index, edge_weight):
            x_seq = x.permute(0, 3, 1, 2).contiguous()  # [B, T, N, F]
            h_seq = self.dcrnn(x_seq, edge_index, edge_weight)
            h_last = h_seq[:, -1, :, :]
            return self.head(F.relu(h_last))
    return M().to(device)


def build_lstm(hidden_dim):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim,
                                num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden_dim, NUM_TIMESTEPS_OUT)
            self.hidden_dim = hidden_dim
        def forward(self, x):
            B, N, F_, T = x.shape
            x = x.permute(0, 1, 3, 2).reshape(B * N, T, F_)
            _, (h_n, _) = self.lstm(x)
            h = h_n.squeeze(0).view(B, N, self.hidden_dim)
            return self.head(h)
    return M().to(device)


# -----------------------------------------------------------------------------
# Model konfigleri (metric JSON'lardan params okur)
# -----------------------------------------------------------------------------
MODEL_CONFIGS = [
    {
        "key"    : "A3T-GCN",
        "ckpt"   : "model_a3tgcn_baseline.pt",
        "metrics": "metrics_a3tgcn_baseline.json",
        "uses_graph": True,
        "needs_edges": True,
        "build"  : lambda m: build_a3tgcn(m["hidden_dim"], m["batch_size"]),
    },
    {
        "key"    : "DCRNN",
        "ckpt"   : "model_dcrnn_baseline.pt",
        "metrics": "metrics_dcrnn_baseline.json",
        "uses_graph": True,
        "needs_edges": True,
        "build"  : lambda m: build_dcrnn(m["hidden_dim"], m.get("K_diffusion", m.get("K", 2))),
    },
    {
        "key"    : "LSTM",
        "ckpt"   : "model_vanilla_lstm.pt",
        "metrics": "metrics_vanilla_lstm.json",
        "uses_graph": False,
        "needs_edges": False,
        "build"  : lambda m: build_lstm(m["hidden_dim"]),
    },
]


# -----------------------------------------------------------------------------
# Dataset (test)
# -----------------------------------------------------------------------------
print("\nDataset yükleniyor...")
loader = METRLADatasetLoader(raw_data_dir=DATA_DIR)
dataset = loader.get_dataset(num_timesteps_in=NUM_TIMESTEPS_IN,
                              num_timesteps_out=NUM_TIMESTEPS_OUT)
all_snapshots = list(dataset)
n_total = len(all_snapshots)
test_set = all_snapshots[int(0.80 * n_total):]
print(f"Test snapshot sayısı: {len(test_set)}")

means, stds = load_stats(DATA_DIR)
SPEED_STD = float(stds[0])
print(f"Speed std (mph): {SPEED_STD:.3f}")


# -----------------------------------------------------------------------------
# Inference (her model için)
# -----------------------------------------------------------------------------
def iterate_batches(snaps, batch_size, device, needs_edges=True):
    edge_index  = snaps[0].edge_index.to(device) if needs_edges else None
    edge_weight = snaps[0].edge_attr.to(device)  if needs_edges else None
    for i in range(0, len(snaps), batch_size):
        b = snaps[i:i + batch_size]
        if len(b) < batch_size:
            break
        x = torch.stack([s.x for s in b], dim=0).to(device)
        y = torch.stack([s.y for s in b], dim=0).to(device)
        yield x, y, edge_index, edge_weight


print("\nModeller yükleniyor ve inference...")
model_outputs = {}     # key -> (preds_z, truth_z, batch_size)

for cfg in MODEL_CONFIGS:
    metrics_path = os.path.join(RESULTS_DIR, cfg["metrics"])
    ckpt_path = os.path.join(RESULTS_DIR, cfg["ckpt"])
    if not os.path.isfile(ckpt_path) or not os.path.isfile(metrics_path):
        print(f"  [{cfg['key']}] eksik, atlanıyor")
        continue

    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)

    print(f"  [{cfg['key']}] hidden={m['hidden_dim']}, batch={m['batch_size']}")
    model = cfg["build"](m)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    BATCH = m["batch_size"]
    preds, truth = [], []
    with torch.no_grad():
        for x, y, ei, ew in iterate_batches(test_set, BATCH, device, cfg["needs_edges"]):
            if cfg["needs_edges"]:
                yhat = model(x, ei, ew)
            else:
                yhat = model(x)
            preds.append(yhat.cpu().numpy())
            truth.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)   # [N_test, N_nodes, T_out]
    truth = np.concatenate(truth, axis=0)
    model_outputs[cfg["key"]] = {
        "preds_z": preds, "truth_z": truth, "metrics": m,
    }
    print(f"    -> preds shape {preds.shape}")


if not model_outputs:
    print("\nHiç model bulunamadı.")
    sys.exit(0)


# -----------------------------------------------------------------------------
# 1. Sensör başına zaman serisi (4 sensör, 3 model + gerçek)
# -----------------------------------------------------------------------------
SENSORS_TO_PLOT = [0, 50, 100, 150]
NUM_STEPS_TO_PLOT = min(300, list(model_outputs.values())[0]["preds_z"].shape[0])

fig, axes = plt.subplots(
    len(SENSORS_TO_PLOT), 1,
    figsize=(13, 3 * len(SENSORS_TO_PLOT)), sharex=True,
)

# Gerçeği herhangi bir modelden al (hepsinde aynı)
any_model = list(model_outputs.values())[0]
truth_mph = inverse_transform(any_model["truth_z"], means, stds, 0)

colors = {"A3T-GCN": "#1f77b4", "DCRNN": "#2ca02c", "LSTM": "#d62728"}

for ax, sensor_idx in zip(axes, SENSORS_TO_PLOT):
    ax.plot(truth_mph[:NUM_STEPS_TO_PLOT, sensor_idx, 0],
            label="Gerçek", color="black", linewidth=1.6)
    for key, mo in model_outputs.items():
        preds_mph = inverse_transform(mo["preds_z"], means, stds, 0)
        ax.plot(preds_mph[:NUM_STEPS_TO_PLOT, sensor_idx, 0],
                label=f"{key} (+5dk)", color=colors.get(key, "gray"),
                linewidth=1, alpha=0.85)
    ax.set_ylabel(f"Sensör #{sensor_idx}\nHız (mph)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Test örneği (5dk adımları)")
plt.suptitle("Tahmin vs Gerçek — 4 örnek sensör, 3 model", y=1.005, fontsize=13)
plt.tight_layout()
p1 = os.path.join(FIGURES_DIR, "viz_predictions_multi.png")
fig.savefig(p1, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\n[1/3] Multi-model zaman serisi: {p1}")


# -----------------------------------------------------------------------------
# 2. Horizon-bazlı MAE (mph) — 3 model overlay
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for key, mo in model_outputs.items():
    preds_mph = inverse_transform(mo["preds_z"], means, stds, 0)
    truth_mph_ = inverse_transform(mo["truth_z"], means, stds, 0)
    horizon_mae = np.abs(preds_mph - truth_mph_).mean(axis=(0, 1))
    horizons = [(h + 1) * 5 for h in range(len(horizon_mae))]
    ax.plot(horizons, horizon_mae, "o-", label=key,
            color=colors.get(key, "gray"), linewidth=2, markersize=7)
ax.set_xlabel("Tahmin horizonu (dakika)")
ax.set_ylabel("MAE (mph)")
ax.set_title("Horizon-bazlı MAE Karşılaştırması (gerçek birim, mph)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
p2 = os.path.join(FIGURES_DIR, "viz_horizon_mae_multi.png")
fig.savefig(p2, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[2/3] Horizon MAE (mph): {p2}")


# -----------------------------------------------------------------------------
# 3. Best model scatter
# -----------------------------------------------------------------------------
best_key = min(model_outputs.keys(),
               key=lambda k: model_outputs[k]["metrics"]["test_mae"])
best_mo  = model_outputs[best_key]
preds_mph = inverse_transform(best_mo["preds_z"], means, stds, 0)
truth_mph_ = inverse_transform(best_mo["truth_z"], means, stds, 0)

y_true = truth_mph_[:, :, 0].flatten()
y_pred = preds_mph[:, :, 0].flatten()

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_true, y_pred, alpha=0.04, s=4, color=colors.get(best_key, "C0"))
lo = min(y_true.min(), y_pred.min())
hi = max(y_true.max(), y_pred.max())
ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x (mükemmel)")
ax.set_xlabel("Gerçek hız (mph)")
ax.set_ylabel("Tahmin (mph)")
ax.set_title(f"{best_key} +5dk Tahmin — Scatter (en iyi model)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
p3 = os.path.join(FIGURES_DIR, "viz_scatter_best.png")
fig.savefig(p3, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[3/3] Scatter ({best_key}): {p3}")

print("\nDONE.")
