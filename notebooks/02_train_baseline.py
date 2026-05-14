"""
Baseline #1 — A3T-GCN (BATCHED, A3TGCN2)
=========================================

Her snapshot için Python loop yerine 32-snapshot batches. PyG-Temporal'in
A3TGCN2 sınıfı batched-aware (constructor'da batch_size sabit). Beklenen:
~125x hız (single-snapshot 160 ms → batched 1.3 ms/snap).

Çalıştırma:
    .\\venv\\Scripts\\python.exe -u notebooks\\02_train_baseline.py
"""

import sys, os, time, json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.nn.recurrent import A3TGCN2


# -----------------------------------------------------------------------------
# Konfig
# -----------------------------------------------------------------------------
DATA_DIR    = "./data/metr-la"
RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"

NUM_TIMESTEPS_IN  = 12
NUM_TIMESTEPS_OUT = 12
HIDDEN_DIM        = 128   # TUNE: capacity (32 -> 128)
BATCH_SIZE        = 32    # A3TGCN2 constructor'a sabit veriyoruz
LEARNING_RATE     = 1e-3
NUM_EPOCHS        = 10    # TUNE: more epochs (3 -> 10)
SEED              = 42

# QUICK_MODE — full-speed batched'la "tam" zaten dakikalar sürdüğü için
# QUICK_MODE'a normalde gerek yok; debug için bırakıyorum.
QUICK_MODE     = False
QUICK_TRAIN_N  = 1000
QUICK_VAL_N    = 200
QUICK_TEST_N   = 400

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# -----------------------------------------------------------------------------
# 1. Dataset
# -----------------------------------------------------------------------------
print("\n[1/5] Dataset yükleniyor...")
loader = METRLADatasetLoader(raw_data_dir=DATA_DIR)
dataset = loader.get_dataset(
    num_timesteps_in=NUM_TIMESTEPS_IN,
    num_timesteps_out=NUM_TIMESTEPS_OUT,
)
all_snapshots = list(dataset)
n_total = len(all_snapshots)
train_end = int(0.70 * n_total)
val_end   = int(0.80 * n_total)
train_set = all_snapshots[:train_end]
val_set   = all_snapshots[train_end:val_end]
test_set  = all_snapshots[val_end:]

if QUICK_MODE:
    train_set = train_set[:QUICK_TRAIN_N]
    val_set   = val_set[:QUICK_VAL_N]
    test_set  = test_set[:QUICK_TEST_N]
    print("      [QUICK_MODE aktif]")

print(f"      Total: {n_total}, Train: {len(train_set)}, Val: {len(val_set)}, "
      f"Test: {len(test_set)}, Batch: {BATCH_SIZE}")


# -----------------------------------------------------------------------------
# 2. Batched iterator (manuel — A3TGCN2 sabit batch ister)
# -----------------------------------------------------------------------------
def iterate_batches(snapshots, batch_size, device):
    """[B, N, F, T] x and [B, N, T_out] y batches; static edge_index/weight."""
    edge_index = snapshots[0].edge_index.to(device)
    edge_weight = snapshots[0].edge_attr.to(device) if snapshots[0].edge_attr is not None else None
    for i in range(0, len(snapshots), batch_size):
        batch = snapshots[i:i + batch_size]
        if len(batch) < batch_size:
            break  # son partial batch'i atla (A3TGCN2 sabit boy ister)
        x = torch.stack([s.x for s in batch], dim=0).to(device)
        y = torch.stack([s.y for s in batch], dim=0).to(device)
        yield x, y, edge_index, edge_weight


# -----------------------------------------------------------------------------
# 3. Model
# -----------------------------------------------------------------------------
class TrafficForecaster(nn.Module):
    """A3TGCN2 encoder + lineer head (T_out tahmini)."""
    def __init__(self):
        super().__init__()
        self.tgnn = A3TGCN2(
            in_channels=2,
            out_channels=HIDDEN_DIM,
            periods=NUM_TIMESTEPS_IN,
            batch_size=BATCH_SIZE,
        )
        self.head = nn.Linear(HIDDEN_DIM, NUM_TIMESTEPS_OUT)

    def forward(self, x, edge_index, edge_weight):
        # x: [B, N, F, T_in]
        h = self.tgnn(x, edge_index, edge_weight)   # [B, N, hidden]
        h = F.relu(h)
        return self.head(h)                          # [B, N, T_out]


model = TrafficForecaster().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n[2/5] Model: A3TGCN2 (batched, {BATCH_SIZE})  |  Parametre: {n_params:,}")


# -----------------------------------------------------------------------------
# 4. Eğitim
# -----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn   = nn.MSELoss()

train_losses, val_losses = [], []
print(f"\n[3/5] Eğitim — {NUM_EPOCHS} epoch, lr={LEARNING_RATE}")

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    epoch_loss, n_batches = 0.0, 0
    t0 = time.time()
    for x, y, ei, ew in iterate_batches(train_set, BATCH_SIZE, device):
        optimizer.zero_grad()
        y_hat = model(x, ei, ew)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    train_loss = epoch_loss / max(n_batches, 1)
    train_losses.append(train_loss)

    # Val
    model.eval()
    val_sum, n_val = 0.0, 0
    with torch.no_grad():
        for x, y, ei, ew in iterate_batches(val_set, BATCH_SIZE, device):
            y_hat = model(x, ei, ew)
            val_sum += loss_fn(y_hat, y).item()
            n_val += 1
    val_loss = val_sum / max(n_val, 1)
    val_losses.append(val_loss)

    print(f"  Epoch {epoch+1:>2}/{NUM_EPOCHS}  "
          f"train_mse={train_loss:.4f}  val_mse={val_loss:.4f}  "
          f"({time.time()-t0:.1f}s, {n_batches} train batch)")


# -----------------------------------------------------------------------------
# 5. Test
# -----------------------------------------------------------------------------
print("\n[4/5] Test seti...")
model.eval()
all_preds, all_truth = [], []
with torch.no_grad():
    for x, y, ei, ew in iterate_batches(test_set, BATCH_SIZE, device):
        y_hat = model(x, ei, ew)
        all_preds.append(y_hat.cpu().numpy())
        all_truth.append(y.cpu().numpy())

preds = np.concatenate(all_preds, axis=0)   # [N_test, N_nodes, T_out]
truth = np.concatenate(all_truth, axis=0)

mae = float(np.abs(preds - truth).mean())
rmse = float(np.sqrt(((preds - truth) ** 2).mean()))
horizon_mae = np.abs(preds - truth).mean(axis=(0, 1))   # [T_out]

print(f"      Test MAE  (z-score) : {mae:.4f}")
print(f"      Test RMSE (z-score) : {rmse:.4f}")
print("      Horizon MAE:")
for h, m in enumerate(horizon_mae):
    print(f"        +{(h+1)*5:>2} dk:  MAE={m:.4f}")


# -----------------------------------------------------------------------------
# 6. Kaydet
# -----------------------------------------------------------------------------
print("\n[5/5] Kaydediliyor...")

ckpt_path = os.path.join(RESULTS_DIR, "model_a3tgcn_baseline.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"      Model: {ckpt_path}")

metrics = {
    "model": "A3T-GCN baseline (batched)",
    "num_params": n_params,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "hidden_dim": HIDDEN_DIM,
    "batch_size": BATCH_SIZE,
    "split": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
    "train_loss_per_epoch": train_losses,
    "val_loss_per_epoch": val_losses,
    "test_mae": mae,
    "test_rmse": rmse,
    "horizon_mae": horizon_mae.tolist(),
    "device": str(device),
}
metrics_path = os.path.join(RESULTS_DIR, "metrics_a3tgcn_baseline.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"      Metrikler: {metrics_path}")

fig, ax = plt.subplots(figsize=(8, 5))
ep_x = range(1, NUM_EPOCHS + 1)
ax.plot(ep_x, train_losses, "o-", label="Train MSE")
ax.plot(ep_x, val_losses,   "s-", label="Val MSE")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (z-score)")
ax.set_title("A3T-GCN (batched) — Loss Curves (METR-LA)")
ax.legend()
ax.grid(True, alpha=0.3)
plot_path = os.path.join(FIGURES_DIR, "loss_a3tgcn_baseline.png")
fig.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"      Loss eğrisi: {plot_path}")

print("\nDONE.")
