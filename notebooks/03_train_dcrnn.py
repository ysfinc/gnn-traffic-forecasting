"""
Baseline #2 — DCRNN (BATCHED, BatchedDCRNN)
============================================

PyG-Temporal'in seq-to-seq batched DCRNN'i: input [B, T, N, F], output [B, T, N, hidden].
Son hidden state'i alıp lineer head'le T_out tahminine çevriliyor.

Çalıştırma:
    .\\venv\\Scripts\\python.exe -u notebooks\\03_train_dcrnn.py
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
from torch_geometric_temporal.nn.recurrent import BatchedDCRNN


# -----------------------------------------------------------------------------
DATA_DIR    = "./data/metr-la"
RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"

NUM_TIMESTEPS_IN  = 12
NUM_TIMESTEPS_OUT = 12
HIDDEN_DIM        = 128   # TUNE: capacity (32 -> 128)
BATCH_SIZE        = 32
LEARNING_RATE     = 1e-3
NUM_EPOCHS        = 10    # TUNE: more epochs (3 -> 10)
SEED              = 42
K_DIFFUSION       = 2     # TUNE: lower diffusion step (3 -> 2, over-smoothing'e karşı)

QUICK_MODE     = False
QUICK_TRAIN_N  = 1000
QUICK_VAL_N    = 200
QUICK_TEST_N   = 400

os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# -----------------------------------------------------------------------------
print("\n[1/5] Dataset yükleniyor...")
loader = METRLADatasetLoader(raw_data_dir=DATA_DIR)
dataset = loader.get_dataset(num_timesteps_in=NUM_TIMESTEPS_IN,
                              num_timesteps_out=NUM_TIMESTEPS_OUT)
all_snapshots = list(dataset)
n_total = len(all_snapshots)
train_end = int(0.70 * n_total); val_end = int(0.80 * n_total)
train_set = all_snapshots[:train_end]
val_set   = all_snapshots[train_end:val_end]
test_set  = all_snapshots[val_end:]
if QUICK_MODE:
    train_set = train_set[:QUICK_TRAIN_N]
    val_set   = val_set[:QUICK_VAL_N]
    test_set  = test_set[:QUICK_TEST_N]
    print("      [QUICK_MODE aktif]")
print(f"      Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")


def iterate_batches(snapshots, batch_size, device):
    edge_index  = snapshots[0].edge_index.to(device)
    edge_weight = snapshots[0].edge_attr.to(device) if snapshots[0].edge_attr is not None else None
    for i in range(0, len(snapshots), batch_size):
        batch = snapshots[i:i+batch_size]
        if len(batch) < batch_size: break
        x = torch.stack([s.x for s in batch], dim=0).to(device)   # [B, N, F, T_in]
        y = torch.stack([s.y for s in batch], dim=0).to(device)   # [B, N, T_out]
        yield x, y, edge_index, edge_weight


# -----------------------------------------------------------------------------
class DCRNNForecaster(nn.Module):
    """
    BatchedDCRNN bir seq-to-seq mimari:
        input  [B, T_in, N, F]   output  [B, T_in, N, hidden]
    Bizim için son time-step'in hidden'ını alıp linear ile T_out'a projekte ediyoruz.
    """
    def __init__(self):
        super().__init__()
        self.dcrnn = BatchedDCRNN(in_channels=2, out_channels=HIDDEN_DIM, K=K_DIFFUSION)
        self.head  = nn.Linear(HIDDEN_DIM, NUM_TIMESTEPS_OUT)

    def forward(self, x, edge_index, edge_weight):
        # x: [B, N, F, T_in]  ->  [B, T_in, N, F]
        x_seq = x.permute(0, 3, 1, 2).contiguous()
        h_seq = self.dcrnn(x_seq, edge_index, edge_weight)   # [B, T_in, N, hidden]
        h_last = h_seq[:, -1, :, :]                          # [B, N, hidden]
        h_last = F.relu(h_last)
        return self.head(h_last)                              # [B, N, T_out]


model = DCRNNForecaster().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n[2/5] Model: BatchedDCRNN K={K_DIFFUSION}  |  Parametre: {n_params:,}")


# -----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn   = nn.MSELoss()
train_losses, val_losses = [], []
print(f"\n[3/5] Eğitim — {NUM_EPOCHS} epoch, lr={LEARNING_RATE}")

for epoch in range(NUM_EPOCHS):
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

    model.eval()
    val_sum, n_val = 0.0, 0
    with torch.no_grad():
        for x, y, ei, ew in iterate_batches(val_set, BATCH_SIZE, device):
            val_sum += loss_fn(model(x, ei, ew), y).item()
            n_val += 1
    val_loss = val_sum / max(n_val, 1)
    val_losses.append(val_loss)

    print(f"  Epoch {epoch+1:>2}/{NUM_EPOCHS}  "
          f"train_mse={train_loss:.4f}  val_mse={val_loss:.4f}  ({time.time()-t0:.1f}s)")


# -----------------------------------------------------------------------------
print("\n[4/5] Test...")
model.eval()
all_preds, all_truth = [], []
with torch.no_grad():
    for x, y, ei, ew in iterate_batches(test_set, BATCH_SIZE, device):
        all_preds.append(model(x, ei, ew).cpu().numpy())
        all_truth.append(y.cpu().numpy())
preds = np.concatenate(all_preds, axis=0)
truth = np.concatenate(all_truth, axis=0)
mae  = float(np.abs(preds - truth).mean())
rmse = float(np.sqrt(((preds - truth)**2).mean()))
horizon_mae = np.abs(preds - truth).mean(axis=(0, 1))
print(f"      Test MAE: {mae:.4f}  RMSE: {rmse:.4f}")


# -----------------------------------------------------------------------------
print("\n[5/5] Kaydediliyor...")
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model_dcrnn_baseline.pt"))
metrics = {
    "model": "DCRNN baseline (batched)",
    "num_params": n_params,
    "K_diffusion": K_DIFFUSION,
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
with open(os.path.join(RESULTS_DIR, "metrics_dcrnn_baseline.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

fig, ax = plt.subplots(figsize=(8, 5))
ep_x = range(1, NUM_EPOCHS + 1)
ax.plot(ep_x, train_losses, "o-", label="Train MSE")
ax.plot(ep_x, val_losses,   "s-", label="Val MSE")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (z-score)")
ax.set_title(f"DCRNN (batched, K={K_DIFFUSION}) — Loss Curves (METR-LA)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(FIGURES_DIR, "loss_dcrnn_baseline.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

print("\nDONE.")
