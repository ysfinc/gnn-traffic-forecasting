"""
Ablation Baseline — Vanilla LSTM (BATCHED, GRAF YOK)
=====================================================

Aynı batched training pipeline, sadece graf yok. Her (sensör, snapshot)
çiftini bağımsız sequence olarak LSTM'e ver. Eğer graf yapısı KATKI
sağlıyorsa, GNN modellerinin bunu geçmesi beklenir.

Çalıştırma:
    .\\venv\\Scripts\\python.exe -u notebooks\\04_train_vanilla_lstm.py
"""

import sys, os, time, json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch_geometric_temporal.dataset import METRLADatasetLoader


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
    """LSTM için edge bilgisi gerekmez."""
    for i in range(0, len(snapshots), batch_size):
        batch = snapshots[i:i+batch_size]
        if len(batch) < batch_size: break
        x = torch.stack([s.x for s in batch], dim=0).to(device)   # [B, N, F, T_in]
        y = torch.stack([s.y for s in batch], dim=0).to(device)   # [B, N, T_out]
        yield x, y


# -----------------------------------------------------------------------------
class VanillaLSTM(nn.Module):
    """
    (B, N) çiftlerini bağımsız sequence olarak LSTM'e veriyoruz.
    Reshape: [B, N, F, T] -> [B*N, T, F]
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(HIDDEN_DIM, NUM_TIMESTEPS_OUT)

    def forward(self, x):
        # x: [B, N, F, T_in]
        B, N, F, T = x.shape
        # her (B,N) için bir sequence olacak: [B*N, T, F]
        x = x.permute(0, 1, 3, 2).reshape(B * N, T, F)
        _, (h_n, _) = self.lstm(x)        # h_n: [1, B*N, hidden]
        h = h_n.squeeze(0).view(B, N, HIDDEN_DIM)    # [B, N, hidden]
        return self.head(h)                            # [B, N, T_out]


model = VanillaLSTM().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n[2/5] Model: Vanilla LSTM (NO graph)  |  Parametre: {n_params:,}")


# -----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn   = nn.MSELoss()
train_losses, val_losses = [], []
print(f"\n[3/5] Eğitim — {NUM_EPOCHS} epoch, lr={LEARNING_RATE}")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss, n_batches = 0.0, 0
    t0 = time.time()
    for x, y in iterate_batches(train_set, BATCH_SIZE, device):
        optimizer.zero_grad()
        y_hat = model(x)
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
        for x, y in iterate_batches(val_set, BATCH_SIZE, device):
            val_sum += loss_fn(model(x), y).item()
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
    for x, y in iterate_batches(test_set, BATCH_SIZE, device):
        all_preds.append(model(x).cpu().numpy())
        all_truth.append(y.cpu().numpy())
preds = np.concatenate(all_preds, axis=0)
truth = np.concatenate(all_truth, axis=0)
mae  = float(np.abs(preds - truth).mean())
rmse = float(np.sqrt(((preds - truth)**2).mean()))
horizon_mae = np.abs(preds - truth).mean(axis=(0, 1))
print(f"      Test MAE: {mae:.4f}  RMSE: {rmse:.4f}")


# -----------------------------------------------------------------------------
print("\n[5/5] Kaydediliyor...")
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model_vanilla_lstm.pt"))
metrics = {
    "model": "Vanilla LSTM (no graph, batched) — ablation baseline",
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
    "uses_graph": False,
}
with open(os.path.join(RESULTS_DIR, "metrics_vanilla_lstm.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

fig, ax = plt.subplots(figsize=(8, 5))
ep_x = range(1, NUM_EPOCHS + 1)
ax.plot(ep_x, train_losses, "o-", label="Train MSE")
ax.plot(ep_x, val_losses,   "s-", label="Val MSE")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (z-score)")
ax.set_title("Vanilla LSTM (batched) — NO graph structure")
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(FIGURES_DIR, "loss_vanilla_lstm.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

print("\nDONE.")
