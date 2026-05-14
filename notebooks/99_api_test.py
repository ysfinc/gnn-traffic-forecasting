"""Hızlı API doğrulama — batched mimarilerin imzaları."""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.nn.recurrent import A3TGCN2, BatchedDCRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1) Data
loader = METRLADatasetLoader(raw_data_dir="./data/metr-la")
ds = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
snaps = list(ds)
print(f"Snapshots: {len(snaps)}")
print(f"snap[0].x: {tuple(snaps[0].x.shape)}")
print(f"snap[0].y: {tuple(snaps[0].y.shape)}")
print(f"snap[0].edge_index: {tuple(snaps[0].edge_index.shape)}")

# 2) Manual batch
B = 32
batch = snaps[:B]
x = torch.stack([s.x for s in batch], dim=0).to(device)        # [B, N, F, T]
y = torch.stack([s.y for s in batch], dim=0).to(device)        # [B, N, T_out]
ei = batch[0].edge_index.to(device)
ew = batch[0].edge_attr.to(device)

print(f"\nBatched x: {tuple(x.shape)}   y: {tuple(y.shape)}")

# 3) A3TGCN2
print("\n--- A3TGCN2 ---")
m1 = A3TGCN2(in_channels=2, out_channels=32, periods=12, batch_size=B).to(device)
try:
    h = m1(x, ei, ew)
    print(f"forward OK, output: {tuple(h.shape)}")  # expected [B, N, 32]
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")

# 4) BatchedDCRNN — input format: [B, T, N, F]
print("\n--- BatchedDCRNN ---")
x_dcrnn = x.permute(0, 3, 1, 2).contiguous()  # [B, T, N, F]
print(f"  input shape: {tuple(x_dcrnn.shape)}")

m2 = BatchedDCRNN(in_channels=2, out_channels=32, K=3).to(device)
try:
    out = m2(x_dcrnn, ei, ew)
    print(f"forward OK, output: {tuple(out.shape) if hasattr(out, 'shape') else type(out)}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")

# Forward timing
import time
print("\n--- Hız ölçümü (warm-up sonrası 10 iter) ---")

# Warmup A3TGCN2
for _ in range(3):
    _ = m1(x, ei, ew)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    _ = m1(x, ei, ew)
torch.cuda.synchronize()
t = (time.time() - t0) / 10
print(f"A3TGCN2 forward: {t*1000:.2f} ms / batch-of-{B} = {t*1000/B:.3f} ms/snap")

# Warmup BatchedDCRNN
for _ in range(3):
    _ = m2(x_dcrnn, ei, ew)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    _ = m2(x_dcrnn, ei, ew)
torch.cuda.synchronize()
t = (time.time() - t0) / 10
print(f"BatchedDCRNN forward: {t*1000:.2f} ms / batch-of-{B} = {t*1000/B:.3f} ms/snap")

print("\nDONE.")
