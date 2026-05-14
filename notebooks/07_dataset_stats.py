"""
Dataset Statistics — METR-LA Detaylı Karakterizasyon
=====================================================

Tezin "Dataset" bölümünde gerekecek sayılar:
  - Boyutlar (zaman, sensör, feature)
  - Graf yapı (kenar sayısı, yoğunluk, derece dağılımı, simetri)
  - Hız dağılımı (min/max/mean/std, eksik veri)
  - Edge weight dağılımı
  - Günün saati dağılımı

Çalıştırma:
    .\\venv\\Scripts\\python.exe -u notebooks\\07_dataset_stats.py
"""

import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

DATA_DIR    = "./data/metr-la"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def banner(t):
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64)


# ---------------------------------------------------------------------------
# Veri yükle
# ---------------------------------------------------------------------------
A = np.load(os.path.join(DATA_DIR, "adj_mat.npy"))
X = np.load(os.path.join(DATA_DIR, "node_values.npy"))   # [T, N, F]
T, N, F = X.shape

stats = {}

# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------
banner("TEMPORAL")
n_days = T * 5 / 60 / 24
print(f"  Toplam zaman dilimi (5dk adım) : {T}")
print(f"  Süre                            : {n_days:.2f} gün ({T*5/60:.1f} saat)")
stats["temporal"] = {"timesteps": int(T), "span_days": round(n_days, 2)}

# ---------------------------------------------------------------------------
# Spatial / Graf
# ---------------------------------------------------------------------------
banner("SPATIAL (Graf)")
A_nz = A > 0
num_edges = int(A_nz.sum())
num_self_loops = int(np.diagonal(A_nz).sum())
density = num_edges / (N * N - N) * 100
sym = bool(np.allclose(A, A.T))

deg = A_nz.sum(axis=1)  # her düğümün giden komşu sayısı

print(f"  Düğüm sayısı (N)               : {N}")
print(f"  Kenar sayısı (A_ij > 0)         : {num_edges}")
print(f"  Self-loops                      : {num_self_loops}")
print(f"  Kenar yoğunluğu                 : {density:.3f}%")
print(f"  Simetrik mi (yönsüz mü)?        : {sym}")
print(f"  Derece dağılımı:")
print(f"    min = {deg.min()},  max = {deg.max()},  mean = {deg.mean():.2f}")
print(f"    median = {np.median(deg):.1f},  std = {deg.std():.2f}")

# Edge weight stats (sadece sıfırdan büyük)
ew = A[A_nz]
print(f"\n  Edge weight (mesafe-tabanlı) dağılımı:")
print(f"    min = {ew.min():.4f},  max = {ew.max():.4f}")
print(f"    mean = {ew.mean():.4f},  median = {np.median(ew):.4f},  std = {ew.std():.4f}")

stats["spatial"] = {
    "num_nodes": int(N),
    "num_edges": num_edges,
    "self_loops": num_self_loops,
    "density_pct": round(density, 3),
    "symmetric": sym,
    "degree_min": int(deg.min()), "degree_max": int(deg.max()),
    "degree_mean": round(float(deg.mean()), 2),
    "edge_weight_mean": round(float(ew.mean()), 4),
    "edge_weight_std": round(float(ew.std()), 4),
}

# ---------------------------------------------------------------------------
# Features (speed + time-of-day)
# ---------------------------------------------------------------------------
banner("FEATURES")
print(f"  Feature sayısı (F)              : {F}")

speed = X[:, :, 0]
print(f"\n  Feature 0 — SPEED (mph):")
print(f"    min  = {speed.min():.2f},  max  = {speed.max():.2f}")
print(f"    mean = {speed.mean():.2f}, std  = {speed.std():.2f}")
print(f"    quantiles q[10/50/90] = "
      f"{np.quantile(speed, 0.1):.1f} / {np.quantile(speed, 0.5):.1f} / {np.quantile(speed, 0.9):.1f}")
n_zeros = int((speed == 0).sum())
print(f"    '0' okumalar (potansiyel eksik): {n_zeros} / {speed.size} "
      f"({100*n_zeros/speed.size:.2f}%)")

if F > 1:
    tod = X[:, :, 1]
    print(f"\n  Feature 1 — TIME-OF-DAY:")
    print(f"    min  = {tod.min():.4f},  max  = {tod.max():.4f}")
    print(f"    mean = {tod.mean():.4f}, std  = {tod.std():.4f}")

stats["features"] = {
    "num_features": int(F),
    "speed_min": round(float(speed.min()), 2),
    "speed_max": round(float(speed.max()), 2),
    "speed_mean": round(float(speed.mean()), 2),
    "speed_std": round(float(speed.std()), 2),
    "speed_q10": round(float(np.quantile(speed, 0.1)), 2),
    "speed_q50": round(float(np.quantile(speed, 0.5)), 2),
    "speed_q90": round(float(np.quantile(speed, 0.9)), 2),
    "missing_zeros_pct": round(100 * n_zeros / speed.size, 4),
}

# ---------------------------------------------------------------------------
# Kaydet
# ---------------------------------------------------------------------------
out_path = os.path.join(RESULTS_DIR, "dataset_stats.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"\nJSON kaydı: {out_path}")

banner("DONE")
