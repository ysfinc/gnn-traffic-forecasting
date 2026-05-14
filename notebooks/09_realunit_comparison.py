"""
Gerçek-Birim (mph) Karşılaştırma Raporu
========================================

Tüm modellerin z-score MAE/RMSE'sini mph'a çevirir, tezde kullanılacak
yapıda bir karşılaştırma raporu üretir.

Dönüşüm:
    MAE_z = MAE_real / std        =>   MAE_real = MAE_z * std
    RMSE_z = RMSE_real / std      =>   RMSE_real = RMSE_z * std

(Bu basit ölçeklemenin sebebi: tahminler ve hedefler z-score uzayında ama
fark/hata o uzayda da ölçeklenir; sabit bir kayma yok çünkü mean iki tarafta
da var.)

Çalıştırma (eğitim sonrası):
    .\\venv\\Scripts\\python.exe -u notebooks\\09_realunit_comparison.py
"""

import os
import sys
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt

from src.normalization import load_stats

RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"
DATA_DIR    = "./data/metr-la"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------
means, stds = load_stats(DATA_DIR)
SPEED_STD = float(stds[0])    # mph cinsinden std
print(f"Hız std       : {SPEED_STD:.4f} mph (z-score MAE'yi bu ile çarpınca mph)")
print(f"Hız mean      : {float(means[0]):.4f} mph")


# ---------------------------------------------------------------------------
# Modeller
# ---------------------------------------------------------------------------
metric_paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "metrics_*.json")))
results = []
for p in metric_paths:
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
        # Z-score -> mph
        d["test_mae_mph"]  = d["test_mae"]  * SPEED_STD
        d["test_rmse_mph"] = d["test_rmse"] * SPEED_STD
        d["horizon_mae_mph"] = [x * SPEED_STD for x in d["horizon_mae"]]
        results.append(d)

if not results:
    print("Hiç metric dosyası yok — önce eğitim script'lerini koşturmalısın.")
    sys.exit(0)

print(f"\nBulunan model sayısı: {len(results)}")


# ---------------------------------------------------------------------------
# 1. Genel karşılaştırma tablosu (z-score + mph)
# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("GENEL TEST METRİKLERİ (z-score + mph)")
print("=" * 96)
print(f"{'Model':<45} {'Params':>8} "
      f"{'MAE_z':>8} {'MAE_mph':>10} "
      f"{'RMSE_z':>8} {'RMSE_mph':>10} "
      f"{'Graph?':>7}")
print("-" * 96)
for r in results:
    uses_graph = r.get("uses_graph", True)
    print(f"{r['model']:<45} {r['num_params']:>8,} "
          f"{r['test_mae']:>8.4f} {r['test_mae_mph']:>10.2f} "
          f"{r['test_rmse']:>8.4f} {r['test_rmse_mph']:>10.2f} "
          f"{'YES' if uses_graph else 'NO':>7}")
print("=" * 96)


# ---------------------------------------------------------------------------
# 2. Horizon-bazlı MAE (mph) tablosu
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("HORIZON-BAZLI MAE (mph) — düşük olan iyi")
print("=" * 90)
n_horizons = len(results[0]["horizon_mae_mph"])
header = f"{'Horizon (dk)':>12} | " + " | ".join(
    f"{r['model'].split(' ')[0]:>10}" for r in results
)
print(header)
print("-" * len(header))
for h in range(n_horizons):
    row = f"{(h+1)*5:>10} dk  | " + " | ".join(
        f"{r['horizon_mae_mph'][h]:>10.2f}" for r in results
    )
    print(row)
print("=" * len(header))


# ---------------------------------------------------------------------------
# 3. Horizon-bazlı MAE plot (mph)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
horizons = [(h + 1) * 5 for h in range(n_horizons)]
for r in results:
    label = r["model"].replace(" — ablation baseline", "").replace(" baseline", "")
    ax.plot(horizons, r["horizon_mae_mph"], "o-", label=label, linewidth=2, markersize=7)
ax.set_xlabel("Tahmin horizonu (dakika)")
ax.set_ylabel("MAE (mph)")
ax.set_title("Horizon-bazlı MAE — METR-LA (gerçek birim)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
p1 = os.path.join(FIGURES_DIR, "realunit_horizon_mae.png")
fig.savefig(p1, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nHorizon MAE (mph) plot: {p1}")


# ---------------------------------------------------------------------------
# 4. Test metrikleri bar chart (mph)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = [r["model"].replace(" — ablation baseline", "").replace(" baseline", "")
          for r in results]
maes  = [r["test_mae_mph"]  for r in results]
rmses = [r["test_rmse_mph"] for r in results]
colors = ["#888" if not r.get("uses_graph", True) else "#2a7" for r in results]

x = np.arange(len(labels))
axes[0].bar(x, maes, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=20, ha="right")
axes[0].set_ylabel("Test MAE (mph)")
axes[0].set_title("Test MAE")
axes[0].grid(True, alpha=0.3, axis="y")
for xi, m in zip(x, maes):
    axes[0].text(xi, m + max(maes) * 0.02, f"{m:.2f}", ha="center", va="bottom")

axes[1].bar(x, rmses, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=20, ha="right")
axes[1].set_ylabel("Test RMSE (mph)")
axes[1].set_title("Test RMSE")
axes[1].grid(True, alpha=0.3, axis="y")
for xi, r_ in zip(x, rmses):
    axes[1].text(xi, r_ + max(rmses) * 0.02, f"{r_:.2f}", ha="center", va="bottom")

plt.tight_layout()
p2 = os.path.join(FIGURES_DIR, "realunit_test_metrics.png")
fig.savefig(p2, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Test metrikleri (mph) : {p2}")


# ---------------------------------------------------------------------------
# 5. Json olarak kaydet (tezde kullanmak üzere)
# ---------------------------------------------------------------------------
output = {
    "speed_std_mph": SPEED_STD,
    "speed_mean_mph": float(means[0]),
    "conversion_note": "MAE_real = MAE_z * std (and same for RMSE)",
    "models": [
        {
            "model": r["model"],
            "num_params": r["num_params"],
            "uses_graph": r.get("uses_graph", True),
            "test_mae_z": r["test_mae"],
            "test_mae_mph": r["test_mae_mph"],
            "test_rmse_z": r["test_rmse"],
            "test_rmse_mph": r["test_rmse_mph"],
            "horizon_mae_mph": r["horizon_mae_mph"],
        }
        for r in results
    ],
}
out_path = os.path.join(RESULTS_DIR, "realunit_comparison.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nRapor JSON: {out_path}")

print("\nDONE.")
