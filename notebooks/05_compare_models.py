"""
Karşılaştırma — Eğitilmiş Modeller
===================================

results/ klasöründeki tüm metrics_*.json dosyalarını okur, tablo + grafik üretir:
  - Karşılaştırma tablosu (Param, Test MAE, Test RMSE)
  - Loss eğrileri (train + val, aynı eksende)
  - Horizon-bazlı MAE karşılaştırması

Çalıştırma (tüm eğitimler bittikten sonra):
    .\\venv\\Scripts\\python.exe -u notebooks\\05_compare_models.py
"""

import sys
import os
import json
import glob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Metric dosyalarını yükle
# ---------------------------------------------------------------------------
metric_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "metrics_*.json")))
if not metric_files:
    print("Henüz metric dosyası yok — önce eğitim script'lerini koştur.")
    sys.exit(0)

results = []
for path in metric_files:
    with open(path, encoding="utf-8") as f:
        results.append(json.load(f))

print(f"Bulunan model sayısı: {len(results)}")


# ---------------------------------------------------------------------------
# 2. Karşılaştırma tablosu
# ---------------------------------------------------------------------------
print()
print("=" * 86)
print(f"{'Model':<45} {'Params':>10} {'Test MAE':>10} {'Test RMSE':>12} {'Graph?':>7}")
print("-" * 86)
for r in results:
    uses_graph = r.get("uses_graph", True)
    print(f"{r['model']:<45} {r['num_params']:>10,} "
          f"{r['test_mae']:>10.4f} {r['test_rmse']:>12.4f} "
          f"{'YES' if uses_graph else 'NO':>7}")
print("=" * 86)


# ---------------------------------------------------------------------------
# 3. Loss eğrileri yan yana
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for r in results:
    label = r["model"].replace(" — ablation baseline", "").replace(" baseline", "")
    epochs_x = range(1, len(r["train_loss_per_epoch"]) + 1)
    axes[0].plot(epochs_x, r["train_loss_per_epoch"], "o-", label=label)
    axes[1].plot(epochs_x, r["val_loss_per_epoch"],   "s-", label=label)
axes[0].set_title("Train MSE per Epoch")
axes[1].set_title("Val MSE per Epoch")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (z-score)")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "comparison_loss_curves.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nLoss karşılaştırma : {out_path}")


# ---------------------------------------------------------------------------
# 4. Horizon-bazlı MAE
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for r in results:
    label = r["model"].replace(" — ablation baseline", "").replace(" baseline", "")
    horizons = [(h + 1) * 5 for h in range(len(r["horizon_mae"]))]
    ax.plot(horizons, r["horizon_mae"], "o-", label=label, linewidth=2, markersize=7)
ax.set_xlabel("Tahmin horizonu (dakika)")
ax.set_ylabel("MAE (z-score)")
ax.set_title("Horizon-bazlı MAE Karşılaştırması — METR-LA")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "comparison_horizon_mae.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Horizon MAE       : {out_path}")


# ---------------------------------------------------------------------------
# 5. Bar chart — Test MAE & RMSE
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = [r["model"].replace(" — ablation baseline", "").replace(" baseline", "")
          for r in results]
maes  = [r["test_mae"]  for r in results]
rmses = [r["test_rmse"] for r in results]
colors = ["#888" if not r.get("uses_graph", True) else "#2a7" for r in results]

x = np.arange(len(labels))
axes[0].bar(x, maes, color=colors)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=20, ha="right")
axes[0].set_ylabel("Test MAE (z-score)")
axes[0].set_title("Test MAE")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(x, rmses, color=colors)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=20, ha="right")
axes[1].set_ylabel("Test RMSE (z-score)")
axes[1].set_title("Test RMSE")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "comparison_test_metrics.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Test metrikleri bar: {out_path}")

print("\nDONE.")
