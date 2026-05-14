"""
METR-LA Normalization Utilities
================================

PyG-Temporal METRLADatasetLoader veriyi z-score normalize edip yüklüyor ama
kullanılan mean/std değerlerini public yapmıyor. Bu modül o değerleri YENİDEN
hesaplayıp (aynı formülle) public ediyor — tahminleri tekrar gerçek birimlere
(mph) çevirebilelim.
"""

import os
import numpy as np


def compute_metrla_stats(data_dir: str = "./data/metr-la"):
    """node_values.npy'den feature-bazlı mean/std (PyG-Temporal iç işlemiyle aynı).

    Returns
    -------
    means : np.ndarray, shape [F]
    stds  : np.ndarray, shape [F]
    """
    X = np.load(os.path.join(data_dir, "node_values.npy"))
    # Loader içi transpose: [T, N, F] -> [N, F, T]
    X = X.transpose((1, 2, 0)).astype(np.float32)
    means = X.mean(axis=(0, 2))  # [F]
    stds  = X.std(axis=(0, 2))   # [F]
    return means, stds


def save_stats(data_dir: str = "./data/metr-la", output_path: str | None = None) -> str:
    """Mean/std'leri npz'e kaydet (cache)."""
    means, stds = compute_metrla_stats(data_dir)
    if output_path is None:
        output_path = os.path.join(data_dir, "normalization_stats.npz")
    np.savez(output_path, means=means, stds=stds)
    return output_path


def load_stats(data_dir: str = "./data/metr-la"):
    """Mean/std'leri yükle (yoksa hesaplayıp kaydet)."""
    path = os.path.join(data_dir, "normalization_stats.npz")
    if not os.path.isfile(path):
        save_stats(data_dir, path)
    npz = np.load(path)
    return npz["means"], npz["stds"]


def inverse_transform(values_z, means, stds, feature_idx: int = 0):
    """
    z-score → real units.

    METR-LA için feature_idx=0 = hız (mph), feature_idx=1 = günün saati (0-1).

    Args
    ----
    values_z : array (any shape) — z-score değerleri
    means, stds : feature-bazlı stats (compute/load ile alınır)
    feature_idx : kaçıncı feature inverse edilecek
    """
    return values_z * stds[feature_idx] + means[feature_idx]


if __name__ == "__main__":
    means, stds = load_stats()
    print(f"METR-LA normalization stats:")
    print(f"  feature 0 (speed, mph)     : mean={means[0]:.3f}  std={stds[0]:.3f}")
    print(f"  feature 1 (time-of-day)    : mean={means[1]:.3f}  std={stds[1]:.3f}")
