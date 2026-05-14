"""
Trafik Tahmini Projesi — METR-LA İlk Keşif
============================================

Amaç:
  1. Tüm kritik kütüphaneleri doğrula
  2. METR-LA veri setini indir ve yükle
  3. Yapısını anlamlı şekilde özetle

Çalıştırma:
    .\\venv\\Scripts\\python.exe notebooks\\01_explore_metrla.py
"""

import sys
import os

# Windows cp1254 console codepage Unicode arrow'ları basamıyor; UTF-8 force.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch_geometric as pyg
import torch_geometric_temporal as pygt
import numpy as np


def banner(text):
    print()
    print("=" * 64)
    print(text)
    print("=" * 64)


# ----------------------------------------------------------------------
# 1. Sistem & kütüphane bilgileri
# ----------------------------------------------------------------------
banner("VERSİYONLAR")
print(f"Python              : {sys.version.split()[0]}")
print(f"PyTorch             : {torch.__version__}")
print(f"  CUDA available    : {torch.cuda.is_available()}")
print(f"  CUDA runtime      : {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"  Device            : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM (toplam)     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"PyG                 : {pyg.__version__}")
print(f"PyG-Temporal        : import OK")
print(f"NumPy               : {np.__version__}")


# ----------------------------------------------------------------------
# 2. Kullanacağımız model sınıfları erişilebilir mi?
# ----------------------------------------------------------------------
banner("MODEL SINIFLARI IMPORT TESTİ")
results = {}
try:
    from torch_geometric_temporal.nn.recurrent import DCRNN
    results["DCRNN (recurrent)"] = "OK"
except Exception as e:
    results["DCRNN (recurrent)"] = f"FAIL — {e}"

try:
    from torch_geometric_temporal.nn.recurrent import GConvGRU
    results["GConvGRU (recurrent)"] = "OK"
except Exception as e:
    results["GConvGRU (recurrent)"] = f"FAIL — {e}"

try:
    from torch_geometric_temporal.nn.attention import STConv
    results["STConv (STGCN-benzeri)"] = "OK"
except Exception as e:
    results["STConv (STGCN-benzeri)"] = f"FAIL — {e}"

try:
    from torch_geometric_temporal.nn.attention import ASTGCN
    results["ASTGCN (attention)"] = "OK"
except Exception as e:
    results["ASTGCN (attention)"] = f"FAIL — {e}"

for name, status in results.items():
    print(f"  {name:30s}: {status}")


# ----------------------------------------------------------------------
# 3. METR-LA yükleme
# ----------------------------------------------------------------------
banner("METR-LA YÜKLENİYOR")

DATA_DIR = "./data/metr-la"
os.makedirs(DATA_DIR, exist_ok=True)

adj_path = os.path.join(DATA_DIR, "adj_mat.npy")
node_path = os.path.join(DATA_DIR, "node_values.npy")
zip_path = os.path.join(DATA_DIR, "METR-LA.zip")

# NOT: PyG-Temporal 0.56.2'nin _download_url metodu raw_data_dir'i 2x join'liyor (bug).
# Çözüm: ZIP'i manuel olarak doğru konuma indir, sonra PyG-Temporal'ın
# kendi extract akışı zip'i bulup açar. ANL Box hosting (yeni URL).
if not (os.path.isfile(adj_path) and os.path.isfile(node_path)):
    if not os.path.isfile(zip_path):
        import requests

        url = "https://anl.app.box.com/shared/static/plgsv3te0akmqluiuqva34su60nn93c2"
        print(f"METR-LA indiriliyor: {url}")
        r = requests.get(url, stream=True, allow_redirects=True, timeout=180)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -10
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = int(100 * downloaded / total)
                        if pct >= last_pct + 10:
                            print(f"  {pct:3d}%  ({downloaded/1024**2:6.1f} / {total/1024**2:6.1f} MB)")
                            last_pct = pct
        print(f"  zip indirildi: {os.path.getsize(zip_path)/1024**2:.2f} MB")
    else:
        print("ZIP zaten yerinde.")
else:
    print("METR-LA dosyaları cache'te (extract edilmiş).")

print("\nVeri dosyaları:")
for f in sorted(os.listdir(DATA_DIR)):
    fp = os.path.join(DATA_DIR, f)
    if os.path.isfile(fp):
        sz = os.path.getsize(fp) / 1024**2
        print(f"  {f:30s} {sz:.2f} MB")

from torch_geometric_temporal.dataset import METRLADatasetLoader

loader = METRLADatasetLoader(raw_data_dir=DATA_DIR)

NUM_IN = 12   # 12 zaman dilimi geçmiş (5dk x 12 = 60 dk)
NUM_OUT = 12  # 12 zaman dilimi tahmin (60 dk öne)
print(f"\nDataset oluşturuluyor (num_timesteps_in={NUM_IN}, num_timesteps_out={NUM_OUT})...")
dataset = loader.get_dataset(num_timesteps_in=NUM_IN, num_timesteps_out=NUM_OUT)
print("Yüklendi.")


# ----------------------------------------------------------------------
# 4. Yapı keşfi
# ----------------------------------------------------------------------
banner("VERİ SETİ YAPISI")

snapshots = list(dataset)
print(f"Toplam snapshot (zaman penceresi)  : {len(snapshots)}")

first = snapshots[0]
print(f"\nIlk snapshot anahtarlari  : {list(first.keys())}")
print(f"  x          shape       : {tuple(first.x.shape)}   "
      "<- [num_nodes, num_features, num_timesteps_in]")
print(f"  edge_index shape       : {tuple(first.edge_index.shape)}  "
      "<- [2, num_edges]")
if hasattr(first, "edge_attr") and first.edge_attr is not None:
    print(f"  edge_attr  shape       : {tuple(first.edge_attr.shape)}  "
          "<- [num_edges] (mesafe agirliklari)")
print(f"  y          shape       : {tuple(first.y.shape)}      "
      "<- [num_nodes, num_timesteps_out]")

num_nodes = first.x.shape[0]
num_edges = first.edge_index.shape[1]
print(f"\n  Düğüm sayısı           : {num_nodes}")
print(f"  Kenar sayısı           : {num_edges}")
print(f"  Ortalama derece        : {num_edges / num_nodes:.2f}")

# Sayısal dağılım
print(f"\nİlk snapshot.x stats:")
x = first.x.numpy()
print(f"  min={x.min():.3f}  max={x.max():.3f}  mean={x.mean():.3f}  std={x.std():.3f}")

print(f"\nİlk snapshot.y stats:")
y = first.y.numpy()
print(f"  min={y.min():.3f}  max={y.max():.3f}  mean={y.mean():.3f}  std={y.std():.3f}")

# Edge weight dağılımı
if hasattr(first, "edge_attr") and first.edge_attr is not None:
    ea = first.edge_attr.numpy()
    print(f"\nEdge weight dağılımı:")
    print(f"  min={ea.min():.4f}  max={ea.max():.4f}  mean={ea.mean():.4f}")
    print(f"  ilk 10 değer: {ea[:10].tolist()}")

banner("TAMAMLANDI")
print(f"Dataset hazır. {len(snapshots)} adet snapshot, her biri:")
print(f"  - x: {num_nodes} sensörün {NUM_IN} adımlık geçmiş okuması")
print(f"  - y: {num_nodes} sensörün {NUM_OUT} adımlık geleceği (hedef)")
print(f"  - edge_index + edge_attr: sensörler arası graf yapısı + mesafe ağırlıkları")
print()
print("Bir sonraki adım: train/val/test split, dataloader, ilk baseline model.")
