# Trafik Tahmini için Spatio-Temporal Graph Neural Networks

**METR-LA dataset üzerinde A3T-GCN, DCRNN ve Vanilla LSTM mimarilerinin karşılaştırmalı analizi**

Yapay Zeka Mühendisliği 3. Sınıf | Derin Öğrenme Dersi Final Projesi  
Yusuf İnce, 2026

---

## TL;DR — Sonuçlar (10 epoch, hidden=128)

| Model | Parametre | MAE (mph) | RMSE (mph) | Graf? |
|---|---|---|---|---|
| A3T-GCN | 101,400 | 8.33 | 13.93 | Yönsüz |
| **DCRNN** | 201,612 | 6.28 | **11.90** | Yönlü |
| **Vanilla LSTM** | 69,132 | **6.05** | 12.05 | (ablation) |

**Kritik bulgu:** ~55 dk horizon'da DCRNN, LSTM'i geçer ve RMSE'de tüm horizon'larda öne çıkar. → ITS uygulamaları için DCRNN tercih edilir.

Tam analiz: [`report/thesis_draft.md`](report/thesis_draft.md) · ISSD entegrasyon önerisi: [`report/issd_integration_framework.md`](report/issd_integration_framework.md)

## 🚦 İnteraktif Simülatör

```bash
streamlit run simulator/app.py
```

LA highway ağındaki bir bölgenin sensörlerinde trafik yoğunluğunu slider'larla değiştir → DCRNN modeli 60 dakika öne tahmin yapar → kavşaklar 🟢/🟡/🔴 olarak işaretlenir. Defansta canlı demo için ideal. Detay: [`simulator/README.md`](simulator/README.md)

## 🌐 Web Tanıtım Sayfası

Self-host edilebilir bir landing page de `web/` altında mevcut — paylaşımlı PHP hosting'e de yüklenebilir (Streamlit Cloud demosu iframe ile gömülü). Detaylı kurulum: [`web/README.md`](web/README.md)

---

## İçindekiler

1. [Hızlı Başlangıç](#hızlı-başlangıç)
2. [Sistem Gereksinimleri](#sistem-gereksinimleri)
3. [Kurulum](#kurulum)
4. [Veri İndirme](#veri-i̇ndirme)
5. [Pipeline'ı Çalıştırma](#pipelineı-çalıştırma)
6. [Çıktılar](#çıktılar)
7. [Depo Yapısı](#depo-yapısı)
8. [Bilinen Sorunlar / Workarounds](#bilinen-sorunlar--workarounds)
9. [Yeniden Üretilebilirlik](#yeniden-üretilebilirlik)
10. [Referanslar](#referanslar)
11. [Lisans](#lisans)

---

## Hızlı Başlangıç

```bash
# 1. Klonla
git clone https://github.com/<kullanıcı>/<repo>.git
cd <repo>

# 2. Sanal ortam (Windows PowerShell)
python -m venv venv
.\venv\Scripts\activate

# 2. Sanal ortam (Linux / macOS / WSL)
python -m venv venv
source venv/bin/activate

# 3. PyTorch (CUDA 12.1 build)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. PyG ve PyG-Temporal
pip install torch_geometric torch-geometric-temporal

# 5. Kalan paketler
pip install -r requirements.txt

# 6. Veriyi indir + yapısını gör
python notebooks/01_explore_metrla.py

# 7. Full pipeline (eğitim + karşılaştırma + görselleştirme + mph raporu)
python notebooks/02_train_baseline.py
python notebooks/03_train_dcrnn.py
python notebooks/04_train_vanilla_lstm.py
python notebooks/05_compare_models.py
python notebooks/06_visualize_predictions.py
python notebooks/09_realunit_comparison.py
```

RTX 4050 GPU üstünde toplam ~35 dakika. Smoke test için sonraki bölüme bakınız.

---

## Sistem Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---|---|---|
| Python | 3.10 | **3.12** |
| GPU | (CPU-only desteklenir, ~30x yavaş) | NVIDIA GPU + **CUDA 12.x** |
| VRAM | 4 GB | 6+ GB |
| RAM | 8 GB | 16+ GB |
| Disk | 5 GB | 10 GB |

Test edilmiş ortam: Windows 11 + Python 3.12.10 + PyTorch 2.5.1+cu121 + RTX 4050 Laptop (6 GB VRAM).

---

## Kurulum

### Adım 1 — Klonla

```bash
git clone https://github.com/<kullanıcı>/<repo>.git
cd <repo>
```

### Adım 2 — Sanal Ortam

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux / macOS / WSL:**
```bash
python -m venv venv
source venv/bin/activate
```

### Adım 3 — PyTorch

CUDA durumunuza göre uygun build:

```bash
# NVIDIA GPU + CUDA 12.x (önerilen)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU-only (yavaş ama çalışır)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Adım 4 — PyTorch Geometric + Temporal

```bash
pip install torch_geometric
pip install torch-geometric-temporal
```

> ⚠️ `torch-geometric-temporal 0.56.2`, `decorator==4.4.2` ister; eğer kurulum sırasında `decorator` uyumsuzluğu görürseniz: `pip install "decorator==4.4.2"` ile alt versiyona düşürün. Bu uyarı genelde fonksiyonelliği etkilemez.

### Adım 5 — Diğer Paketler

```bash
pip install -r requirements.txt
```

Kuruluyor: numpy, pandas, matplotlib, networkx, seaborn, plotly, folium, scikit-learn, tqdm, jupyter, statsmodels.

### Adım 6 — Doğrulama

```bash
python notebooks/99_api_test.py
```

Bu, `torch + cu121 + PyG-Temporal`'in yüklü ve GPU'nun erişilebilir olduğunu doğrular. Output şuna benzer olmalı:

```
Device: cuda
A3TGCN2 forward: 41.14 ms / batch-of-32 = 1.286 ms/snap
BatchedDCRNN forward: 27.86 ms / batch-of-32 = 0.871 ms/snap
```

---

## Veri İndirme

```bash
python notebooks/01_explore_metrla.py
```

Bu script:
1. METR-LA'yı indirir (ANL Box mirror, ~14 MB)
2. `data/metr-la/` altına extrakt eder (`adj_mat.npy` ≈ 0.16 MB, `node_values.npy` ≈ 108 MB)
3. PyG-Temporal'ın `METRLADatasetLoader` ile yükler
4. Yapı özetini ekrana yazar (snapshot sayısı, düğüm/kenar, feature shape'leri)

> ⚠️ **Neden manuel indirme?** PyG-Temporal'ın eski URL'i (graphmining.ai) dead; yeni URL'inin (ANL Box) `_download_url` metodu path'i çift join'liyor ve hata atıyor. Bu script bug'ı bypass eder.

Sensör konum verisi (harita için, opsiyonel):
```bash
python notebooks/08_sensor_map.py
```

HuggingFace'ten `sensor_locations.csv`, `adj_mx_mapping.json` indirir; LA haritasına yerleştirir.

---

## Pipeline'ı Çalıştırma

### Smoke Test (Hızlı Doğrulama, ~3 dakika)

Tüm eğitim scriptlerinde dosya başındaki sabitler değiştirerek subsample'lı çalıştırın:

```python
QUICK_MODE = True       # bu satırı True yapın
QUICK_TRAIN_N = 5000    # mevcut konfigde
NUM_EPOCHS = 1          # 10'dan 1'e indirin
```

Sonra:
```bash
python notebooks/02_train_baseline.py
python notebooks/03_train_dcrnn.py
python notebooks/04_train_vanilla_lstm.py
```

Her biri ~30-60 saniye. Pipeline'ın baştan sona çalıştığını doğrular.

### Full Eğitim (~35 dakika RTX 4050'de)

Mevcut konfigle (hidden=128, 10 epoch, QUICK_MODE=False):

```bash
python notebooks/02_train_baseline.py    # A3T-GCN  (~15 dk)
python notebooks/03_train_dcrnn.py        # DCRNN    (~21 dk)
python notebooks/04_train_vanilla_lstm.py # LSTM     (~4 dk)
```

Her script otomatik olarak:
- Veriyi yükler
- Kronolojik 70/10/20 split yapar
- Eğitir (epoch başına train + val MSE basar)
- Test setinde MAE, RMSE, horizon-bazlı MAE hesaplar
- Model checkpoint + metric JSON + loss plot kaydeder

### Analizler (~10 saniye)

```bash
python notebooks/05_compare_models.py      # Karşılaştırma tabloları + plotlar
python notebooks/06_visualize_predictions.py # Çoklu-model tahmin görselleri
python notebooks/07_dataset_stats.py        # Dataset istatistikleri
python notebooks/09_realunit_comparison.py  # mph cinsinden rapor
```

### Tek Komutla Tüm Pipeline (önerilmez ama mümkün)

**Linux/macOS:**
```bash
for s in 02_train_baseline 03_train_dcrnn 04_train_vanilla_lstm \
         05_compare_models 06_visualize_predictions 09_realunit_comparison; do
  python notebooks/${s}.py
done
```

**Windows PowerShell:**
```powershell
@('02_train_baseline', '03_train_dcrnn', '04_train_vanilla_lstm',
  '05_compare_models', '06_visualize_predictions', '09_realunit_comparison') |
  ForEach-Object { python notebooks/$_.py }
```

---

## Çıktılar

### `results/` (her tam çalışmadan sonra)

```
results/
├── metrics_a3tgcn_baseline.json    # train/val/test MSE, MAE, RMSE, horizon-bazlı MAE
├── metrics_dcrnn_baseline.json
├── metrics_vanilla_lstm.json
├── realunit_comparison.json        # Tüm modellerin mph cinsinden karşılaştırması
├── dataset_stats.json              # METR-LA detaylı istatistikleri
├── model_a3tgcn_baseline.pt        # PyTorch checkpoint'ler (gitignored)
├── model_dcrnn_baseline.pt
└── model_vanilla_lstm.pt
```

### `figures/` (PNG ve HTML)

```
figures/
├── loss_a3tgcn_baseline.png         # Eğitim eğrileri (her model)
├── loss_dcrnn_baseline.png
├── loss_vanilla_lstm.png
├── comparison_loss_curves.png       # 3 modelin train/val eğrileri
├── comparison_horizon_mae.png       # Horizon-bazlı MAE (z-score)
├── comparison_test_metrics.png      # Test MAE/RMSE bar chart
├── realunit_horizon_mae.png         # Horizon MAE (mph cinsinden)
├── realunit_test_metrics.png        # Test metrikleri (mph)
├── viz_predictions_multi.png        # 4 sensörde 3 modelin tahmini + gerçek
├── viz_horizon_mae_multi.png        # Horizon MAE overlay (3 model)
├── viz_scatter_best.png             # En iyi modelin pred-vs-true scatter
├── sensor_map.html                  # Folium interaktif harita
└── sensor_map_static.png            # Matplotlib statik harita
```

### `report/` (tez taslakları)

- `thesis_draft.md` — Tam tez gövdesi (özet, giriş, yöntem, sonuçlar, tartışma, sonuç)
- `issd_integration_framework.md` — ISSD entegrasyon öneri bölümü

---

## Depo Yapısı

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/                            # gitignored, veri indirilince oluşur
│   └── metr-la/
│       ├── adj_mat.npy
│       ├── node_values.npy
│       └── sensor_graph/
│           ├── sensor_locations.csv
│           └── adj_mx_mapping.json
├── notebooks/                       # script'ler (numara sırayla çalıştır)
│   ├── 01_explore_metrla.py
│   ├── 02_train_baseline.py         # A3T-GCN
│   ├── 03_train_dcrnn.py            # DCRNN
│   ├── 04_train_vanilla_lstm.py     # LSTM (ablation)
│   ├── 05_compare_models.py
│   ├── 06_visualize_predictions.py
│   ├── 07_dataset_stats.py
│   ├── 08_sensor_map.py
│   ├── 09_realunit_comparison.py
│   └── 99_api_test.py               # smoke test (3 saniye)
├── src/
│   ├── __init__.py
│   └── normalization.py             # z-score ↔ mph dönüşümü
├── results/                         # metric JSON'lar repoda; .pt dosyaları gitignored
├── figures/                         # PNG + HTML çıktılar
└── report/
    ├── thesis_draft.md
    └── issd_integration_framework.md
```

---

## Bilinen Sorunlar / Workarounds

### 1. METR-LA download bug (PyG-Temporal 0.56.2)

**Sorun:** `METRLADatasetLoader._download_url` path'i iki kez join'liyor → `FileNotFoundError`. Ayrıca eski URL (graphmining.ai) 404.

**Çözüm:** `notebooks/01_explore_metrla.py` ANL Box mirror'ından dosyayı manuel indirip doğru konuma yerleştirir, sonra PyG-Temporal'ın loader'ı yalnız zip'i açar. Tek seferlik.

### 2. Windows Türkçe console encoding

**Sorun:** Windows cp1254 codepage'i `←`, `→`, `✓` gibi Unicode karakterleri basamıyor → `UnicodeEncodeError`.

**Çözüm:** Tüm script'ler `sys.stdout.reconfigure(encoding="utf-8")` çağırıyor. Sorun yok.

### 3. `decorator` uyumsuzluk uyarısı

**Sorun:** `torch-geometric-temporal 0.56.2`, `decorator==4.4.2` ister ama PyG-Temporal kurulum sırasında `decorator>=5.x` da yükleyebilir.

**Çözüm:** Genelde fonksiyonalite etkilenmez. Hata alırsanız: `pip install "decorator==4.4.2" --force-reinstall`.

### 4. `A3TGCN2` sabit batch_size

**Sorun:** Constructor'da fixed; partial batch'lerle hata atar.

**Çözüm:** Manuel batch'leme partial batch'leri atlıyor (`if len(batch) < batch_size: break`). Veri kaybı 0.1-1% düzeyinde, fair karşılaştırmayı etkilemez.

### 5. 6 GB VRAM ile büyük hidden boyutu

**Sorun:** `hidden=256+` ve `batch=64+` denerseniz OOM.

**Çözüm:** Mevcut konfig (hidden=128, batch=32) RTX 4050'de güvenli. Daha güçlü GPU varsa yukarı doğru taranabilir.

---

## Yeniden Üretilebilirlik

- Tüm script'ler `SEED = 42` ile başlar; `torch.manual_seed`, `numpy.random.seed` çağırır
- Aynı donanım + aynı PyTorch versiyonu ile birebir aynı sayıları üretir
- Sonuçlar `results/*.json` içinde commit edilebilir → CI ile farkı izleyebilirsiniz

---

## Referanslar

Tam referans listesi `report/thesis_draft.md` içinde. Ana kaynaklar:

1. Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926). *ICLR*.
2. Zhu, J., et al. (2020). [A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting](https://arxiv.org/abs/2006.11583).
3. Rozemberczki, B., et al. (2021). [PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models](https://arxiv.org/abs/2104.07788). *CIKM*.

---

## Lisans

[MIT](LICENSE). Kullanma + dağıtma serbest, attribütion verilirse iyi olur.

---

## İletişim

Yusuf İnce — inceyusuf2121@gmail.com
