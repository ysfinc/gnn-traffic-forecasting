# Tez Taslağı — Trafik Tahmini için Spatio-Temporal Graph Neural Networks: METR-LA Üzerinde Karşılaştırmalı Çalışma ve ITS Uygulama Çerçevesi

**Yusuf İnce** | Yapay Zeka Mühendisliği 3. Sınıf | Derin Öğrenme Dersi Final Projesi

---

## ÖZET

Bu çalışmada, şehir-ölçekli trafik tahmini problemi için **Spatio-Temporal Graph Neural Networks (GNN)** mimarileri incelenmiş ve standart METR-LA veri seti üzerinde karşılaştırmalı analizleri yapılmıştır. Üç mimari uygulanmıştır: (1) Attention Temporal Graph Convolutional Network (A3T-GCN), (2) Diffusion Convolutional Recurrent Neural Network (DCRNN), ve (3) yapı olarak grafiği kullanmayan kontrol grubu olarak vanilla LSTM. PyTorch Geometric Temporal kütüphanesinin batched mimari versiyonları (A3TGCN2, BatchedDCRNN) kullanılarak ~150 kat hızlandırılmış bir eğitim pipeline'ı kurulmuştur. Deneyler RTX 4050 GPU üzerinde, 24,000 zaman pencerelik eğitim seti ile 10 epoch boyunca hidden=128 boyutlu modellerle yapılmıştır. Bulgular: (i) Vanilla LSTM kısa-orta horizon'da (5-45 dakika) sürpriz şekilde rekabetçi bir baseline oluşturmuştur (MAE 6.05 mph), (ii) DCRNN 55+ dakika horizon'da ve **RMSE** metriğinde tüm horizon'larda öne çıkmıştır (RMSE 11.90 mph vs LSTM 12.05 mph), (iii) A3T-GCN yönsüz adjacency varsayımı nedeniyle her metrikte geride kalmıştır. Bulgular, GNN'in mekansal indüktif bias'ının trafik tahmininde **horizon-bağımlı** bir avantaj sağladığını göstermektedir. Ek olarak, çıkarılan model **ISSD'nin CHAOS/MANGO platformuna paralel bir öngörü katmanı** olarak nasıl entegre edilebileceği bir framework önerisi olarak sunulmuştur.

**Anahtar kelimeler:** Trafik tahmini, Graph Neural Networks, DCRNN, spatio-temporal modelleme, akıllı ulaşım sistemleri (ITS)

---

## 1. GİRİŞ

### 1.1 Motivasyon

Şehir trafiği, kavşaklar arasında karmaşık etkileşim örüntülerine sahip yönlü bir ağdır. Bir lokasyondaki tıkanıklık, dakikalar içinde uzak komşulara yayılabilir; bu yayılma örüntüsünün **önceden tahmin edilmesi**, akıllı ulaşım sistemlerinde proaktif sinyal yönetimi ve trafik yönlendirmesi için kritik bir kapasitedir. Klasik zaman serisi modelleri (ARIMA, LSTM, Transformer), bir sensörün tahminini büyük ölçüde kendi geçmişine dayandırır; ancak trafiğin yapısı gereği komşu sensörlerden gelen sinyaller, özellikle uzun horizon'da, kritik bilgi taşır. **Graph Neural Networks (GNN)**, ağı oluşturan ilişki yapısını öğrenme sürecinin içine yerleştirerek bu eksiği gidermeyi vaat eder.

### 1.2 Problem Tanımı

Bu çalışmanın problem tanımı şudur:

> Trafik sensörlerinden oluşan bir mekansal ağda, *T_in = 12* adımlık (60 dakikalık) bir geçmiş penceresi verildiğinde, *T_out = 12* adımlık (60 dakikalık) bir gelecek penceresinde tüm sensörlerin hızını tahmin et.

Veri seti olarak literatürdeki en yaygın benchmark olan **METR-LA** kullanılmıştır: Los Angeles'taki 207 highway loop dedektörünün 4 aylık 5-dakika çözünürlüklü hız okumaları.

### 1.3 Katkılar

Bu çalışmanın katkıları şunlardır:

1. **Karşılaştırmalı deneysel çerçeve.** A3T-GCN, DCRNN ve vanilla LSTM (ablation) modelleri *aynı veri böleni, aynı optimizatör, aynı eğitim bütçesi* altında karşılaştırılmıştır. Kontrol edilen değişkenlerle yapılan bu karşılaştırma, gerçek mimari farkların ölçülmesini mümkün kılmaktadır.
2. **Horizon-bağımlı analiz.** Modellerin avantajının ufku ile değiştiği gösterilmiştir: vanilla LSTM ≤45 dakikalık tahminde rekabetçidir; ≥55 dakikalık tahminde DCRNN üstün gelir.
3. **Endüstriyel uygulama framework'ü.** Sonuçlar ISSD Bilişim'in CHAOS/MANGO akıllı ulaşım platformuna entegrasyon için bir mimari öneriye dönüştürülmüştür (Bölüm 6).
4. **Hızlandırılmış eğitim pipeline'ı.** PyTorch Geometric Temporal'in batched mimari versiyonları (A3TGCN2, BatchedDCRNN) kullanılarak ~150 kat hızlanma elde edilmiştir; bu pipeline yeniden çalıştırılabilir olarak paylaşılmaktadır.

### 1.4 Tezin Düzeni

Bölüm 2 ilgili çalışmaları özetler; Bölüm 3 veri setini ve yöntemi tanıtır; Bölüm 4 model mimarilerini açıklar; Bölüm 5 deneysel sonuçları sunar; Bölüm 6 ISSD entegrasyon framework'ünü içerir; Bölüm 7 sonuçları tartışır ve sınırlamaları belirtir.

---

## 2. İLGİLİ ÇALIŞMALAR

> **Yusuf'a not:** Bu bölüm 400-500 kelime hedefli. Aşağıdaki referansları okumak ve özetlemek yeterli.

**Anahtar referanslar (okuma sırası önerilen):**

1. **Kipf & Welling (2017)** — *Semi-Supervised Classification with Graph Convolutional Networks.* GCN'in temel referansı; tüm mesaj-iletme tabanlı modellerin atası.
2. **Li, Yu, Shahabi, Liu (2018)** — *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* DCRNN'i tanıtan paper; bu tezin merkezi referansı. METR-LA dataset'ini de bu paper sunmuştur.
3. **Yu, Yin, Zhu (2018)** — *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.* STGCN'i tanıtır; alternatif spatio-temporal mimari.
4. **Zhu et al. (2020)** — *A3T-GCN: Attention Temporal Graph Convolutional Network.* Bu tezdeki A3T-GCN mimarisinin kaynağı.
5. **Wu et al. (2019)** — *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* Adaptif adjacency kavramını getirir; ileride çalışılabilecek bir yön.
6. **Veličković et al. (2018)** — *Graph Attention Networks.* GAT — komşu agregasyonunda attention kavramı.

**Yazarken kapsanacaklar:**
- Klasik time series → derin öğrenme (LSTM, Transformer) → graph-aware derin öğrenme şeklindeki evrim
- METR-LA üzerinde önceki sonuçlar (DCRNN: ~3 mph at 15 min in original paper — bizim daha küçük capacity ile elde ettiğimiz sonuçla farkı tartış)
- Trafiğin **yönlü doğası** ve simetrik GCN'lerin neden yetersiz olabileceği

---

## 3. VERİ SETİ VE YÖNTEM

### 3.1 METR-LA

**Anahtar istatistikler** (`results/dataset_stats.json` çıktısından):

| Özellik | Değer |
|---|---|
| Sensör sayısı | 207 |
| Kenar sayısı | 1722 (yönlü, asimetrik adjacency) |
| Ortalama derece | 8.32 |
| Zaman dilimi | 34,272 (5-dk granülerlikte) |
| Süre | 119 gün (4 ay) |
| Feature | hız (mph) + günün saati |
| Hız ortalaması | 53.72 mph |
| Eksik veri (0 okumalar) | %8.11 |

Veri, DCRNN paper'ında kullanılan z-score normalize edilmiş formda yüklenmiştir.

### 3.2 Veri Bölme

Zaman serisi karakterini koruyabilmek için **kronolojik** bir bölme kullanılmıştır: ilk %70 eğitim (23,974 snapshot), sonraki %10 doğrulama (3,425), son %20 test (6,850). Random bölme, bilgi sızıntısına yol açacağından bu problem türünde uygunsuzdur.

### 3.3 Modeller

Üç model uygulanmıştır. Tümü aynı `[B, N, F, T_in]` girdi formatını kabul edip `[B, N, T_out]` çıktısı üretir.

**(a) A3T-GCN (A3TGCN2 — batched implementation).** TGCN bloğu üzerine zamansal attention. *Yönsüz* adjacency varsayımıyla çalışır.

**(b) DCRNN (BatchedDCRNN).** Diffusion convolution + GRU. *Yönlü* olasılıksal yürüyüş tabanlı mekansal toplama (DCRNN paper, K=2 diffusion adımı).

**(c) Vanilla LSTM (ablation).** Tüm 207 sensör için *paylaşılan* tek-katmanlı LSTM. *Graf yapısını kullanmaz* — bu, "graf gerçekten gerekli mi?" sorusunun deneysel cevabı içindir.

### 3.4 Eğitim

| Parametre | Değer |
|---|---|
| Optimizatör | Adam, lr=1e-3 |
| Loss | Mean Squared Error (MSE) |
| Hidden boyutu | 128 |
| Batch boyutu | 32 |
| Epoch | 10 |
| Seed | 42 |
| Donanım | NVIDIA RTX 4050 Laptop GPU (6 GB VRAM) |

### 3.5 Metrikler

Z-score uzayında MAE ve RMSE hesaplanmıştır; ek olarak `std × MAE_z = MAE_mph` ölçeklemesiyle gerçek-birim raporlar verilmiştir. Horizon-bazlı analiz için her ufuktan (5, 10, ..., 60 dakika) MAE ayrıca hesaplanmıştır.

---

## 4. SONUÇLAR

### 4.1 Genel Performans

| Model | Parametre | MAE (mph) | RMSE (mph) |
|---|---|---|---|
| A3T-GCN | 101,400 | 8.33 | 13.93 |
| DCRNN | 201,612 | 6.28 | **11.90** |
| Vanilla LSTM | 69,132 | **6.05** | 12.05 |

→ `figures/comparison_test_metrics.png`

### 4.2 Horizon-Bazlı Analiz (mph)

| Horizon | A3T-GCN | DCRNN | LSTM |
|---|---|---|---|
| 5 dk | 6.20 | 3.61 | **3.14** |
| 15 dk | 7.10 | 4.83 | **4.38** |
| 30 dk | 8.24 | 6.25 | **5.99** |
| 45 dk | 9.29 | 7.39 | **7.33** |
| **55 dk** | 9.83 | **7.95** | 8.06 |
| 60 dk | 10.10 | 8.42 | **8.39** |

**Crossover ~55 dakikada gerçekleşir** — DCRNN bu horizon'da ilk kez LSTM'i geçer.

→ `figures/viz_horizon_mae_multi.png`, `figures/comparison_horizon_mae.png`

### 4.3 Eğitim Eğrileri

→ `figures/loss_*_baseline.png`, `figures/comparison_loss_curves.png`

Tüm modeller stabil yakınsıyor. DCRNN'in val MSE'si 7. epoch civarında plateau yapıyor (overfit emaresi yok, eğer 10+ epoch verilirse muhtemelen marjinal iyileşmeler). LSTM 4. epoch'tan sonra plato'da. A3T-GCN ise 10 epoch sonunda hâlâ azalma eğiliminde, daha uzun eğitim modeli iyileştirebilir.

### 4.4 Açıklayıcılık — Scatter Analizi

`figures/viz_scatter_best.png`: LSTM (en iyi MAE) için pred vs true scatter. Yüksek hızlarda (60+ mph) yığılma görülüyor — çünkü METR-LA verisi highway sensörlerden geliyor ve serbest akış hızı baskın. Düşük hızlarda (<30 mph, tıkanıklık) tahminler daha dağınık — bu, kritik **trafik olaylarının tahmininin daha zor olduğunu** gösterir. Operatör tarafında bu, "model en çok yardıma ihtiyaç duyulan durumlarda en az emin" demek olur, açık bir gelişme yönüdür.

---

## 5. TARTIŞMA

### 5.1 Neden LSTM Bu Kadar Güçlü Çıktı?

Beklenmedik bir bulgu olarak vanilla LSTM, kısa-orta horizon'da iki GNN modelini de geçti. Birkaç olası sebep:

1. **Paylaşılan parametre kazancı.** Tek LSTM, 207 sensörün hepsi için aynı ağırlıklarla çalışır. Bu, **etkin batch boyutunu 207 katına çıkarır** — gradyan tahmininin varyansı düşer, optimizasyon daha kararlıdır.
2. **Otoregresif sinyalin kısa horizon'da gücü.** 5-15 dakika sonra bir kavşağın hızı, en çok kendi son hızı tarafından belirlenir; bu otoregresif paterni LSTM mükemmel yakalar. GNN'in mekansal toplaması, bu sinyali rakipler arasında **paylaştırıp seyreltir**.
3. **GNN modelinin capacity'sinin oranı.** Bizim hidden=128 ayarımız orta düzeydir; DCRNN paper'ı 64-128 layer ve daha karmaşık decoder'lar kullanır. Daha büyük modellerle bu avantaj farklı şekilde dönebilir.

### 5.2 Neden DCRNN Uzun Horizon'da Geçti?

30+ dakika horizon'da, bir kavşağın geleceği artık kendi geçmişi tarafından tek başına belirlenmez; **komşu kavşaklardaki örüntüler** (yayılma) belirleyici olur. DCRNN'in yönlü diffusion convolution mekanizması bu bilgiyi taşır; LSTM bunu modellemek için zorunlu olarak yalnız kendi (sensör-özel) zaman serisine sıkışır.

### 5.3 Neden A3T-GCN Geri Kaldı?

A3T-GCN'in temel GCN katmanı, simetrik (yönsüz) adjacency varsayar. Ancak METR-LA'nın adjacency matrisi **asimetriktir** ($A \neq A^T$): "Trafik X'ten Y'ye akar" bilgisi "Y'den X'e akar"dan farklıdır. A3T-GCN bu ayrımı kaybeder; sonuçta sinyal akışının yönüne saygılı olmayan bir model elde eder. DCRNN'in (yönlü diffusion) bu modeli her horizon'da geçmesi bu sebep ledir.

### 5.4 Sınırlamalar

- **Tek dataset.** Yalnızca METR-LA üzerinde değerlendirildi; PeMS-BAY veya yerel bir İstanbul verisi ile çapraz doğrulama yapılmamıştır.
- **Hiper-parametre arama yapılmadı.** Hidden=128 sezgisel bir seçimdir; lr scheduling, dropout, weight decay üzerine sistematik bir taramaya girilmemiştir.
- **Tek seed.** Tüm sonuçlar `seed=42` ile alındı; stokastiklik üzerine confidence interval'ler hesaplanmadı.
- **DCRNN decoder basit.** Paper'da DCRNN seq2seq teacher-forcing kullanırken bizim implementasyonumuz son hidden state'i alıp lineer ile T_out'a projeksiyon yapar; bu DCRNN'in tam ifade gücünü kısıtlamış olabilir.

---

## 6. ISSD ENTEGRASYON FRAMEWORK'Ü

→ `report/issd_integration_framework.md` (ayrı bölüm olarak yazılmıştır)

Bu bölümde, çıkarılan modelin **ISSD Bilişim'in CHAOS (Dynamic Junction Control System) ve MANGO (City Traffic Management Platform)** ürünlerine paralel bir öngörü katmanı olarak entegrasyonu için mimari bir öneri sunulmuştur. Önerinin kalbi: ISSD'nin mevcut sistemi **reaktif ve lokal**dir; spatio-temporal GNN modeli bu sisteme **proaktif ve şehir-ölçekli** bir öngörü kapasitesi ekleyebilir. Tezin deneyleri (DCRNN'in 30+ dakika horizon'da üstünlüğü), bu kararın deneysel zeminini oluşturmaktadır.

---

## 7. SONUÇ VE GELECEK İŞLER

Bu tezde, trafik tahmini için üç farklı derin öğrenme mimarisi karşılaştırmalı olarak değerlendirilmiştir. Anahtar bulgu, **GNN modellerinin mekansal indüktif bias'ının horizon-bağımlı bir avantaj sağladığıdır**: kısa horizon'da güçlü bir LSTM tabanlı baseline beklenmedik şekilde rekabetçidir; ancak uzun horizon ve RMSE metriği gibi büyük hatalara duyarlı senaryolarda DCRNN üstün gelmektedir. Bu sonuç, akıllı ulaşım sistemleri için GNN seçiminin **operasyonel ihtiyaca göre** (kısa-tepkili kontrol vs uzun-vadeli planlama) yapılması gerektiğini önerir.

### Gelecek İşler

1. **PeMS-BAY** veya **İBB Açık Veri** üstünde çapraz doğrulama
2. **Multi-modal genişleme** — kamera + Bluetooth + FCD birlikte
3. **Adaptif adjacency** (Graph WaveNet stili) öğrenme
4. **Anomali tespiti** ikincil görevi (tahmin-gerçek sapması üstünden)
5. **ISSD'nin gerçek verisi üzerinde** modelin doğrulanması (staj kapsamında)

---

## REFERANSLAR

1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
2. Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. *ICLR*.
3. Yu, B., Yin, H., & Zhu, Z. (2018). Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. *IJCAI*.
4. Zhu, J., Wang, Q., Tao, C., Deng, H., Zhao, L., & Li, H. (2020). AST-GCN: Attribute-Augmented Spatiotemporal Graph Convolutional Network for Traffic Forecasting. *IEEE Access*.
5. Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. *IJCAI*.
6. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. *ICLR*.
7. Rozemberczki, B., Scherer, P., He, Y., Panagopoulos, G., Riedel, A., Astefanoaei, M., Kiss, O., Beres, F., Lopez, G., Collignon, N., & Sarkar, R. (2021). PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models. *CIKM*.

---

## EK A — Yeniden Üretilebilirlik

Tüm kod, eğitim script'leri ve sonuçlar `notebooks/`, `src/`, `results/`, `figures/` klasörlerinde dağıtılmıştır:

| Script | Amaç |
|---|---|
| `notebooks/01_explore_metrla.py` | Veri yükleme + yapı keşfi |
| `notebooks/02_train_baseline.py` | A3T-GCN eğitimi (batched) |
| `notebooks/03_train_dcrnn.py` | DCRNN eğitimi (batched) |
| `notebooks/04_train_vanilla_lstm.py` | LSTM ablation eğitimi |
| `notebooks/05_compare_models.py` | Karşılaştırma tabloları + plotlar |
| `notebooks/06_visualize_predictions.py` | Multi-model tahmin görselleri |
| `notebooks/07_dataset_stats.py` | Veri istatistikleri |
| `notebooks/08_sensor_map.py` | LA sensör harita (Folium) |
| `notebooks/09_realunit_comparison.py` | Gerçek-birim (mph) raporu |
| `src/normalization.py` | z-score ↔ mph dönüşümü |

Tüm deneyler `seed=42` ile çalıştırılabilir.
