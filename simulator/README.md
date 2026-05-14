# Trafik Simülatörü

Eğitilmiş **DCRNN** modeli üzerinde **interaktif trafik kontrolü simülasyonu**.

## Çalıştırma

Ön koşullar (sırayla tamamlanmış olmalı):

```bash
python notebooks/01_explore_metrla.py    # veri indir
python notebooks/08_sensor_map.py         # sensör konum verisi indir
python notebooks/03_train_dcrnn.py        # DCRNN modeli eğit (tek sefer)
```

Sonra:

```bash
streamlit run simulator/app.py
```

Tarayıcı otomatik açılır (genelde `http://localhost:8501`). Açılmazsa adresi
manuel yaz.

## Ne Yapıyor?

Kullanıcı LA highway ağındaki **belirli bir bölgenin** (Downtown / Hollywood /
I-405 koridoru / yüksek-varyans cluster) sensörlerinin **son 15 dakikalık
hızını** slider'larla değiştirir. DCRNN modeli **bütün 207 sensörü** (graf
yapısı + zaman) kullanarak **60 dakika öne** tahminde bulunur. Sistem bu
tahmine göre **kavşak sinyallerini** renklendirir:

- 🟢 **Yeşil** (≥45 mph): Akış serbest, sinyal normal
- 🟡 **Sarı** (30-45 mph): Yavaşlama gözleniyor
- 🔴 **Kırmızı** (<30 mph): **Öngörülen tıkanıklık** — proaktif müdahale
  gerekir

## Defansta Demo Önerisi

1. **Baseline'ı göster:** Normal trafik durumunda bölge yeşil/sarı dengede
2. **Bir sensöre yoğunluk ekle:** Slider'ı 70 mph'tan 10 mph'a çek → o sensör
   anında kırmızı olur
3. **Yayılmayı göster:** Komşu sensörler de etkileniyor mu? DCRNN'in mekansal
   indüktif bias'ı burada görünür hale gelir
4. **Karşılaştırma:** Eğer LSTM (graf-yok) modeliyle de aynı şeyi yapsaydık,
   tek sensörde değişiklik komşu sensörleri etkilemezdi — *graf yapısının
   katkısının* görsel ispatı

## Konfigürasyon

`app.py` başındaki sabitleri değiştirerek özelleştirebilirsin:

```python
RED_THRESH    = 30   # mph altı → kırmızı
GREEN_THRESH  = 45   # mph üstü → yeşil
SIGNAL_START_STEP = 5    # 30 dk
SIGNAL_END_STEP   = 12   # 60 dk
MOD_LAST_N_STEPS  = 3    # kullanıcı son 15 dk'yı değiştirir
```

## Bilinen Sınırlar

- **Statik tahmin:** Şu an "anlık önümüze bak" yapılıyor; otomatik zaman ilerleme yok. İleride: simülatörün her N saniyede bir tahmin yapıp "oynatma"
  modunda çalışması eklenebilir.
- **Tek model:** Şu an sadece DCRNN. A3T-GCN ve LSTM ile yan yana
  karşılaştırma modu ileride eklenir.
- **Sensör seçimi statik:** Preset bölgeler programatik. Harita üstünden
  drag-select / lasso ileride.
