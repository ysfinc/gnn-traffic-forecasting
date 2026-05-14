# Bölüm: Spatio-Temporal GNN'in ISSD Akıllı Ulaşım Platformuna Entegrasyon Önerisi

*Tezin "Application Framework / Discussion" bölümünde kullanılacak alt-bölüm taslağı.*

---

## 1. Giriş ve Motivasyon

Bu tezde geliştirilen ve değerlendirilen DCRNN tabanlı spatio-temporal grafik sinir ağı modeli, METR-LA gibi standart literatür veri setlerinde rekabetçi tahmin performansı göstermiştir (1-saatlik horizon için ortalama mutlak hata 6.28 mph, RMSE 11.90 mph). Bu sonuç, modelin **gerçek bir endüstriyel akıllı ulaşım platformuna entegrasyonunun da pratik olabileceğini** düşündürmektedir. Bu alt-bölüm, **ISSD Bilişim Elektronik A.Ş.**'nin Türkiye'de ve uluslararası 6 ülkede konuşlandırdığı mevcut platforma — özellikle **CHAOS** (Dynamic Junction Control System) ve **MANGO** (Next Generation City Traffic Management Platform) — bu modelin nasıl entegre edilebileceğine dair mimari bir öneri sunar.

Burada ortaya konan tasarım kodlu uygulama düzeyinde değil, **mimari ve operasyonel bir framework önerisi** düzeyindedir; doğrudan deneysel doğrulama, gerçek ISSD verisinin erişilebilir olduğu bir takip projesinin konusu olabilir.

## 2. Mevcut Sistem (Public Bilgilere Dayanan Anlayışım)

ISSD'nin halka açık yayınlarında ve teknoloji ortaklıklarında (özellikle Intel/Advantech vaka çalışmaları) aktarılan bilgiye göre:

- **CHAOS — Dynamic Junction Control System:** 1000+ kavşakta, kameralar aracılığıyla anlık araç yoğunluğunu ölçer ve sinyalizasyon süresini bu girdiye göre **anlık ayarlar**. Kontrol mantığı **reaktif** (mevcut durumu görüp tepki veren) ve **lokal** (her kavşak büyük ölçüde kendi başına optimize edilen) bir karakterdedir. Performans iyileştirmesi olarak ortalama yaklaşık %30 bekleme süresi azalması raporlanmıştır.
- **MANGO — City Traffic Management Platform:** Şehir-ölçekli bir izleme/kontrol katmanıdır; kavşakların çıktıları burada toplanır.
- **VIERO-AI:** Kamera tabanlı araç sayma (YOLOv8 + OpenVINO), **anlık görüntü işleme** odaklıdır.
- **BLUESIS:** Bluetooth tabanlı seyahat süresi tahmini; kavşaklar arası dolaylı bir ağ ölçümü sunar.
- **FCD (Floating Car Data) analitiği:** Mevcut akışın ölçümü için kullanılır.

Bu kümeden çıkan ortak resim şudur: **mevcut algılama ve analitik yetenekleri güçlüdür**, ancak halka açıklanan kapsamda iki kritik boşluk göze çarpar:

1. **Öngörülü (predictive) tahmin yok.** Sinyalizasyon mevcut akışa tepki verir; 15-60 dk sonrası için aktif tahmin görünmemektedir.
2. **Şehir-ölçekli yayılma (propagation) modellemesi yok.** Her kavşak büyük ölçüde bağımsız optimize edilir; bir kavşaktaki tıkanıklığın komşu kavşaklara nasıl yayılacağı **ön-tahmin** yoluyla modellenmiyor görünmektedir.

## 3. Önerilen Eklenti: Spatio-Temporal Forecasting Katmanı

Bu boşlukları kapatmak için CHAOS/MANGO mimarisine **paralel bir öngörü katmanı** olarak şu şekilde bir spatio-temporal GNN modeli entegre edilebilir:

```
                +-------------------------------------------+
                |              MANGO (Şehir Katmanı)        |
                |                                           |
                |  +-----------------+   +-----------------+|
                |  | Anlık Görünüm   |   | ÖNGÖRÜ KATMANI  ||
                |  | (Mevcut)        |   |   (ÖNERİLEN)    ||
                |  +-----------------+   +-----------------+|
                |          ^                      ^         |
                +----------|----------------------|---------+
                           |                      |
                           |                      |
              +-----------------+    +-----------------------------+
              | CHAOS (Lokal)   |    | Spatio-Temporal GNN Modeli  |
              | per-junction    |    | - Input: T_in adım geçmiş,  |
              | reactive ctrl   |    |   tüm kavşak grafı          |
              +-----------------+    | - Output: T_out adım gelecek|
                  ^                  |   trafik akışı (tüm sensör) |
                  |                  +-----------------------------+
                  |                              ^
                  |                              |
              +-------------------+   +---------------------------+
              | VIERO-AI / FCD /  |-->|  Veri Hattı (Real-time)   |
              | BLUESIS sensörleri|   |  5 dk granülerlikte buffer|
              +-------------------+   +---------------------------+
```

### Bileşenler

**(a) Veri hattı.** Mevcut sensör altyapısı (VIERO-AI'dan gelen araç sayıları, FCD'den gelen hız, BLUESIS'ten gelen seyahat süreleri) zaten 5-dk granülerlikte agregat edilebilen ölçümler üretir. Bu, METR-LA'nın yapısıyla doğrudan uyumludur. Bir akış (streaming) ön-işlemcisi, son 60 dakikayı (T_in = 12 adım) bellekte tutarak modeli besler.

**(b) Adjacency matrisi.** Sensörler arası graf yapısı, ISSD'nin kavşak topolojisi ve mesafe verilerinden (BLUESIS'in zaten doğal olarak ürettiği) statik olarak inşa edilebilir. Daha gelişmiş bir varyantta **adaptif adjacency** (Graph WaveNet stili) öğrenilebilir, ancak ilk uygulama için statik yön-aware (DCRNN tarzı) yeterlidir.

**(c) Model çıktısı.** Tüm 1000+ kavşak için **5, 15, 30, 60 dakika sonrası tahmini akış/hız**. Bu çıktı MANGO'da:
- **Operatör dashboard'una** dalga-uyarısı (örn. "Kavşak X 25 dk sonra kritik tıkanıklığa giriyor") olarak,
- **CHAOS'a** ek bir input olarak (lokal reaktif kontrole **proaktif bias** kazandırmak için)
yerleştirilebilir.

**(d) Eğitim ve güncelleme döngüsü.** İlk model offline eğitilir (geçmiş 6-12 ay verisi üzerinde). Periyodik retraining (örn. haftalık) yeni paternleri yakalar. Online learning, mevsim/tatil paternlerine adaptasyon için ileride değerlendirilebilir.

## 4. Beklenen Etki

Bu tezdeki METR-LA deneylerimiz, mimari kararlar için pratik üç sonuç sunmaktadır:

1. **DCRNN mimarisinin uzun-horizon avantajı.** Vanilla LSTM modeli, 5-45 dk horizon'da DCRNN'i %4-10 farkla geçmiştir; ancak **55 dk ve sonrasında DCRNN üstün gelmektedir** ve RMSE metriğinde (büyük hatalara duyarlı) tüm horizon'larda öne çıkmaktadır. ITS uygulamalarında, özellikle **proaktif sinyal yönetimi için 30-60 dk öne tahminler kritik olduğundan**, DCRNN tercih edilen seçenektir.
2. **Mekansal indüktif bias değerini uzun horizon'da ortaya çıkarır.** Bu, "neden ilave graf yapısı taşımalıyız?" sorusunun deneysel cevabıdır: kısa horizon'da sensörün kendi geçmişi yeterken, uzun horizon'da komşu kavşaklardaki örüntülerin yayılma bilgisi belirleyici hale gelir.
3. **Capacity gereklidir.** Küçük modeller (h=32) bu farkı görünür hale getiremiyor; **h=128 ve ≥10 epoch** ile yapılan eğitim mimari farkını netleştiriyor. ISSD ölçeğinde (1000+ kavşak) bu, prodüksiyon modelinin daha geniş tutulması gerektiğini önerir.

## 5. Operasyonel Hususlar

**Latans.** Tek inference çağrısı RTX 4050 sınıfı bir GPU'da batch başına ~25 ms civarındadır. ISSD'nin 1000+ kavşaklık bir grafında — yaklaşık 5x METR-LA boyutu — bu hâlâ 100-200 ms düzeyindedir, yani 5-dk'lık tahmin döngüsünde sorun değildir.

**Donanım.** Edge yerine merkezi (MANGO katmanı) inference önerilir; tek GPU yeterlidir. Bu, ISSD'nin Advantech/Intel ortaklığındaki mevcut GPU sunucu altyapısıyla uyumludur.

**Sensör hatası ve eksik veri.** METR-LA verisinin yaklaşık %8'i eksik (sensör kesintisi); ISSD'nin sahada da benzer oran beklenebilir. Üretim modeli, **mask-aware loss** veya basit forward-fill imputation ile bu durumda sağlam çalışacak şekilde sertleştirilmelidir.

**Adaptasyon.** Bayram/tatil/hava durumu gibi rejim değişiklikleri için **özel covariates** (haftanın günü, saat, hava sıcaklığı) modele ek feature olarak eklenebilir; bu mimari değişiklik gerektirmez, sadece feature genişlemesi.

## 6. Sınırlamalar ve İleride Çalışılabilecekler

- **METR-LA → ISSD transferi doğrulanmadı.** Bu tez yalnızca standart bir benchmark'ta sonuç verir. Gerçek ISSD verisinde (anonimleştirilmiş 1-2 ay'lık alt-küme yeterlidir) modelin yerel davranışı ayrıca doğrulanmalıdır — bu, takip eden bir staj projesinin doğal kapsamıdır.
- **Cevap zamanı (response time) gereklilikleri.** ISSD'nin CHAOS'unun "kavşak başına saniyeler" düzeyinde tepki verdiği bilinmektedir; öngörü katmanı bu döngünün dışında, paralel bir saatlik öngörü ürettiği için doğrudan kritik yola eklenmez. Yine de operatörün karara katması için **açıklanabilir** olması beneficial olacaktır — örneğin **attention ağırlıklarının görselleştirilmesi** ("Sistem bu tahminini hangi sensörlerin durumuna dayanarak yapıyor?").
- **Çoklu-modlu genişleme.** Mevcut modelin yalnız hız + günün saati feature'ı kullanır; ISSD ekosisteminde **kamera tabanlı yoğunluk + Bluetooth seyahat süresi + FCD** birleştirilirse model daha zengin bir tahmin yapabilir. Bu, **multi-modal/heterogeneous graph extension** gerektirir ve ileri bir araştırma yönüdür.
- **Anomali tespiti.** Tahmin ile gerçek arasındaki büyük sapma, **kaza/olay tespitinin** doğal bir göstergesi olabilir. Bu, ISSD'nin SPECTO (tünel olay tespiti) ürününün şehir-üstü genelleşmesine alanı oluşturabilir — modelin **ikincil bir kullanım** olarak değerlendirilebilir.

## 7. Sonuç

Bu tezde geliştirilen DCRNN tabanlı spatio-temporal GNN modeli, ISSD'nin mevcut akıllı ulaşım platformuna **paralel öngörü katmanı** olarak entegre edilebilir. Bu entegrasyon, mevcut reaktif sinyalizasyon yaklaşımını **proaktif** ve **şehir-ölçekli koordineli** bir versiyona dönüştürür. Önerilen mimari, ISSD'nin halen sahip olduğu sensör altyapısı ve donanım stack'i ile uyumludur ve büyük bir yatırım gerektirmez. Tezin deneysel bölümünde elde edilen sonuçlar (DCRNN'in 30+ dakika horizon'da ve RMSE metriğinde üstünlüğü), bu mimari kararın bilinçli bir tasarım tercihi olarak savunulmasını sağlar.

---

*Bu doküman, tezin uygulama-bağlamı bölümü için temel taslaktır; ISSD'den daha fazla teknik detay alınırsa (NDA dahilinde) buraya daha somut entegrasyon noktaları (API, protokol) eklenebilir.*
