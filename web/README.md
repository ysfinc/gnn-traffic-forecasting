# Tanıtım Sayfası (Landing Page)

Bu klasör, projenin web tabanlı tanıtım sayfasını içerir. Paylaşımlı bir
PHP/Apache hosting (veya benzeri statik dosya sunan herhangi bir altyapı)
üzerine doğrudan yüklenebilir.

## İçerik

```
web/
├── index.html                     Modern tanıtım sayfası (Linear/Stripe-vari tasarım)
├── sensor_map.html                Etkileşimli LA sensör haritası (Folium)
├── comparison_test_metrics.png    Test MAE/RMSE bar grafiği
├── realunit_horizon_mae.png       Ufuk-bazlı MAE eğrileri (mph)
├── sensor_map_static.png          Statik sensör haritası
└── viz_horizon_mae_multi.png      Üç mimari karşılaştırma
```

**Toplam boyut:** ~840 KB · saf HTML/CSS/JS (sunucu yan kodu yok)

## Demoyu Canlıya Almak (3 Adım)

### Adım 1 — Streamlit Cloud'a Demoyu Deploy Et

1. [share.streamlit.io](https://share.streamlit.io) → "Sign in with GitHub"
2. **"Create app"** veya **"New app"**
3. Form'u doldur:
   - Repository: `<kullanıcı>/<repo>`
   - Branch: `main`
   - Main file path: `simulator/app.py`
   - Advanced settings → Python version: `3.12`
4. **Deploy** → 3-5 dakika
5. URL'i kopyala (örn. `https://...-streamlit.app`)

### Adım 2 — URL'i HTML'e Yaz

`index.html` içinde **iki yerde** `STREAMLIT_DEMO_URL` placeholder geçer.
Bunları gerçek URL ile değiştir.

PowerShell ile otomatik:

```powershell
$URL = "https://gnn-traffic-forecasting-XXXXXX.streamlit.app"
(Get-Content web\index.html -Raw) -replace 'STREAMLIT_DEMO_URL', $URL |
  Set-Content web\index.html
```

Bash/macOS ile:

```bash
sed -i "s|STREAMLIT_DEMO_URL|https://gnn-traffic-forecasting-XXXXXX.streamlit.app|g" web/index.html
```

### Adım 3 — Sunucuya Yükle

Klasörü olduğu gibi, web sunucusunun `public_html/<alt-klasör>/` altına yükle.
FTP / cPanel File Manager / `scp` ile.

Örnek hedef: `https://idekyazilim.com/trafik/`

## Akıllı Fallback Davranışı

`index.html` JavaScript ile placeholder algılaması yapar:

- **URL ayarlanmamışsa:** "🚀 Demo hazırlanıyor" kartı + GitHub'a yönlendirici buton
- **URL ayarlanmışsa:** Gerçek Streamlit iframe gömülü

Bu sayede deploy sırasında bile sayfa profesyonel görünür.

## Yerel Test

Tarayıcıda doğrudan açılabilir:

```bash
# Python ile basit HTTP sunucusu
cd web && python -m http.server 8000
# http://localhost:8000
```

veya `index.html`'i çift tıkla.
