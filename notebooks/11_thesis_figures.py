"""
Tez İçin Ek Görseller
======================

Bu script teze gömülecek ek figüraları üretir:
  1. Dataset dağılım grafikleri (4-panel histogram)
  2. A3T-GCN mimari blok diyagramı
  3. DCRNN mimari blok diyagramı
  4. Vanilla LSTM mimari blok diyagramı
  5. Proje pipeline akış diyagramı
  6. ISSD entegrasyon mimarisi
  7. Message passing kavramsal görseli
  8. Per-sensor MAE heatmap
  9. Region-bazlı yayılma karşılaştırma

Çalıştırma:
    python notebooks/11_thesis_figures.py
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

DATA_DIR = "./data/metr-la"
FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Renk paleti — akademik temiz
CLR = dict(
    bg="#FFFFFF",
    primary="#2E5984",     # mavi
    secondary="#A23B72",   # mor
    accent="#F18F01",      # turuncu
    success="#06A77D",     # yeşil
    danger="#D62246",      # kırmızı
    gray="#5C6B73",
    light_gray="#E0E0E0",
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.dpi": 120,
})


# =============================================================================
# Yardımcı: bloklu mimari diyagram için
# =============================================================================
def block(ax, x, y, w, h, text, color, fontsize=10, edge="black"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor=edge, linewidth=1.5, alpha=0.85,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight="bold", color="white",
            wrap=True)


def arrow(ax, x1, y1, x2, y2, label=None, fontsize=9, color="black"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_length=8,head_width=6",
        color=color, linewidth=1.5, mutation_scale=10,
    ))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my + 0.05, label, fontsize=fontsize,
                style="italic", color=CLR["gray"])


# =============================================================================
# 1) Dataset Distributions (4 panel)
# =============================================================================
print("[1/9] Dataset distributions...")
A = np.load(os.path.join(DATA_DIR, "adj_mat.npy"))
X = np.load(os.path.join(DATA_DIR, "node_values.npy"))   # [T, N, F]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) Speed
ax = axes[0, 0]
speeds = X[:, :, 0].flatten()
speeds_nz = speeds[speeds > 0]   # 0'lar = eksik
ax.hist(speeds_nz, bins=50, color=CLR["primary"], alpha=0.85, edgecolor="white")
ax.set_xlabel("Hız (mph)")
ax.set_ylabel("Frekans")
ax.set_title("(a) Hız Dağılımı")
ax.axvline(speeds_nz.mean(), color=CLR["danger"], linestyle="--",
           linewidth=2, label=f"Ortalama: {speeds_nz.mean():.1f} mph")
ax.legend()

# (b) Edge weights
ax = axes[0, 1]
ew = A[A > 0]
ax.hist(ew, bins=40, color=CLR["secondary"], alpha=0.85, edgecolor="white")
ax.set_xlabel("Kenar ağırlığı")
ax.set_ylabel("Frekans")
ax.set_title("(b) Kenar Ağırlık Dağılımı (mesafe-tabanlı)")
ax.axvline(ew.mean(), color=CLR["danger"], linestyle="--",
           linewidth=2, label=f"Ortalama: {ew.mean():.3f}")
ax.legend()

# (c) Degree distribution
ax = axes[1, 0]
degrees = (A > 0).sum(axis=1)
ax.hist(degrees, bins=range(0, 21), color=CLR["success"], alpha=0.85, edgecolor="white", align="left")
ax.set_xlabel("Düğüm derecesi (komşu sayısı)")
ax.set_ylabel("Düğüm sayısı")
ax.set_title("(c) Düğüm Derece Dağılımı")
ax.set_xticks(range(0, 21, 2))
ax.axvline(degrees.mean(), color=CLR["danger"], linestyle="--",
           linewidth=2, label=f"Ortalama: {degrees.mean():.1f}")
ax.legend()

# (d) Time-of-day
ax = axes[1, 1]
tod = X[:, :, 1].flatten()
ax.hist(tod, bins=48, color=CLR["accent"], alpha=0.85, edgecolor="white")
ax.set_xlabel("Günün saati (0=00:00, 1=24:00)")
ax.set_ylabel("Frekans")
ax.set_title("(d) Günün Saati Dağılımı")

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_dataset_distributions.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 2) A3T-GCN Mimarisi
# =============================================================================
print("[2/9] A3T-GCN architecture diagram...")
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.5)
ax.axis("off")

block(ax, 0.2, 1.5, 1.4, 1.2, "Girdi\nX: [B,N,F,T]", CLR["primary"], fontsize=10)
block(ax, 2.0, 1.5, 1.7, 1.2, "GCN Katmanı\n(yönsüz adj.)", CLR["primary"])
block(ax, 4.1, 1.5, 1.7, 1.2, "TGCN Bloğu\n(zaman güncelleme)", CLR["secondary"])
block(ax, 6.2, 1.5, 1.7, 1.2, "Temporal\nAttention", CLR["accent"])
block(ax, 8.3, 1.5, 1.5, 1.2, "Linear\nHead", CLR["success"])
block(ax, 10.0, 1.5, 0.9, 1.2, "Tahmin\n[B,N,T_out]", CLR["primary"])

arrow(ax, 1.6, 2.1, 2.0, 2.1)
arrow(ax, 3.7, 2.1, 4.1, 2.1)
arrow(ax, 5.8, 2.1, 6.2, 2.1)
arrow(ax, 7.9, 2.1, 8.3, 2.1)
arrow(ax, 9.8, 2.1, 10.0, 2.1)

# Detay metni
ax.text(5.5, 0.6,
        "A3TGCN2 (batched): Her zaman adımı T için GCN+TGCN ile mekansal-zamansal özellik üretilir,\n"
        "12 adımın çıktısı 'temporal attention' ile ağırlıklı toplanır. Lineer head 12 adımlık tahmine çevirir.",
        ha="center", va="center", fontsize=10, style="italic", color=CLR["gray"])

ax.set_title("A3T-GCN Mimari Akışı", fontsize=13, weight="bold", pad=20)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_arch_a3tgcn.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 3) DCRNN Mimarisi
# =============================================================================
print("[3/9] DCRNN architecture diagram...")
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5)
ax.axis("off")

block(ax, 0.2, 2.0, 1.4, 1.0, "Girdi\n[B,T,N,F]", CLR["primary"], fontsize=10)
block(ax, 2.0, 2.0, 2.0, 1.0, "Diffusion\nConvolution\n(K=2 adım)", CLR["secondary"])
block(ax, 4.4, 2.0, 1.6, 1.0, "GRU\nHücresi", CLR["accent"])
block(ax, 6.4, 2.0, 1.8, 1.0, "Son hidden\nseçimi", CLR["accent"])
block(ax, 8.4, 2.0, 1.4, 1.0, "Linear\nHead", CLR["success"])
block(ax, 10.0, 2.0, 0.9, 1.0, "Tahmin\n[B,N,T_out]", CLR["primary"])

arrow(ax, 1.6, 2.5, 2.0, 2.5)
arrow(ax, 4.0, 2.5, 4.4, 2.5)
arrow(ax, 6.0, 2.5, 6.4, 2.5)
arrow(ax, 8.2, 2.5, 8.4, 2.5)
arrow(ax, 9.8, 2.5, 10.0, 2.5)

# Recurrent loop hint
ax.annotate("", xy=(4.4, 1.9), xytext=(6.0, 1.0),
            arrowprops=dict(arrowstyle="->", color=CLR["danger"], linewidth=1.5,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(5.5, 0.9, "h_{t-1}", fontsize=10, color=CLR["danger"],
        style="italic", weight="bold")

# Yönlü adjacency vurgusu
ax.text(3.0, 3.5, "← Yönlü adjacency:\n  A ≠ Aᵀ", fontsize=10,
        weight="bold", color=CLR["danger"])

ax.text(5.5, 0.2,
        "BatchedDCRNN: T_in adım boyunca diffusion conv + GRU recurrent hücresi çalıştırılır.\n"
        "Son hidden state lineer head ile T_out adımlık tahmine projeksiyon yapılır.",
        ha="center", va="center", fontsize=10, style="italic", color=CLR["gray"])

ax.set_title("DCRNN Mimari Akışı (Yönlü Diffusion + GRU)", fontsize=13, weight="bold", pad=20)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_arch_dcrnn.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 4) Vanilla LSTM Mimarisi
# =============================================================================
print("[4/9] Vanilla LSTM architecture diagram...")
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.5)
ax.axis("off")

block(ax, 0.2, 1.5, 1.6, 1.2, "Girdi\n[B,N,F,T]\nReshape:\n[B·N, T, F]", CLR["primary"], fontsize=9)
block(ax, 2.2, 1.5, 2.0, 1.2, "Tek katmanlı\nLSTM\n(paylaşılan params)", CLR["accent"])
block(ax, 4.5, 1.5, 1.6, 1.2, "Son hidden\nh_n", CLR["accent"])
block(ax, 6.4, 1.5, 1.5, 1.2, "Linear\nHead", CLR["success"])
block(ax, 8.2, 1.5, 1.6, 1.2, "Reshape:\n[B,N,T_out]", CLR["primary"], fontsize=9)
block(ax, 10.0, 1.5, 0.9, 1.2, "Tahmin", CLR["primary"])

arrow(ax, 1.8, 2.1, 2.2, 2.1)
arrow(ax, 4.2, 2.1, 4.5, 2.1)
arrow(ax, 6.1, 2.1, 6.4, 2.1)
arrow(ax, 7.9, 2.1, 8.2, 2.1)
arrow(ax, 9.8, 2.1, 10.0, 2.1)

# "Graf bilgisi yok" uyarısı
ax.text(5.5, 3.6, "⚠ Graf bilgisi YOK — her sensör bağımsız, kendi geçmişinden tahmin",
        ha="center", fontsize=11, weight="bold", color=CLR["danger"])

ax.text(5.5, 0.7,
        "Her (batch, sensör) çifti bağımsız sequence olarak işlenir. 207 sensör için aynı LSTM ağırlıkları\n"
        "kullanılır — paylaşılan parametre etkin batch boyutunu 207 kat çıkarır (optimizasyon avantajı).",
        ha="center", va="center", fontsize=10, style="italic", color=CLR["gray"])

ax.set_title("Vanilla LSTM Mimari Akışı (Ablation Baseline — Graf Yok)",
             fontsize=13, weight="bold", pad=20)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_arch_lstm.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 5) Proje Pipeline Akışı
# =============================================================================
print("[5/9] Project pipeline diagram...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis("off")

# Stage 1: Data
block(ax, 0.3, 4.0, 1.8, 1.0, "1. Veri\nMETR-LA\n(207×34272)", CLR["primary"])
# Stage 2: Preprocessing
block(ax, 2.4, 4.0, 1.8, 1.0, "2. Ön İşleme\nNormalizasyon\nSplit (70/10/20)", CLR["primary"])
# Stage 3: Three models
block(ax, 4.5, 5.0, 1.8, 0.8, "A3T-GCN", CLR["secondary"])
block(ax, 4.5, 4.0, 1.8, 0.8, "DCRNN", CLR["accent"])
block(ax, 4.5, 3.0, 1.8, 0.8, "LSTM\n(ablation)", CLR["gray"])
# Stage 4: Evaluation
block(ax, 6.6, 4.0, 1.8, 1.0, "4. Değerlendirme\nMAE, RMSE\nHorizon", CLR["primary"])
# Stage 5: Compare
block(ax, 8.7, 4.0, 1.8, 1.0, "5. Karşılaştırma\nrealunit\n(mph)", CLR["primary"])
# Stage 6: Simulator
block(ax, 10.0, 4.0, 1.7, 1.0, "6. Simülatör\nDashboard", CLR["success"])

# Arrows
arrow(ax, 2.1, 4.5, 2.4, 4.5)
arrow(ax, 4.2, 4.7, 4.5, 5.3)
arrow(ax, 4.2, 4.5, 4.5, 4.3)
arrow(ax, 4.2, 4.3, 4.5, 3.4)
arrow(ax, 6.3, 5.3, 6.6, 4.7)
arrow(ax, 6.3, 4.3, 6.6, 4.5)
arrow(ax, 6.3, 3.4, 6.6, 4.3)
arrow(ax, 8.4, 4.5, 8.7, 4.5)
arrow(ax, 10.5, 4.5, 10.0, 4.5)

# Outputs at bottom
block(ax, 1.0, 1.0, 2.0, 1.2, "Sonuç:\n3 metric JSON\n3 model .pt", CLR["light_gray"], edge="gray")
block(ax, 4.0, 1.0, 2.0, 1.2, "Sonuç:\n14 figura\n(loss, MAE, ...)", CLR["light_gray"], edge="gray")
block(ax, 7.0, 1.0, 2.0, 1.2, "Sonuç:\nInterakt. demo\n(Streamlit)", CLR["light_gray"], edge="gray")
block(ax, 10.0, 1.0, 1.7, 1.2, "Sonuç:\nTez.docx\nrapor", CLR["light_gray"], edge="gray")

for r1, r2 in [(2.0, 5.0), (5.0, 6.0), (8.0, 8.0), (10.8, 10.7)]:
    arrow(ax, r1, 4.0, r2, 2.2, color=CLR["gray"])

ax.set_title("Proje Pipeline'ı: Veriden Tezdeki Görsele",
             fontsize=14, weight="bold", pad=12)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_pipeline.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 6) ISSD Entegrasyon Mimarisi
# =============================================================================
print("[6/9] ISSD integration architecture...")
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.5)
ax.axis("off")

# MANGO katmanı (üst)
mango_box = FancyBboxPatch((0.5, 4.8), 11, 1.4,
                            boxstyle="round,pad=0.05",
                            facecolor="#F0F4F8",
                            edgecolor=CLR["primary"], linewidth=2)
ax.add_patch(mango_box)
ax.text(6.0, 5.95, "MANGO — Şehir Trafik Yönetim Platformu",
        ha="center", fontsize=12, weight="bold", color=CLR["primary"])
# Mango bileşenleri
block(ax, 1.5, 5.0, 2.5, 0.7, "Anlık Görünüm\n(MEVCUT)", CLR["gray"], fontsize=9)
block(ax, 4.5, 5.0, 2.8, 0.7, "Öngörü Katmanı\n(YENİ — DCRNN)", CLR["success"], fontsize=9)
block(ax, 7.8, 5.0, 2.7, 0.7, "Operatör Dashboard\n(uyarılar)", CLR["gray"], fontsize=9)

# CHAOS katmanı
chaos_box = FancyBboxPatch((0.5, 3.0), 11, 1.3,
                            boxstyle="round,pad=0.05",
                            facecolor="#FFF4E6",
                            edgecolor=CLR["accent"], linewidth=2)
ax.add_patch(chaos_box)
ax.text(6.0, 4.05, "CHAOS — Kavşak Sinyal Kontrolü (1000+ kavşak)",
        ha="center", fontsize=12, weight="bold", color=CLR["accent"])
ax.text(6.0, 3.5, "Reaktif: anlık akıştan + öngörüden sinyalizasyon ayarı",
        ha="center", fontsize=10, style="italic", color=CLR["gray"])

# Sensör altyapısı (alt)
sensor_box = FancyBboxPatch((0.5, 1.0), 11, 1.3,
                             boxstyle="round,pad=0.05",
                             facecolor="#E8F4F8",
                             edgecolor=CLR["secondary"], linewidth=2)
ax.add_patch(sensor_box)
ax.text(6.0, 2.05, "Sensör Altyapısı (Mevcut — 5dk veri akışı)",
        ha="center", fontsize=12, weight="bold", color=CLR["secondary"])
# Sensör bileşenleri
block(ax, 1.0, 1.1, 2.0, 0.7, "VIERO-AI\nKamera", CLR["secondary"], fontsize=8)
block(ax, 3.3, 1.1, 2.0, 0.7, "BLUESIS\nBluetooth", CLR["secondary"], fontsize=8)
block(ax, 5.6, 1.1, 2.0, 0.7, "FCD\nGPS", CLR["secondary"], fontsize=8)
block(ax, 7.9, 1.1, 2.0, 0.7, "Diğer\n(SPECTO vb.)", CLR["secondary"], fontsize=8)

# Data akış okları
arrow(ax, 2.0, 1.9, 4.5, 5.0, color=CLR["secondary"])
arrow(ax, 4.5, 1.9, 5.5, 5.0, color=CLR["secondary"])
arrow(ax, 6.0, 1.9, 6.0, 5.0, color=CLR["secondary"])

# Öngörü → CHAOS
arrow(ax, 6.0, 5.0, 6.0, 4.3, color=CLR["success"])

# Notlar
ax.text(6.0, 0.4,
        "Önerilen ek katman (YEŞİL kutu) bu çalışmanın DCRNN modelinin entegrasyonudur.\n"
        "Mevcut sensör altyapısı tek değişmeden 60 dakika öne öngörü üretir, sonuç hem operatöre hem CHAOS'a beslenir.",
        ha="center", fontsize=10, style="italic", color=CLR["gray"])

ax.set_title("ISSD CHAOS/MANGO Platformu için Önerilen Öngörü Katmanı Entegrasyonu",
             fontsize=13, weight="bold", pad=20)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_issd_integration.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 7) Message Passing Kavramsal Görsel
# =============================================================================
print("[7/9] Message passing concept...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

# Center node v
cx, cy = 5, 2.5
ax.add_patch(Circle((cx, cy), 0.5, facecolor=CLR["primary"], edgecolor="black", linewidth=2))
ax.text(cx, cy, "v", ha="center", va="center", fontsize=18,
        weight="bold", color="white")

# Neighbor nodes
neighbors = [(2.5, 1.0, "u₁"), (1.5, 3.0, "u₂"), (3.5, 4.2, "u₃"),
             (6.5, 4.2, "u₄"), (8.0, 3.0, "u₅"), (7.5, 1.0, "u₆")]
for nx, ny, name in neighbors:
    ax.add_patch(Circle((nx, ny), 0.35, facecolor=CLR["accent"],
                         edgecolor="black", linewidth=1.5, alpha=0.85))
    ax.text(nx, ny, name, ha="center", va="center", fontsize=11,
            weight="bold", color="white")
    ax.add_patch(FancyArrowPatch(
        (nx, ny), (cx, cy),
        arrowstyle="->,head_length=8,head_width=6",
        color=CLR["gray"], linewidth=1.3,
        connectionstyle="arc3,rad=0.1",
        shrinkA=12, shrinkB=18, mutation_scale=10,
    ))

# Labels
ax.text(0.2, 0.2,
        "Aggregate: m_v = Σ MESSAGE(h_v, h_u, e_uv) for u in N(v)",
        fontsize=10, style="italic", color=CLR["gray"])
ax.text(0.2, 4.7, "Update: h_v^(l+1) = UPDATE(h_v^(l), m_v)",
        fontsize=10, style="italic", color=CLR["gray"])

ax.text(5.0, 0.55, "Düğüm v komşularından bilgi toplayıp güncellenir",
        ha="center", fontsize=11, color="black")

ax.set_title("GNN'in Temeli: Mesaj İletme (Message Passing)",
             fontsize=13, weight="bold", pad=12)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_message_passing.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 8) Per-sensor MAE heatmap (örneklem)
# =============================================================================
print("[8/9] Per-sensor MAE heatmap...")
# Modeli yüklemeden, mevcut horizon_mae datalarını sensör×horizon olarak göster
# Bunun için 3 model x 12 horizon değerini gösteren bir heatmap

with open("./results/realunit_comparison.json", encoding="utf-8") as f:
    real = json.load(f)

fig, ax = plt.subplots(figsize=(11, 4))
data = np.array([m["horizon_mae_mph"] for m in real["models"]])
labels = [m["model"].replace(" baseline (batched)", "").replace(" — ablation baseline", "")
          for m in real["models"]]
horizons = [(i + 1) * 5 for i in range(12)]

im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=2, vmax=12)
ax.set_xticks(range(12))
ax.set_xticklabels([f"+{h}" for h in horizons])
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Horizon (dakika)")
ax.set_title("Horizon × Model MAE Isı Haritası (mph)",
             fontsize=13, weight="bold")

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                fontsize=9, color="black" if data[i, j] < 8 else "white",
                weight="bold")

cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label("MAE (mph)")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_horizon_heatmap.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 9) Hız profili — Zamansal pattern (24 saatlik)
# =============================================================================
print("[9/9] Hız temporal pattern...")
# 24 saatlik bir periyot pick et ve ortalama hız üzerinden tipik patern göster
# METR-LA 5dk granül = 288 step/gün

steps_per_day = 288
days_to_average = 30  # 30 gün ortalaması

# İlk 30 günü al, gün başlangıçlarını hizala
n_days_avail = X.shape[0] // steps_per_day
days_to_use = min(days_to_average, n_days_avail)

daily_pattern = np.zeros(steps_per_day)
for d in range(days_to_use):
    start = d * steps_per_day
    end = start + steps_per_day
    day_data = X[start:end, :, 0]  # [288, 207]
    # Average across sensors (excluding zeros = missing)
    valid_mask = day_data > 0
    daily_pattern += np.where(valid_mask, day_data, np.nan).mean(axis=1)

daily_pattern /= days_to_use

# Stop wait, NaN issue — fix:
daily_pattern = np.zeros(steps_per_day)
counts = np.zeros(steps_per_day)
for d in range(days_to_use):
    start = d * steps_per_day
    end = start + steps_per_day
    day_data = X[start:end, :, 0]   # [288, 207]
    for t in range(steps_per_day):
        valid = day_data[t, :] > 0
        if valid.sum() > 0:
            daily_pattern[t] += day_data[t, valid].mean()
            counts[t] += 1
daily_pattern = daily_pattern / np.maximum(counts, 1)

fig, ax = plt.subplots(figsize=(11, 4.5))
hours = np.arange(steps_per_day) * 5 / 60   # 0-24 saat aralığı
ax.plot(hours, daily_pattern, color=CLR["primary"], linewidth=2)
ax.fill_between(hours, daily_pattern, alpha=0.2, color=CLR["primary"])

# Rush hour spans
ax.axvspan(7, 10, alpha=0.15, color=CLR["danger"], label="Sabah rush")
ax.axvspan(16, 19, alpha=0.15, color=CLR["accent"], label="Akşam rush")
ax.axhline(daily_pattern.mean(), color=CLR["gray"], linestyle="--",
           linewidth=1, label=f"Ortalama: {daily_pattern.mean():.1f} mph")

ax.set_xlabel("Günün saati")
ax.set_ylabel("Ortalama hız (mph) — tüm sensörler")
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
ax.set_title(f"Tipik Günlük Trafik Profili (30 günlük ortalama)",
             fontsize=13, weight="bold")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig_daily_pattern.png"),
            dpi=130, bbox_inches="tight")
plt.close(fig)

print()
print("=" * 60)
print("✓ 9 ek figura üretildi:")
for f in ["fig_dataset_distributions", "fig_arch_a3tgcn", "fig_arch_dcrnn",
          "fig_arch_lstm", "fig_pipeline", "fig_issd_integration",
          "fig_message_passing", "fig_horizon_heatmap", "fig_daily_pattern"]:
    path = os.path.join(FIGURES_DIR, f"{f}.png")
    size_kb = os.path.getsize(path) / 1024 if os.path.isfile(path) else 0
    print(f"  {f}.png  ({size_kb:.0f} KB)")
print("=" * 60)
