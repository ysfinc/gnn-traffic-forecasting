"""
METR-LA Sensör Ağı Haritası
============================

207 LA highway sensörünü gerçek lat/lon konumlarında haritada gösterir.
İki çıktı:
  1. Folium interaktif HTML (hover, zoom — defans için)
  2. Matplotlib statik PNG (tezde figura olarak)

Renk: ortalama hız (kırmızı = yavaş, yeşil = hızlı)
Çizgiler: edge_weight > threshold olan kenarlar
"""

import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import folium

DATA_DIR = "./data/metr-la"
FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

EDGE_VIZ_THRESHOLD = 0.5  # haritada yalnız bu eşik üstü kenarlar çizilir


# ---------------------------------------------------------------------------
# Veri yükle ve adj_mat sırasıyla hizala
# ---------------------------------------------------------------------------
loc_df = pd.read_csv(os.path.join(DATA_DIR, "sensor_graph", "sensor_locations.csv"))
with open(os.path.join(DATA_DIR, "sensor_graph", "adj_mx_mapping.json")) as f:
    mapping = json.load(f)

# adj_mat index 0..206 -> sensor_id string sırası
sensor_ids_in_adj_order = mapping["sensor_ids"]

loc_df["sensor_id_str"] = loc_df["sensor_id"].astype(str)
loc_indexed = (
    loc_df.set_index("sensor_id_str")
    .reindex(sensor_ids_in_adj_order)
    .reset_index()
    .rename(columns={"sensor_id_str": "sensor_id"})
)
# Sanity: 207 satır, hiç NaN yok
assert len(loc_indexed) == 207, f"beklenen 207 satır, bulundu {len(loc_indexed)}"
assert not loc_indexed[["latitude", "longitude"]].isna().any().any(), \
    "Bazı sensör konumları eşleşmedi"
print(f"Konum eşleşmesi OK: {len(loc_indexed)} sensör")

A = np.load(os.path.join(DATA_DIR, "adj_mat.npy"))
X = np.load(os.path.join(DATA_DIR, "node_values.npy"))   # [T, N, F]
mean_speeds = X[:, :, 0].mean(axis=0)                     # [N]
degrees = (A > 0).sum(axis=1)                             # [N]

print(f"Mean speed: min={mean_speeds.min():.1f}, max={mean_speeds.max():.1f}, "
      f"avg={mean_speeds.mean():.1f}")
print(f"Degree    : min={degrees.min()}, max={degrees.max()}, mean={degrees.mean():.2f}")


# ---------------------------------------------------------------------------
# 1. Folium interaktif harita
# ---------------------------------------------------------------------------
center_lat = loc_indexed["latitude"].mean()
center_lon = loc_indexed["longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles="CartoDB Positron",
)

cmap = cm.get_cmap("RdYlGn")
vmin, vmax = float(mean_speeds.min()), float(mean_speeds.max())

# Markers
for i in range(len(loc_indexed)):
    row = loc_indexed.iloc[i]
    lat, lon = row["latitude"], row["longitude"]
    speed = mean_speeds[i]
    sensor_id = row["sensor_id"]
    deg = int(degrees[i])

    norm = (speed - vmin) / max(vmax - vmin, 1e-6)
    color_hex = mcolors.to_hex(cmap(norm))

    popup_html = (
        f"<b>Sensör {sensor_id}</b><br>"
        f"index = {i}<br>"
        f"Ort. hız = {speed:.1f} mph<br>"
        f"Komşu sayısı = {deg}"
    )
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        popup=folium.Popup(popup_html, max_width=230),
        color="black",
        weight=1,
        fillColor=color_hex,
        fillOpacity=0.85,
    ).add_to(m)

# Edges (sadece güçlü olanları, harita çok karışmasın)
n_edges_drawn = 0
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if i >= j:
            continue
        # Yönlü; en az birinde threshold üstü
        max_w = max(A[i, j], A[j, i])
        if max_w > EDGE_VIZ_THRESHOLD:
            folium.PolyLine(
                locations=[
                    [loc_indexed.iloc[i]["latitude"], loc_indexed.iloc[i]["longitude"]],
                    [loc_indexed.iloc[j]["latitude"], loc_indexed.iloc[j]["longitude"]],
                ],
                weight=0.6,
                color="#3a7",
                opacity=0.35,
            ).add_to(m)
            n_edges_drawn += 1

html_path = os.path.join(FIGURES_DIR, "sensor_map.html")
m.save(html_path)
print(f"\nFolium HTML        : {html_path}  ({n_edges_drawn} kenar çizildi)")


# ---------------------------------------------------------------------------
# 2. Matplotlib statik PNG (tez raporuna)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))

# Kenarları arka plana (gri, hafif)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if i >= j:
            continue
        if max(A[i, j], A[j, i]) > EDGE_VIZ_THRESHOLD:
            ax.plot(
                [loc_indexed.iloc[i]["longitude"], loc_indexed.iloc[j]["longitude"]],
                [loc_indexed.iloc[i]["latitude"],  loc_indexed.iloc[j]["latitude"]],
                color="gray", linewidth=0.4, alpha=0.5, zorder=1,
            )

# Sensörler üstte, renkli
sc = ax.scatter(
    loc_indexed["longitude"], loc_indexed["latitude"],
    c=mean_speeds, cmap="RdYlGn", s=70,
    edgecolors="black", linewidth=0.6, zorder=2,
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
cbar.set_label("Ortalama hız (mph)")
ax.set_xlabel("Boylam")
ax.set_ylabel("Enlem")
ax.set_title(f"METR-LA Sensör Ağı — 207 sensör · {n_edges_drawn} güçlü kenar (w > {EDGE_VIZ_THRESHOLD})")
ax.grid(True, alpha=0.3)
ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()

png_path = os.path.join(FIGURES_DIR, "sensor_map_static.png")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Statik PNG         : {png_path}")

print("\nDONE.")
