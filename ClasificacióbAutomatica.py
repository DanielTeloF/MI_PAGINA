import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import matplotlib.patches as mpatches

# === CONFIGURACIÓN ===
folder =   # ← Reemplaza esta ruta por el directorio que contiene las bandas 

# === LECTURA DE BANDAS ===
with rasterio.open(os.path.join(folder, "B02.tiff")) as blue_src:
    blue = blue_src.read(1).astype('float32')
with rasterio.open(os.path.join(folder, "B03.tiff")) as green_src:
    green = green_src.read(1).astype('float32')
with rasterio.open(os.path.join(folder, "B04.tiff")) as red_src:
    red = red_src.read(1).astype('float32')
with rasterio.open(os.path.join(folder, "B08.tiff")) as nir_src:
    nir = nir_src.read(1).astype('float32')

# === VISUALIZACIÓN RGB ===
rgb = np.stack([red, green, blue], axis=-1)
rgb_norm = rgb / np.max(rgb)
plt.imsave(os.path.join(folder, "rgb_visual.png"), rgb_norm)

# === NDVI ===
ndvi = (nir - red) / (nir + red + 1e-6)
plt.imsave(os.path.join(folder, "ndvi_visual.png"), ndvi, cmap='RdYlGn')

# NDVI con leyenda
plt.figure(figsize=(6, 6))
ndvi_plot = plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar(ndvi_plot, label='NDVI')
plt.title('Índice de Vegetación NDVI')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(folder, "ndvi_leyenda.png"))
plt.close()

# === CLASIFICACIÓN K-MEANS ===
rows, cols = red.shape
X = rgb.reshape(-1, 3)
X_norm = X / np.max(X, axis=0)
labels = KMeans(n_clusters=4, random_state=42).fit_predict(X_norm)
classified = labels.reshape((rows, cols))
plt.imsave(os.path.join(folder, "clasificacion_kmeans.png"), classified, cmap='tab10')

# Clasificación con leyenda
colors = plt.cm.tab10(np.linspace(0, 1, 4))
patches = [mpatches.Patch(color=colors[i], label=f"Clase {i}") for i in range(4)]
plt.figure(figsize=(6, 6))
plt.imshow(classified, cmap='tab10')
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Clasificación K-Means (4 clases)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(folder, "kmeans_leyenda.png"), bbox_inches='tight')
plt.close()

# === EXPORTAR GEOREFERENCIADOS ===
with rasterio.open(os.path.join(folder, "B04.tiff")) as ref:
    profile = ref.profile

# NDVI GeoTIFF
ndvi_profile = profile.copy()
ndvi_profile.update(dtype=rasterio.float32, count=1)
with rasterio.open(os.path.join(folder, "ndvi.tif"), "w", **ndvi_profile) as dst:
    dst.write(ndvi, 1)

# Clasificación GeoTIFF
class_profile = profile.copy()
class_profile.update(dtype=rasterio.uint8, count=1)
with rasterio.open(os.path.join(folder, "clasificacion_kmeans.tif"), "w", **class_profile) as dst:
    dst.write(classified.astype('uint8'), 1)
