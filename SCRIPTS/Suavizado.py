import rasterio
import numpy as np
from scipy.ndimage import generic_filter
import os

# === 1. INPUT DEL USUARIO ===
archivo_original = input("📂 Introduzca la ruta del archivo .tif de predicción a suavizar: ").strip()
archivo_suavizado = input("💾 Introduzca la ruta donde desea guardar el archivo suavizado: ").strip()

# === 2. FUNCIÓN DE MODO COMPATIBLE ===
def moda_entera(valores):
    valores = valores.astype(np.int32)
    return np.bincount(valores).argmax()

# === 3. APLICAR SUAVIZADO CON FILTRO 3x3 ===
with rasterio.open(archivo_original) as src:
    data = src.read(1)
    perfil = src.profile

    print("🧹 Aplicando suavizado 3x3...")
    data_suavizada = generic_filter(data, moda_entera, size=3, mode='nearest')

# === 4. GUARDAR NUEVO TIF ===
with rasterio.open(archivo_suavizado, "w", **perfil) as dst:
    dst.write(data_suavizada, 1)

print(f"✅ Capa suavizada guardada en:\n{archivo_suavizado}")
