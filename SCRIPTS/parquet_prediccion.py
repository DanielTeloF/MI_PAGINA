import os
import numpy as np
import pandas as pd
import rasterio

# === RUTAS DE ENTRADA ===
RUTA_BANDAS_2018 = r"📁 Introduzca la ruta del directorio correspondiente a bandas año 1"
RUTA_BANDAS_2021 = r"📁 Introduzca la ruta del directorio correspondiente a bandas año 2"
RUTA_INDICES_2018 = r"📁 Introduzca la ruta del directorio correspondiente indices año 1"
RUTA_INDICES_2021 = r"📁 Introduzca la ruta del directorio correspondiente indices año 2"

# === SALIDA ===
RUTA_SALIDA_PARQUET = r"📂 Introduzca la ruta del archivo Parquet correspondiente"

# === FUNCIÓN PARA CARGAR BANDAS ===
def cargar_capas(ruta):
    capas = []
    nombres = []
    for archivo in sorted(os.listdir(ruta)):
        if archivo.endswith(".tif"):
            path = os.path.join(ruta, archivo)
            with rasterio.open(path) as src:
                array = src.read(1)
            capas.append(array)
            nombres.append(os.path.splitext(archivo)[0])
    return np.stack(capas), nombres

# === CARGAR TODAS LAS BANDAS E ÍNDICES ===
print("📥 Cargando bandas e índices...")
bandas_2018, nombres_b18 = cargar_capas(RUTA_BANDAS_2018)
bandas_2021, nombres_b21 = cargar_capas(RUTA_BANDAS_2021)
indices_2018, nombres_i18 = cargar_capas(RUTA_INDICES_2018)
indices_2021, nombres_i21 = cargar_capas(RUTA_INDICES_2021)

# === VERIFICAR DIMENSIONES ===
alto, ancho = bandas_2018.shape[1:]
assert all(arr.shape[1:] == (alto, ancho) for arr in [
    bandas_2021, indices_2018, indices_2021]), "❌ Las dimensiones de las capas no coinciden."

# === APILAR Y FORMATEAR ===
stack = np.concatenate([bandas_2018, bandas_2021, indices_2018, indices_2021], axis=0)
pixeles = stack.reshape(stack.shape[0], -1).T
nombres_columnas = nombres_b18 + nombres_b21 + nombres_i18 + nombres_i21

# === FILTRAR PIXELES CON VALORES VÁLIDOS ===
pixeles_validos = ~np.isnan(pixeles).any(axis=1)
pixeles_filtrados = pixeles[pixeles_validos]
print(f"✅ Total de píxeles válidos: {len(pixeles_filtrados):,}")

# === CREAR Y GUARDAR DATAFRAME ===
df = pd.DataFrame(pixeles_filtrados, columns=nombres_columnas)
df.to_parquet(RUTA_SALIDA_PARQUET)
print(f"💾 Dataset Ribarroja guardado en: {RUTA_SALIDA_PARQUET}")
