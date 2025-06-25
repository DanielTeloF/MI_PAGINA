import pandas as pd
import numpy as np
import rasterio
import joblib
import os

# === RUTAS ===
RUTA_PARQUET_RIBARROJA = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Prediccion_Valencia/Dataset/Datasetribarroja_sin_clases.parquet"
RUTA_PARQUET_TRAIN = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/Entrenamiento/combined_donana_rivas_filtrado.parquet"
RUTA_MODELO = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/ModeloEntrenado/xgboost_donana_rivas_classifier.pkl"
RUTA_ENCODER = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/ModeloEntrenado/label_encoder.pkl"
RUTA_TIF_REF = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Prediccion_Valencia/TIFs/Bandas2018/d18_B02_10m.tif"
RUTA_TIF_SALIDA = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Prediccion_Valencia/Prediccion/prediccion_ribarroja.tif"

# === CARGAR MODELO Y ENCODER ===
print("📦 Cargando modelo y LabelEncoder...")
model = joblib.load(RUTA_MODELO)
encoder = joblib.load(RUTA_ENCODER)

# === CARGAR PARQUETS ===
print("📂 Cargando parquet de entrenamiento (referencia columnas)...")
df_train = pd.read_parquet(RUTA_PARQUET_TRAIN)

print("📂 Cargando parquet de Ribarroja...")
df_ribarroja = pd.read_parquet(RUTA_PARQUET_RIBARROJA)

# === REORDENAR COLUMNAS PARA COINCIDIR CON ENTRENAMIENTO ===
columnas_entrenamiento = [col for col in df_train.columns if col != "target"]
df_ribarroja = df_ribarroja[columnas_entrenamiento]

# === FILTRAR PIXELES VÁLIDOS ===
print("🧼 Eliminando píxeles con NaN...")
X = df_ribarroja.values
mask_validos = ~np.isnan(X).any(axis=1)
X_valid = X[mask_validos]

print(f"🔎 Píxeles válidos para predecir: {len(X_valid)} / {len(X)}")

# === PREDICCIÓN ===
print("🔮 Realizando predicción...")
y_pred_encoded = model.predict(X_valid)

print(f"🔢 Valores únicos predichos (codificados): {np.unique(y_pred_encoded)}")

y_pred_decoded = encoder.inverse_transform(y_pred_encoded)

print(f"🎯 Valores únicos predichos (decodificados): {np.unique(y_pred_decoded)}")

# === CONSTRUIR RÁSTER ===
print("🗺 Cargando referencia TIFF...")
with rasterio.open(RUTA_TIF_REF) as ref:
    alto, ancho = ref.height, ref.width
    meta = ref.meta.copy()
    transform = ref.transform

print(f"📐 Dimensiones TIFF: {alto}x{ancho}")

# Inicializar con 0s y rellenar los válidos
raster_pred_flat = np.zeros((alto * ancho,), dtype=np.uint16)
raster_pred_flat[mask_validos] = y_pred_decoded
raster_pred_2d = raster_pred_flat.reshape((alto, ancho))

# === GUARDAR PREDICCIÓN COMO TIFF ===
print("💾 Guardando predicción como GeoTIFF...")
meta.update({
    "count": 1,
    "dtype": rasterio.uint16,
    "compress": "lzw",
    "nodata": 0
})

with rasterio.open(RUTA_TIF_SALIDA, "w", **meta) as dst:
    dst.write(raster_pred_2d, 1)

print(f"✅ TIFF de predicción guardado en:\n{RUTA_TIF_SALIDA}")

# === VERIFICACIÓN FINAL ===
print(f"🧪 Valores únicos en el raster generado:")
for val in np.unique(raster_pred_2d):
    print(f"  Valor {val}: {(raster_pred_2d == val).sum():,} píxeles")
