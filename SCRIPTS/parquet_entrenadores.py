import os
import numpy as np
import pandas as pd
import rasterio

# === RUTAS ===

# Entradas Doñana
RUTA_BANDAS_2018_D = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/TIFs/Bandas/donana2018"
RUTA_BANDAS_2021_D = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/TIFs/Bandas/donana2021"
RUTA_INDICES_2018_D = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/TIFs/Indices/donana2018"
RUTA_INDICES_2021_D = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/TIFs/Indices/donana2021"
RUTA_CAMBIO_D = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/TIFs/MapaDeCambio/donana_mapa_cambio.tif"

# Entradas Rivas
RUTA_BANDAS_2018_R = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/TIFs/Bandas/rivas2018"
RUTA_BANDAS_2021_R = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/TIFs/Bandas/rivas2021"
RUTA_INDICES_2018_R = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/TIFs/Indices/rivas2018"
RUTA_INDICES_2021_R = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/TIFs/Indices/rivas2021"
RUTA_CAMBIO_R = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/TIFs/MapaDeCambio/rivas_mapa_cambio.tif"

# Salidas
RUTA_DONANA_SALIDA = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/Entrenamiento/donana_dataset_filtrado.parquet"
RUTA_RIVAS_SALIDA = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/Entrenamiento/rivas_dataset_filtrado.parquet"
RUTA_COMBINADO_SALIDA = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/Entrenamiento/combined_donana_rivas_filtrado.parquet"

UMBRAL_MINIMO = 5


# === FUNCIONES ===

def cargar_capas(ruta):
    capas, nombres = [], []
    for archivo in sorted(os.listdir(ruta)):
        if archivo.endswith(".tif"):
            path = os.path.join(ruta, archivo)
            with rasterio.open(path) as src:
                capas.append(src.read(1))
            nombres.append(os.path.splitext(archivo)[0])
    return np.stack(capas), nombres

def procesar_zona(ruta_b18, ruta_b21, ruta_i18, ruta_i21, ruta_cambio):
    print(f"\n📥 Procesando zona: {ruta_cambio}")
    b18, nb18 = cargar_capas(ruta_b18)
    b21, nb21 = cargar_capas(ruta_b21)
    i18, ni18 = cargar_capas(ruta_i18)
    i21, ni21 = cargar_capas(ruta_i21)

    alto, ancho = b18.shape[1:]
    assert all(arr.shape[1:] == (alto, ancho) for arr in [b21, i18, i21]), "❌ Las dimensiones no coinciden."

    stack = np.concatenate([b18, b21, i18, i21], axis=0)
    pixeles = stack.reshape(stack.shape[0], -1).T
    columnas = nb18 + nb21 + ni18 + ni21

    with rasterio.open(ruta_cambio) as src:
        cambio = src.read(1)[:alto, :ancho]
    
    clases_raw = cambio.flatten()
    mask_validos = (clases_raw > 0) & (~np.isnan(pixeles).any(axis=1))

    pix_validos = pixeles[mask_validos]
    clases_validas = clases_raw[mask_validos]

    df = pd.DataFrame(pix_validos, columns=columnas)
    df["target"] = clases_validas

    print(f"✅ Total válidos: {len(df):,}")
    return df

def filtrar_por_clase(df, umbral=5):
    conteo = df["target"].value_counts()
    clases_validas = conteo[conteo >= umbral].index
    return df[df["target"].isin(clases_validas)].copy()


# === PROCESAMIENTO DOÑANA ===
df_donana = procesar_zona(RUTA_BANDAS_2018_D, RUTA_BANDAS_2021_D, RUTA_INDICES_2018_D, RUTA_INDICES_2021_D, RUTA_CAMBIO_D)
df_donana_filtrado = filtrar_por_clase(df_donana, umbral=UMBRAL_MINIMO)
df_donana_filtrado.to_parquet(RUTA_DONANA_SALIDA)
print(f"💾 Dataset Doñana guardado en: {RUTA_DONANA_SALIDA}")

# === PROCESAMIENTO RIVAS ===
df_rivas = procesar_zona(RUTA_BANDAS_2018_R, RUTA_BANDAS_2021_R, RUTA_INDICES_2018_R, RUTA_INDICES_2021_R, RUTA_CAMBIO_R)
df_rivas_filtrado = filtrar_por_clase(df_rivas, umbral=UMBRAL_MINIMO)
df_rivas_filtrado.to_parquet(RUTA_RIVAS_SALIDA)
print(f"💾 Dataset Rivas guardado en: {RUTA_RIVAS_SALIDA}")

# === COMBINACIÓN FINAL ===
print("\n🔄 Combinando Doñana + Rivas...")
all_columns = sorted(set(df_donana_filtrado.columns).union(df_rivas_filtrado.columns))
df_donana_filtrado = df_donana_filtrado.reindex(columns=all_columns)
df_rivas_filtrado = df_rivas_filtrado.reindex(columns=all_columns)
df_combinado = pd.concat([df_donana_filtrado, df_rivas_filtrado], ignore_index=True)

df_combinado.to_parquet(RUTA_COMBINADO_SALIDA)
print(f"💾 Dataset combinado guardado en: {RUTA_COMBINADO_SALIDA}")

# === RESUMEN FINAL ===
print("\n📊 Distribución final de clases:")
print(df_combinado['target'].value_counts().sort_index())