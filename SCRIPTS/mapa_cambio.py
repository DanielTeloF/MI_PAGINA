import os
import rasterio
import numpy as np

# === RUTAS DE ENTRADA ===
CLC_PATHS = {
    "rivas": {
        "clc18": r"C:\Users\Josem\OneDrive\Escritorio\Proyecto_IA\Datos\TIFs\CorineRecortado\CLC18_rivas.tif",
        "clc21": r"C:\Users\Josem\OneDrive\Escritorio\Proyecto_IA\Datos\TIFs\CorineRecortado\CLC21_rivas.tif",
    },
    "donana": {
        "clc18": r"C:\Users\Josem\OneDrive\Escritorio\Proyecto_IA\Datos\TIFs\CorineRecortado\CLC18_donana.tif",
        "clc21": r"C:\Users\Josem\OneDrive\Escritorio\Proyecto_IA\Datos\TIFs\CorineRecortado\CLC21_donana.tif",
    }
}

# === RUTA DE SALIDA ===
output_dir = r"C:\Users\Josem\OneDrive\Escritorio\Proyecto_IA\Datos\TIFs\MapaDeCambio"
os.makedirs(output_dir, exist_ok=True)

# === FUNCION PRINCIPAL ===
def generar_mapa_de_cambio(clc18_path, clc21_path, zona):
    print(f"\n📍 Procesando zona: {zona}")

    with rasterio.open(clc18_path) as src18, rasterio.open(clc21_path) as src21:
        assert src18.shape == src21.shape, "❌ Las imágenes no tienen el mismo tamaño"
        assert src18.transform == src21.transform, "❌ Las imágenes no tienen el mismo transform"
        assert src18.crs == src21.crs, "❌ Las imágenes no tienen la misma proyección"

        clc18 = src18.read(1)
        clc21 = src21.read(1)

        # Generar mapa de cambio codificado
        cambio = clc18 * 100 + clc21

        # Guardar
        output_path = os.path.join(output_dir, f"{zona}_mapa_cambio.tif")
        profile = src18.profile
        profile.update(dtype='int32', count=1)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(cambio.astype('int32'), 1)

        print(f"✅ Mapa de cambio guardado: {output_path}")

# === EJECUTAR PARA CADA ZONA ===
for zona in CLC_PATHS:
    generar_mapa_de_cambio(CLC_PATHS[zona]["clc18"], CLC_PATHS[zona]["clc21"], zona)
