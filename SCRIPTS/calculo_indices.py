import os
import rasterio
import numpy as np
from rasterio.enums import Resampling
import glob

# === PARÁMETROS ===
input_base_dir = r"C:\Users\Josem\OneDrive\Escritorio\Proyecto_IA\Prediccion_Valencia\TIFs"
output_dir_base = r"C:\Users\Josem\OneDrive\Escritorio\Proyecto_IA\Prediccion_Valencia\TIFs\Indices"
os.makedirs(output_dir_base, exist_ok=True)

# === FUNCIONES ===
def load_band(path_dict, band):
    return rasterio.open(path_dict[band]).read(1).astype('float32')

def save_index(index_array, ref_path, index_name, output_dir):
    with rasterio.open(ref_path) as src:
        meta = src.meta
    meta.update(dtype='float32', count=1)
    output_path = os.path.join(output_dir, f"{index_name}.tif")
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(index_array.astype('float32'), 1)
    print(f"✅ Guardado índice: {index_name}")

def compute_indices(year_folder):
    folder_path = os.path.join(input_base_dir, year_folder)
    output_dir = os.path.join(output_dir_base, year_folder)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Calculando índices para {year_folder} ---")

    tiff_files = glob.glob(os.path.join(folder_path, "*.tif"))

    # Extrae nombres de bandas (B02, B03, etc.)
    path_dict = {}
    for p in tiff_files:
        name = os.path.basename(p)
        if "_B02" in name:
            path_dict["B02"] = p
        elif "_B03" in name:
            path_dict["B03"] = p
        elif "_B04" in name:
            path_dict["B04"] = p
        elif "_B08" in name:
            path_dict["B08"] = p
        elif "_B11" in name:
            path_dict["B11"] = p

    # Comprobación de bandas necesarias
    required = ["B02", "B03", "B04", "B08", "B11"]
    for r in required:
        if r not in path_dict:
            raise FileNotFoundError(f"No se encontró la banda {r} en {folder_path}")

    B02 = load_band(path_dict, 'B02')
    B03 = load_band(path_dict, 'B03')
    B04 = load_band(path_dict, 'B04')
    B08 = load_band(path_dict, 'B08')
    B11 = load_band(path_dict, 'B11')

    # === Cálculo de índices ===
    ndvi = (B08 - B04) / (B08 + B04 + 1e-6)
    save_index(ndvi, path_dict['B08'], f"NDVI_{year_folder[-4:]}", output_dir)

    ndwi = (B08 - B11) / (B08 + B11 + 1e-6)
    save_index(ndwi, path_dict['B08'], f"NDWI_{year_folder[-4:]}", output_dir)

    ndbi = (B11 - B08) / (B11 + B08 + 1e-6)
    save_index(ndbi, path_dict['B08'], f"NDBI_{year_folder[-4:]}", output_dir)

    bsi = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02) + 1e-6)
    save_index(bsi, path_dict['B08'], f"BSI_{year_folder[-4:]}", output_dir)

    msavi = (2 * B08 + 1 - np.sqrt((2 * B08 + 1)**2 - 8 * (B08 - B04))) / 2
    save_index(msavi, path_dict['B08'], f"MSAVI_{year_folder[-4:]}", output_dir)

    gci = (B08 / (B03 + 1e-6)) - 1
    save_index(gci, path_dict['B08'], f"GCI_{year_folder[-4:]}", output_dir)

    savi = ((B08 - B04) / (B08 + B04 + 0.5)) * 1.5
    save_index(savi, path_dict['B08'], f"SAVI_{year_folder[-4:]}", output_dir)

# === Ejecutar para cada subcarpeta (por ejemplo: rivas2018, donana2021, etc.)
for subfolder in ["Bandas2018","Bandas2021"]:
    compute_indices(subfolder)

