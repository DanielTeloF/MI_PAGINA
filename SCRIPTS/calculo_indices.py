
import os
import rasterio
import numpy as np
from rasterio.enums import Resampling
import glob

# === ENTRADA DEL USUARIO ===
input_base_dir = input("📂 Introduzca la ruta de la carpeta donde están los archivos TIF: ").strip()
output_dir_base = input("💾 Introduzca la carpeta donde desea guardar los índices generados: ").strip()

if not os.path.isdir(input_base_dir):
    raise FileNotFoundError(f"La ruta {input_base_dir} no existe.")

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

def compute_indices(folder_path, year_label):
    output_dir = os.path.join(output_dir_base, year_label)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Calculando índices para {year_label} ---")

    tiff_files = glob.glob(os.path.join(folder_path, "*.tif"))

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

    required = ["B02", "B03", "B04", "B08", "B11"]
    for r in required:
        if r not in path_dict:
            raise FileNotFoundError(f"No se encontró la banda {r} en {folder_path}")

    B02 = load_band(path_dict, 'B02')
    B03 = load_band(path_dict, 'B03')
    B04 = load_band(path_dict, 'B04')
    B08 = load_band(path_dict, 'B08')
    B11 = load_band(path_dict, 'B11')

    ndvi = (B08 - B04) / (B08 + B04 + 1e-6)
    save_index(ndvi, path_dict['B08'], f"NDVI_{year_label}", output_dir)

    ndwi = (B08 - B11) / (B08 + B11 + 1e-6)
    save_index(ndwi, path_dict['B08'], f"NDWI_{year_label}", output_dir)

    ndbi = (B11 - B08) / (B11 + B08 + 1e-6)
    save_index(ndbi, path_dict['B08'], f"NDBI_{year_label}", output_dir)

    bsi = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02) + 1e-6)
    save_index(bsi, path_dict['B08'], f"BSI_{year_label}", output_dir)

    msavi = (2 * B08 + 1 - np.sqrt((2 * B08 + 1)**2 - 8 * (B08 - B04))) / 2
    save_index(msavi, path_dict['B08'], f"MSAVI_{year_label}", output_dir)

    gci = (B08 / (B03 + 1e-6)) - 1
    save_index(gci, path_dict['B08'], f"GCI_{year_label}", output_dir)

    savi = ((B08 - B04) / (B08 + B04 + 0.5)) * 1.5
    save_index(savi, path_dict['B08'], f"SAVI_{year_label}", output_dir)

# === PREGUNTAR SUBCARPETAS AL USUARIO ===
print("\n📁 Introduzca los nombres de las subcarpetas a procesar (separadas por coma):")
print("   Ejemplo: Bandas2018,Bandas2021")
subfolders_input = input("👉 Subcarpetas: ").strip()

for subfolder in [s.strip() for s in subfolders_input.split(",")]:
    full_path = os.path.join(input_base_dir, subfolder)
    compute_indices(full_path, subfolder)
