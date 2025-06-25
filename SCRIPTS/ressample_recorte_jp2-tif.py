
import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import fiona
import numpy as np
import tempfile

# === ENTRADA DEL USUARIO ===
input_folder = input("📁 Introduzca la carpeta con los archivos JP2: ").strip()
output_folder = input("💾 Introduzca la carpeta de salida para los archivos TIF: ").strip()
geojson_path = input("🌍 Introduzca la ruta al archivo GeoJSON del área de recorte: ").strip()

os.makedirs(output_folder, exist_ok=True)

# Leer geometría del GeoJSON
with fiona.open(geojson_path, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]
    dst_crs = shapefile.crs

print("\n--- VERIFICANDO Y PROCESANDO BANDAS ---\n")

# Tomamos la primera banda como referencia
ref_band_path = [f for f in os.listdir(input_folder) if f.endswith(".jp2")][0]
with rasterio.open(os.path.join(input_folder, ref_band_path)) as ref:
    ref_crs = ref.crs
    ref_res = ref.res
    ref_shape = ref.shape

def procesar_banda(in_path, ref_crs, ref_res, ref_shape):
    with rasterio.open(in_path) as src:
        print(f"Procesando: {os.path.basename(in_path)}")
        same_crs = src.crs == ref_crs
        same_res = src.res == ref_res
        print(f" - CRS igual: {same_crs}")
        print(f" - Resolución igual: {same_res}")

        temp_path = None
        if not same_crs or not same_res:
            transform, width, height = calculate_default_transform(
                src.crs, ref_crs, src.width, src.height, *src.bounds, dst_width=ref_shape[1], dst_height=ref_shape[0]
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': ref_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'dtype': 'float32',
                'driver': 'GTiff'
            })

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
                temp_path = tmpfile.name

            with rasterio.open(temp_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )
            src.close()
            src = rasterio.open(temp_path)

        if src.crs != dst_crs:
            print(" - Reproyectando geometría al CRS de la imagen...")
            with fiona.open(geojson_path, "r") as shapefile:
                project = rasterio.warp.transform_geom(
                    shapefile.crs, src.crs, shapes[0]
                )
                out_image, out_transform = mask(src, [project], crop=True)
        else:
            out_image, out_transform = mask(src, shapes, crop=True)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "dtype": "float32"
        })

        tif_name = os.path.basename(in_path).replace(".jp2", ".tif")
        output_tif = os.path.join(output_folder, tif_name)

        with rasterio.open(output_tif, "w", **out_meta) as dst:
            dst.write(out_image.astype(np.float32))

        print(f"✅ Guardado: {output_tif}\n")

        return output_tif

# Procesar recursivamente hasta que todo esté bien alineado
final_outputs = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".jp2"):
        continue

    in_path = os.path.join(input_folder, fname)
    tif_path = procesar_banda(in_path, ref_crs, ref_res, ref_shape)
    final_outputs.append(tif_path)

# Verificación final
print("--- VERIFICACIÓN FINAL DE CONSISTENCIA ---")
ref_shape = None
ref_res = None
for f in final_outputs:
    with rasterio.open(f) as src:
        if ref_shape is None:
            ref_shape = src.shape
            ref_res = src.res
            continue
        if src.shape != ref_shape or src.res != ref_res:
            print(f"❌ Inconsistencia encontrada en: {f}")
        else:
            print(f"✅ {os.path.basename(f)} consistente")
