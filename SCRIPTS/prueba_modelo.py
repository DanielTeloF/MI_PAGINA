import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import geopandas as gpd
from rasterio.mask import mask

# === CONFIGURACIÓN ===
ruta_base = r"C:\Users\Josem\OneDrive\Escritorio\Proyecto_IA\Prediccion_Valencia\Prediccion"
salidas_base = ruta_base

tif_ref = os.path.join(ruta_base, "Cambio_CLC.tif")
tif_pred = os.path.join(ruta_base, "prediccion_ribarroja.tif")
tif_pred_suav = os.path.join(ruta_base, "prediccion_ribarroja_suavizada.tif")
geojson_roi = os.path.join(ruta_base, "areaRealInteres.geojson")

# === FUNCIONES ===

def cargar_y_recortar_tif(path, shapes):
    with rasterio.open(path) as src:
        out_image, _ = mask(src, shapes, crop=True)
        profile = src.profile
        profile.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": src.transform
        })
    return out_image[0], profile

def preparar_datos(ref, pred, nodata_val):
    rows = min(ref.shape[0], pred.shape[0])
    cols = min(ref.shape[1], pred.shape[1])
    ref = ref[:rows, :cols]
    pred = pred[:rows, :cols]
    mask = (ref != nodata_val) & (pred != nodata_val)
    return ref[mask].flatten(), pred[mask].flatten()

def analizar_comparacion(ref_flat, pred_flat, titulo=""):
    acc = accuracy_score(ref_flat, pred_flat)
    labels = np.unique(np.concatenate((ref_flat, pred_flat)))
    cm = confusion_matrix(ref_flat, pred_flat, labels=labels)
    report = classification_report(ref_flat, pred_flat, labels=labels, output_dict=True, zero_division=0)
    
    print(f"\n📊 Resultados {titulo}")
    print(f"Precisión global: {acc*100:.2f}%")
    
    return report, labels, cm, acc

def guardar_matriz_confusion(cm, labels, titulo, nombre_archivo):
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(titulo)
    plt.xlabel("Predicción")
    plt.ylabel("Referencia")
    plt.tight_layout()
    ruta = os.path.join(salidas_base, nombre_archivo)
    plt.savefig(ruta, dpi=300)
    plt.close()
    print(f"📁 Matriz guardada como: {ruta}")

def graficar_f1(report1, report2, nombre_archivo="comparacion_f1.png"):
    valid_labels = [
        k for k in report1.keys()
        if k.isdigit() and k in report2
    ]

    if not valid_labels:
        print("⚠️ No hay clases válidas en ambos reportes para comparar F1.")
        return

    f1_1 = [report1[k]["f1-score"] for k in valid_labels]
    f1_2 = [report2[k]["f1-score"] for k in valid_labels]
    x = np.arange(len(valid_labels))

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, f1_1, width=0.4, label="IA original")
    plt.bar(x + 0.2, f1_2, width=0.4, label="IA suavizada")
    plt.xticks(x, valid_labels, rotation=90)
    plt.ylabel("F1-Score")
    plt.title("Comparación de F1 por clase")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    ruta = os.path.join(salidas_base, nombre_archivo)
    plt.savefig(ruta, dpi=300)
    print(f"📊 Gráfico de F1 guardado en: {ruta}")
    plt.show()

    print("\n📈 Diferencias de F1 por clase:")
    for i, k in enumerate(valid_labels):
        diff = f1_2[i] - f1_1[i]
        estado = "⬆️ mejora" if diff > 0 else "⬇️ peor" if diff < 0 else "➖ igual"
        print(f"Clase {k}: ΔF1 = {diff:.3f} {estado}")

def exportar_csv(report, nombre_csv, tipo=""):
    filas = []
    for clase, valores in report.items():
        if isinstance(valores, dict) and "f1-score" in valores:
            fila = {
                "Clase": clase,
                "Precision": valores["precision"],
                "Recall": valores["recall"],
                "F1-score": valores["f1-score"],
                "Support": valores["support"],
                "Tipo": tipo
            }
            filas.append(fila)
    df = pd.DataFrame(filas)
    ruta = os.path.join(salidas_base, nombre_csv)
    df.to_csv(ruta, index=False)
    print(f"📄 CSV exportado: {ruta}")


# === 1. LEER POLÍGONO DEL GEOJSON ===
print("📍 Leyendo área de interés desde GeoJSON...")
roi = gpd.read_file(geojson_roi)
shapes = [feature["geometry"] for feature in roi.__geo_interface__["features"]]

# === 2. CARGA Y RECORTE DE RASTERS ===
print("📦 Recortando capas...")
ref_data, ref_profile = cargar_y_recortar_tif(tif_ref, shapes)
pred_data, _ = cargar_y_recortar_tif(tif_pred, shapes)
pred_suav_data, _ = cargar_y_recortar_tif(tif_pred_suav, shapes)

nodata_val = ref_profile.get("nodata", 0)

# === 3. PREPARACIÓN DE DATOS ===
ref_flat1, pred_flat = preparar_datos(ref_data, pred_data, nodata_val)
ref_flat2, pred_suav_flat = preparar_datos(ref_data, pred_suav_data, nodata_val)

# === 4. ANÁLISIS ORIGINAL ===
report_orig, labels_orig, cm_orig, acc_orig = analizar_comparacion(ref_flat1, pred_flat, "IA ORIGINAL")
guardar_matriz_confusion(cm_orig, labels_orig, "Matriz de Confusión (IA ORIGINAL)", "matriz_ia_original.png")
exportar_csv(report_orig, "metricas_ia_original.csv", tipo="IA Original")

# === 5. ANÁLISIS SUAVIZADA ===
report_suav, labels_suav, cm_suav, acc_suav = analizar_comparacion(ref_flat2, pred_suav_flat, "IA SUAVIZADA")
guardar_matriz_confusion(cm_suav, labels_suav, "Matriz de Confusión (IA SUAVIZADA)", "matriz_ia_suavizada.png")
exportar_csv(report_suav, "metricas_ia_suavizada.csv", tipo="IA Suavizada")

# === 6. COMPARACIÓN F1 ===
graficar_f1(report_orig, report_suav, nombre_archivo="comparacion_f1.png")
