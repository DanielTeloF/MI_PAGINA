import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# === Rutas de entrada/salida ===
RUTA_PARQUET = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/Datos/Entrenamiento/combined_donana_rivas_filtrado.parquet"
RUTA_MODELO_PKL = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/ModeloEntrenado/xgboost_donana_rivas_classifier.pkl"
RUTA_ENCODER_PKL = r"C:/Users/Josem/OneDrive/Escritorio/Proyecto_IA/ModeloEntrenado/label_encoder.pkl"

# === Cargar dataset ===
print(f"📂 Cargando dataset desde: {RUTA_PARQUET}")
df = pd.read_parquet(RUTA_PARQUET)
print(f"✅ Dataset cargado: {len(df):,} filas, {df.shape[1]} columnas")

# === Identificar columna de clases ===
columna_clase = None
for col in df.columns:
    if col.lower() in ['target', 'clase', 'label']:
        columna_clase = col
        break

if columna_clase is None:
    raise ValueError("❌ No se encontró la columna de clases en el dataset.")

# === Codificar etiquetas ===
encoder = LabelEncoder()
df[columna_clase] = encoder.fit_transform(df[columna_clase])
n_classes = len(encoder.classes_)
print(f"🔤 Clases codificadas: {n_classes} clases distintas.")

# === Separar datos y etiquetas ===
X = df.drop(columns=[columna_clase])
y = df[columna_clase]

# === División entrenamiento/validación ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🧪 División: {len(X_train):,} train | {len(X_val):,} val")

# === Entrenar modelo ===
print("🚀 Entrenando modelo XGBoost...")
model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    tree_method='hist',
    verbosity=1,
    num_class=n_classes,
    use_label_encoder=False
)
model.fit(X_train, y_train)
print("✅ Entrenamiento completado.")

# === Guardar modelo y codificador ===
print("💾 Guardando modelo y encoder...")
with open(RUTA_MODELO_PKL, "wb") as f:
    pickle.dump(model, f)

with open(RUTA_ENCODER_PKL, "wb") as f:
    pickle.dump(encoder, f)

print(f"✅ Modelo guardado en: {RUTA_MODELO_PKL}")
print(f"✅ LabelEncoder guardado en: {RUTA_ENCODER_PKL}")
