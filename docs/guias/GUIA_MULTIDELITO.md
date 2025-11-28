# Guía Rápida: Sistema Multi-Delito

## 🎯 Objetivo
Este sistema permite ejecutar el **mismo pipeline** para **múltiples delitos** sin duplicar código.

---

## 🚀 Ejecución Rápida

### Opción 1: Pipeline Automático (Recomendado)

```bash
# HURTO (modelo técnicamente superior)
python pipeline_maestro.py hurto

# EXTORSIÓN (mayor impacto político/social)
python pipeline_maestro.py extorsion

# AMBOS (ejecuta todo el pipeline para los dos delitos)
python pipeline_maestro.py --all
```

### Opción 2: Paso a Paso Manual

#### Para HURTO:
```bash
# 1. Preparar datos
python -m 01_data_preparation.prepare_geo_data_universal hurto

# 2. Features con lags temporales
python -m 02_feature_engineering.create_grid_features_universal hurto

# 3. Features con lags espaciales
python -m 02_feature_engineering.create_spatial_features_universal hurto

# 4. Secuencias para LSTM
python -m 02_feature_engineering.create_sequence_features_universal hurto

# 5. Entrenar modelo LSTM
python -m 03_model_training.train_lstm_universal hurto

# 6. Generar visualizaciones
python -m 04_visualization.visualize_hotspots_universal hurto
```

#### Para EXTORSIÓN:
Simplemente reemplaza `hurto` por `extorsion` en todos los comandos arriba.

---

## 📁 Archivos Generados

```
data/processed/
├── hurto_lima.parquet                          # Datos limpios HURTO
├── extorsion_lima.parquet                      # Datos limpios EXTORSIÓN
├── hotspot_features_hurto_lima.parquet         # Features HURTO
├── hotspot_features_extorsion_lima.parquet     # Features EXTORSIÓN
├── hotspot_features_hurto_lima_spatial.parquet
├── hotspot_features_extorsion_lima_spatial.parquet
├── X_sequences_hurto_lima.npy                  # Secuencias HURTO
├── y_targets_hurto_lima.npy
├── X_sequences_extorsion_lima.npy              # Secuencias EXTORSIÓN
└── y_targets_extorsion_lima.npy

models/
├── hotspot_model_hurto_rf_lima.joblib          # Random Forest HURTO
├── hotspot_model_hurto_lstm_tuned_lima.keras   # LSTM HURTO
├── hotspot_model_extorsion_rf_lima.joblib      # Random Forest EXTORSIÓN
└── hotspot_model_extorsion_lstm_tuned_lima.keras  # LSTM EXTORSIÓN

04_visualization/
├── hotspot_comparison_hurto_lima.png
└── hotspot_comparison_extorsion_lima.png
```

---

## ⚙️ Configuración

Edita `config_delitos.py` para:
- Ajustar parámetros de grid (tamaño de celdas)
- Cambiar lags temporales
- Modificar hiperparámetros de modelos
- Agregar nuevos delitos

Ejemplo para agregar un nuevo delito:
```python
DELITOS_CONFIG = {
    # ... delitos existentes ...

    'hurto_celular': {
        'nombre': 'HURTO DE CELULAR',
        'nombre_archivo': 'hurto_celular',
        'descripcion': 'Hurto de celulares en Lima',
        'grid_size': 0.005,
        'lags_temporales': 4,
        'lags_secuencia': 24,
        'color_mapa': 'Purples',
    }
}
```

Luego ejecuta:
```bash
python pipeline_maestro.py hurto_celular
```

---

## 📊 Comparación de Resultados

Después de ejecutar ambos pipelines, compara:

```python
import pandas as pd

# Cargar features
hurto = pd.read_parquet('data/processed/hotspot_features_hurto_lima.parquet')
extorsion = pd.read_parquet('data/processed/hotspot_features_extorsion_lima.parquet')

print(f"HURTO: {len(hurto):,} registros")
print(f"EXTORSIÓN: {len(extorsion):,} registros")
```

---

## 🎓 Para el Paper

Estructura sugerida del documento:

### Sección 4: Experimentación Comparativa

**4.1. Caso 1: HURTO (Delito de Alto Volumen)**
- Características del delito
- Resultados de modelos
- R² esperado: ~0.75-0.80

**4.2. Caso 2: EXTORSIÓN (Delito Emergente)**
- Contexto del crecimiento (+755.6%)
- Resultados de modelos
- Desafíos de predictibilidad

**4.3. Análisis Comparativo**
- Tabla comparativa de métricas
- Discusión sobre predictibilidad vs volumen
- Lecciones sobre aplicabilidad de la metodología

---

## 🔥 Comandos Útiles

### Ver configuración actual
```bash
python config_delitos.py
```

### Solo entrenar modelos (si datos ya existen)
```bash
python pipeline_maestro.py hurto --solo-modelos
python pipeline_maestro.py extorsion --solo-modelos
```

### Verificar datos procesados
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/processed/hurto_lima.parquet'); print(df.info())"
```

---

## ⚡ Troubleshooting

### Error: "Delito no configurado"
→ Verifica que el nombre del delito esté en `config_delitos.py`

### Error: "Archivo no encontrado"
→ Ejecuta primero el paso 1 (preparación de datos)

### Error: Conexión a MySQL
→ Verifica credenciales en `.env`

### Modelo tarda mucho
→ Reduce `epochs` en `MODELOS_CONFIG['lstm']` (config_delitos.py)

---

## 📝 Notas Importantes

1. **HURTO tiene ~81K casos (2024-25)** vs **EXTORSIÓN ~13K casos**
   - Hurto tendrá mejor R²
   - Extorsión tendrá mayor impacto/relevancia

2. **Orden recomendado de ejecución:**
   1. HURTO primero (valida metodología)
   2. EXTORSIÓN después (demuestra versatilidad)

3. **Tiempo estimado por delito:**
   - HURTO: ~15-20 min (más datos)
   - EXTORSIÓN: ~8-12 min (menos datos)

---

## 🎯 Siguiente Paso

**Ejecuta ahora:**
```bash
python -m 01_data_preparation.prepare_geo_data_universal hurto
```

Luego revisa si los datos se generaron correctamente:
```bash
ls -lh data/processed/hurto_lima.parquet
```

¡Listo para comenzar! 🚀
