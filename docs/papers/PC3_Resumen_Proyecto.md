# Resumen del Proyecto: PC3 - Predicción de Hotspots de Robo Agravado en Lima

Este documento resume el proceso, las decisiones y los resultados obtenidos en el desarrollo del proyecto para la Práctica Calificada 3 (PC3) del curso de Analítica de Datos, siguiendo las directrices del "Grupo 2" y aplicando los conceptos del Capítulo 3 del libro de referencia, con un enfoque redefinido hacia la predicción de hotspots de criminalidad.

## 1. Objetivo del Proyecto (Redefinido)

Inicialmente, el proyecto se centró en predecir la `modalidad_hecho` de un crimen. Sin embargo, tras una reflexión crítica sobre el valor práctico de esta pregunta, se redefinió el objetivo principal para abordar un problema de mayor impacto y relevancia operacional:

**Nuevo Objetivo:** Desarrollar un modelo que, utilizando datos históricos de ubicación y tiempo, prediga la **densidad de criminalidad (hotspots)** de un tipo de delito específico ("Robo Agravado") en **celdas geográficas del departamento de Lima** para la próxima semana.

**Justificación del Cambio:**
*   La predicción de la modalidad de un crimen a partir de datos iniciales tenía un valor práctico limitado.
*   La predicción de hotspots permite una **asignación proactiva de recursos** (patrullaje policial, serenazgo) a las áreas de mayor riesgo *antes* de que los delitos ocurran, lo que genera un valor operacional significativamente mayor.
*   El enfoque en **Lima** se debe a que un análisis a nivel nacional introducía demasiado ruido geográfico, diluyendo los patrones locales de criminalidad.

## 2. Fase 1: Extracción y Preparación de Datos (Enfoque Hotspot - Lima)

El primer paso fue preparar los datos específicamente para el análisis de hotspots en Lima.

-   **Script:** `01_data_preparation/1b_prepare_geo_data_lima.py`
-   **Motivo:** Cargar los datos brutos de la base de datos, filtrar por un tipo de delito específico y por el departamento de Lima, y limpiar las coordenadas geográficas.
-   **Proceso:**
    1.  Conexión a la base de datos `denuncias_peru` (utilizando `models/common.py`).
    2.  Filtrado de denuncias para el tipo de delito **"ROBO AGRAVADO"** y el departamento **"LIMA"**.
    3.  Limpieza y conversión de las columnas `lat_hecho` y `long_hecho` a formato numérico, eliminando registros con coordenadas inválidas o nulas.
    4.  Conversión de `fecha_hora_hecho` a formato datetime.
    5.  Guardado del dataset limpio y enfocado en `data/processed/robo_agravado_lima.parquet`.
-   **Resultado:** Se obtuvieron **64,812 registros** de "Robo Agravado" en Lima con coordenadas y fechas válidas, listos para la ingeniería de características espacio-temporales.

## 3. Fase 2: Ingeniería de Características Espacio-Temporales (Lima)

Esta fase es crucial para transformar los datos de incidentes individuales en un formato adecuado para la predicción de hotspots.

-   **Script:** `02_feature_engineering/2b_create_grid_features_lima.py`
-   **Motivo:** Crear una representación de los datos que permita al modelo aprender patrones espaciales y temporales de la criminalidad en Lima.
-   **Proceso:**
    1.  **Definición de Grilla Geográfica:** Se dividió el área geográfica de Lima en una cuadrícula de celdas, asignando cada crimen a una `grid_cell_id` basada en `lat_hecho` y `long_hecho` (con un tamaño de celda de 0.005 grados).
    2.  **Agregación Temporal:** Los crímenes se agruparon por `grid_cell_id`, `año` y `semana` para contar el `crime_count` (número de crímenes) en cada celda por semana.
    3.  **Creación de Características de Lag:** Para cada `(celda, semana)`, se calcularon los conteos de crímenes de las 4 semanas anteriores (`crime_count_lag_1` a `crime_count_lag_4`) en la misma celda.
    4.  **Características Temporales:** Se extrajeron el `mes`, `día_de_la_semana` y `is_weekend` (si el inicio de la semana cae en fin de semana).
-   **Resultado:** Un dataset de **39,078 registros** (cada uno representando una celda en una semana) con características espacio-temporales para Lima, guardado en `data/processed/hotspot_features_robo_agravado_lima.parquet`.

## 4. Fase 3: Entrenamiento y Comparación de Modelos de Regresión (Lima)

Con las características preparadas, se entrenaron y compararon varios modelos de regresión para predecir el número de crímenes en Lima.

### 4.1. `RandomForestRegressor` (Lima)
-   **Script:** `03_model_training/3d_train_rf_regressor_lima.py`
-   **Motivo:** Entrenar un modelo de ensamble robusto y versátil para la predicción de conteos.
-   **Resultados:**
    *   MAE: 0.7268
    *   RMSE: 1.5949
    *   R²: 0.6935
    *   Tiempo de Entrenamiento: 0.42s

### 4.2. `KNeighborsRegressor` (Lima)
-   **Script:** `03_model_training/3e_train_knn_regressor_lima.py`
-   **Motivo:** Probar un modelo basado en la distancia, con optimización de hiperparámetros (`GridSearchCV`).
-   **Resultados:**
    *   Mejores parámetros: `{'metric': 'manhattan', 'n_neighbors': 21, 'weights': 'uniform'}`
    *   MAE: 0.7783
    *   RMSE: 1.6326
    *   R²: 0.6789
    *   Tiempo de Entrenamiento: ~30s

### 4.3. `SVR` (Support Vector Regressor) (Lima)
-   **Script:** `03_model_training/3f_train_svr_lima.py`
-   **Motivo:** Probar un modelo basado en máquinas de vectores de soporte para regresión.
-   **Resultados:**
    *   MAE: 0.6773
    *   RMSE: 1.6504
    *   R²: 0.6718
    *   Tiempo de Entrenamiento: ~30s

### 4.4. Comparativa Final de Modelos de Regresión (Datos de Lima)

| Modelo | MAE en Test | RMSE en Test | R² en Test | Tiempo de Entrenamiento (aprox.) |
| :--- | :---: | :---: | :---: | :--- |
| **`RandomForestRegressor` (Lima)** | 0.7268 | **1.5949** | **0.6935** | **0.42s** |
| `KNeighborsRegressor` (Lima) | 0.7783 | 1.6326 | 0.6789 | ~30s |
| `SVR` (Lima) | **0.6773** | 1.6504 | 0.6718 | ~30s |

**Conclusión y Elección del Mejor Modelo:**
El **`RandomForestRegressor` (Lima)** se selecciona como el mejor modelo para la predicción de hotspots. Aunque el `SVR` logró el MAE más bajo, el `RandomForestRegressor` presenta un R² más alto (explicando una mayor varianza en los datos) y un RMSE más bajo (indicando menos errores grandes), lo cual es crucial en la predicción de criminalidad. Además, su tiempo de entrenamiento es drásticamente menor, lo que lo hace más práctico para implementaciones reales. El R² de 0.6935 es un resultado muy sólido para la predicción de un fenómeno tan complejo como la criminalidad.

## 5. Fase 4: Visualización de Hotspots (Lima)

La visualización es clave para interpretar y comunicar el valor del modelo.

-   **Script:** `04_visualization/4b_visualize_hotspots_lima.py` (aún no creado)
-   **Motivo:** Generar una representación visual de los hotspots de criminalidad predichos en Lima y compararlos con los reales.
-   **Proceso:**
    1.  Carga del modelo `RandomForestRegressor` (Lima) y los datos de prueba.
    2.  Selección de la última semana del conjunto de prueba para la visualización.
    3.  Creación de dos mapas de calor (heatmaps) lado a lado: uno para los conteos de crímenes **reales** y otro para los **predichos** por el modelo.
    4.  Guardado de la visualización como `04_visualization/hotspot_comparison_robo_agravado_lima.png`.
-   **Análisis Detallado de la Visualización (Basado en el análisis previo, se espera un comportamiento similar):**
    *   **Capacidad de Localización (Acierto Geográfico):** Se espera que el modelo demuestre una excelente capacidad de discriminación espacial, reflejando los patrones de hotspots reales en la ubicación de los predichos.
    *   **Calibración Cuantitativa (Sesgo de Predicción):** Es probable que el modelo aún presente un sesgo de sobreestimación en la cantidad de crímenes, actuando más como un predictor de riesgo relativo que como un contador absoluto.
    *   **Conclusión Operacional:** El modelo será un excelente predictor de riesgo relativo, útil para priorizar zonas de peligro.

## 6. Conclusiones y Próximos Pasos

Hemos completado con éxito un pipeline de Machine Learning para la predicción de hotspots de "Robo Agravado" en Lima, generando un modelo robusto y con un valor práctico significativo para la asignación de recursos.

**Próximos Pasos:**
1.  **Generar la Visualización Final:** Crear y ejecutar el script `04_visualization/4b_visualize_hotspots_lima.py`.
2.  **Redactar el Artículo Científico (`Paper1.md`):** Con toda la información y análisis detallado, procederemos a escribir el contenido completo del artículo.
3.  **Crear Esquemas de Presentación:** Preparar los esquemas para las dos presentaciones requeridas.
4.  **Revisión de Código:** Asegurar que todos los scripts estén limpios, comentados y listos para la entrega.
5.  **Traducción:** Completar la traducción del `LibroCap3.md`.
6.  **Expansión (Futuro Trabajo):** Considerar la aplicación de este pipeline a otros tipos de delitos (ej. "Extorsión") y explorar métodos para mejorar la calibración cuantitativa del modelo, así como la incorporación de características de lag espacial.