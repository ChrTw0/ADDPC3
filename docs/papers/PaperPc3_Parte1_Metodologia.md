# Paper PC3 - Parte 1: Problema, Metodología y Preparación de Datos

**Autores:** (Nombres de los integrantes del Grupo 2)

**Fecha:** 16 de Noviembre de 2025

---

### **Resumen (Abstract)**

Este estudio presenta una investigación exhaustiva sobre la predicción de hotspots de criminalidad para los delitos de **HURTO** y **EXTORSIÓN** en Lima, Perú. Se detalla la transición de un problema de clasificación a uno de regresión espacio-temporal de mayor impacto, y se establece la metodología de evaluación y preparación de datos que servirá como base para la experimentación de modelos. Un análisis de validación preliminar confirma la idoneidad de los datos, revelando autocorrelación temporal fuerte (r = 0.802 en lag-1), concentración espacial muy marcada (Índice de Gini = 0.771), y persistencia de hotspots en el tiempo (correlación espacial = 0.881), lo que justifica técnicamente la inversión en modelos predictivos. Se describen en profundidad tres enfoques de ingeniería de características: (1) un modelo de características de lag temporal, (2) el enriquecimiento de este modelo con lags espaciales para capturar el contexto de vecindad, y (3) una transformación de los datos a un formato de secuencias temporales largas (24 semanas) diseñado para arquitecturas de Deep Learning avanzadas como LSTMs y Transformers. Esta sección fundamenta el trabajo experimental que se detallará en la Parte 2, estableciendo un marco riguroso para la comparación de modelos.

---

### **1. Introducción**

La seguridad ciudadana es una de las principales preocupaciones en grandes metrópolis como Lima. La capacidad de anticipar y prevenir la actividad delictiva es fundamental para una gestión policial eficiente. Tradicionalmente, la asignación de recursos de seguridad se ha basado en la experiencia y en análisis históricos estáticos. Sin embargo, el avance en el campo del Machine Learning ofrece la oportunidad de crear modelos predictivos dinámicos que pueden identificar patrones complejos y anticipar la aparición de "hotspots" o zonas de alta concentración de crímenes.

Este proyecto, enmarcado en la Práctica Calificada 3 (PC3), se propuso inicialmente clasificar la modalidad de las denuncias. Sin embargo, un análisis preliminar reveló que este enfoque presentaba desafíos significativos, como el severo desbalance de clases y, más importante, un valor práctico limitado para las operaciones de seguridad. Por ello, se tomó la decisión estratégica de redefinir el objetivo hacia un problema de mayor impacto: **desarrollar y evaluar un sistema capaz de predecir la cantidad de crímenes (HURTO y EXTORSIÓN) por semana en una cuadrícula geográfica definida sobre el departamento de Lima.**

La selección de estos dos delitos específicos se fundamenta en:
- **HURTO:** Alto volumen de datos (213,019 registros) que permite entrenamiento robusto, tendencia creciente (+18.5%), alta concentración espacial (Gini=0.806)
- **EXTORSIÓN:** Relevancia crítica actual (+755.6% crecimiento 2020-2025), delito prioritario en agenda nacional de seguridad

Este cambio de enfoque permitió transformar un ejercicio académico en una solución con un claro valor operacional potencial: la asignación proactiva de recursos.

### **2. Metodología General**

El proyecto se estructuró sobre un pipeline de experimentación iterativo, con una metodología de evaluación constante para todos los modelos.

#### **2.1. Fuente de Datos y Preparación Inicial**
-   **Fuente:** Se utilizó la base de datos `denuncias_peru`, que contiene registros históricos de denuncias policiales a nivel nacional, con un total de **7,425,530 registros** que abarcan múltiples tipos de delitos y departamentos del Perú.
-   **Limpieza:** Se realizó un preprocesamiento estándar para asegurar la calidad de los datos, con un foco particular en las columnas de coordenadas (`lat_hecho`, `long_hecho`) y fecha (`fecha_hora_hecho`), convirtiéndolas a formatos numéricos y `datetime` respectivamente, y eliminando registros inválidos.
-   **Focalización Geográfica:** Un análisis exploratorio inicial reveló que la variabilidad de patrones criminales entre diferentes departamentos del país introducía un ruido significativo. Para construir un modelo más preciso y relevante, se tomó la decisión de filtrar el dataset para incluir únicamente las denuncias correspondientes al **departamento de Lima**. Esto resultó en:
    - **HURTO:** 213,019 denuncias válidas
    - **EXTORSIÓN:** 32,021 denuncias válidas
-   **Rango Temporal:** Los datos abarcan desde el **1 de enero de 2020 hasta el 20 de enero de 2025**, cubriendo **5 años** de registros históricos. Esta ventana incluye el período de pandemia COVID-19, lo cual se justifica por:
    - Suficiencia estadística para modelos robustos
    - Capacidad de los modelos ML para aprender patrones en condiciones diversas
    - Split temporal (80/20) asegura que la evaluación se realiza en datos post-pandemia (2024-2025)
    - Captura completa de la tendencia explosiva de EXTORSIÓN (+755.6% desde 2020)

#### **2.2. Validación de Idoneidad de los Datos**

Antes de proceder con el desarrollo de modelos, se realizó un análisis exhaustivo para validar que el problema es técnicamente abordable y que los datos presentan patrones predecibles. Este análisis de validación es crucial para justificar la inversión de recursos en el desarrollo de modelos de Machine Learning.

**Análisis Temporal:**
-   Se identificó una **autocorrelación temporal fuerte** en la serie de crímenes por semana:
    -   Lag 1 semana: **r = 0.802** (correlación muy alta)
    -   Lag 2 semanas: **r = 0.696**
    -   Lag 4 semanas: **r = 0.570**
-   Esta alta autocorrelación confirma que el **pasado reciente predice fuertemente el futuro cercano**, validando el uso de características de lag temporal.

**Análisis Espacial:**
-   La criminalidad presenta una **concentración espacial muy marcada**:
    -   Índice de Gini: **0.7712** (donde 0 = uniformidad total, 1 = máxima concentración)
    -   El **top 5%** de las celdas concentra el **56.2%** de todos los crímenes
    -   El **top 10%** de las celdas concentra el **66.7%** de todos los crímenes
-   Se verificó la **persistencia de hotspots** en el tiempo:
    -   60% de los top 50 hotspots se mantienen constantes entre períodos temporales
    -   Correlación espacial entre períodos: **r = 0.881** (muy alta)
-   Estos hallazgos confirman que los hotspots **no son aleatorios** y son **geográficamente estables**, lo que justifica la predicción espacial.

**Comparación con Modelos Baseline:**
Para validar que un modelo de Machine Learning aporta valor real, se comparó con dos baselines simples:
-   **Baseline 1 - Predecir la Media:** R² = 0.000 (no aporta información)
-   **Baseline 2 - Persistencia (semana anterior):** R² = 0.576
-   **LSTM Optimizado (modelo propuesto):** R² = 0.697
-   El modelo LSTM supera al baseline de persistencia en **21%**, demostrando que el aprendizaje automático aporta valor sobre métodos simples.

**Conclusión de Validación:**
El análisis confirma que **sí vale la pena** desarrollar modelos predictivos para este problema. Los datos presentan patrones temporales y espaciales claros y predecibles, con hotspots estables en el tiempo. Un R² de 0.697 en el contexto de predicción criminal se considera **excelente**, ya que explica aproximadamente el 70% de la variabilidad en un fenómeno social inherentemente complejo.

Para más detalles sobre este análisis de validación, consulte el script `validacion_metodologia_mysql.py` y la visualización generada en `validacion_metodologia_completa.png` (Figura 1).

![Figura 1: Análisis de Validación de Metodología](validacion_metodologia_completa.png)
*Figura 1: Análisis exhaustivo de patrones temporales, espaciales y validación de la idoneidad del enfoque predictivo.*

#### **2.3. Estrategia de Evaluación**
-   **División Cronológica:** Para simular un escenario de predicción real, todos los conjuntos de datos se dividieron de forma cronológica: el 80% de los datos más antiguos se usó para entrenamiento y el 20% más reciente para prueba.
-   **Métricas de Evaluación:** Se seleccionó un conjunto de métricas estándar para problemas de regresión:
    -   **Error Absoluto Medio (MAE):** Mide el promedio del error absoluto entre las predicciones y los valores reales. Es fácil de interpretar.
    -   **Raíz del Error Cuadrático Medio (RMSE):** Similar al MAE, pero penaliza más los errores grandes.
    -   **Coeficiente de Determinación (R²):** Mide la proporción de la varianza en la variable objetivo que es predecible a partir de las características. Un valor cercano a 1 indica un buen ajuste del modelo. En el contexto de predicción criminal, un R² > 0.6 se considera muy bueno, dado que la criminalidad tiene componentes inherentemente aleatorios.

### **3. Ingeniería de Características**

Se exploraron tres enfoques distintos y progresivos para la creación de características, cada uno diseñado para alimentar diferentes tipos de arquitecturas de modelos.

#### **3.1. Enfoque 1: Características de Lag Temporal**
-   **Script:** `02_feature_engineering/2b_create_grid_features_lima.py`
-   **Objetivo:** Crear un dataset tabular clásico para modelos como `RandomForest`.
-   **Proceso:**
    1.  **Discretización Espacio-Temporal:** Se definió una grilla geográfica (celdas de 0.005°) y se agruparon los crímenes por celda y semana para obtener el `crime_count`.
    2.  **Lags Temporales:** Para cada celda y semana, se calcularon los conteos de crímenes de las 4 semanas anteriores (`crime_count_lag_1` a `_4`).
    3.  **Features de Calendario:** Se añadieron características como `mes` y `día_de_la_semana`.
-   **Resultado:** Un DataFrame de 39,078 filas donde cada una representa una celda en una semana, con su historial temporal como columnas.

#### **3.2. Enfoque 2: Enriquecimiento con Lags Espaciales**
-   **Script:** `02_feature_engineering/2c_create_spatial_features_lima.py`
-   **Objetivo:** Añadir contexto geográfico al modelo anterior, bajo la hipótesis de que la criminalidad en una celda está influenciada por la de sus vecinas.
-   **Proceso:**
    1.  Se tomó como base el dataset del Enfoque 1.
    2.  Para cada celda, se identificaron sus 8 celdas adyacentes.
    3.  Se calculó una nueva característica, `spatial_lag_crime_count`, que suma los `crime_count` de las celdas vecinas en la semana inmediatamente anterior.
-   **Resultado:** El mismo DataFrame de 39,078 filas, pero con una columna adicional que captura la "presión" criminal del entorno.

#### **3.3. Enfoque 3: Transformación a Secuencias Temporales Largas**
-   **Script:** `02_feature_engineering/2d_create_sequence_features_lima.py`
-   **Objetivo:** Crear un formato de datos radicalmente diferente, diseñado específicamente para explotar las capacidades de arquitecturas de Deep Learning como `LSTM` y `Transformer`.
-   **Proceso:**
    1.  Se partió de los conteos de crímenes por celda y semana.
    2.  En lugar de lags como columnas, se crearon secuencias deslizantes. Para cada punto a predecir (`y`), la característica de entrada (`X`) se convirtió en un vector que contiene los `crime_count` de las **24 semanas anteriores** (aprox. 6 meses).
    3.  Este proceso se repitió para cada celda y para cada semana posible, generando un gran número de secuencias.
-   **Resultado:** Dos arrays NumPy, `X_sequences_lima.npy` e `y_targets_lima.npy`, donde cada elemento de `X` es una secuencia de 24 pasos de tiempo, lista para ser procesada por una red neuronal.
