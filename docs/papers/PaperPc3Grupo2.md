# Predicción de Hotspots de Robo Agravado en Lima: Un Estudio Comparativo de Modelos de Machine Learning y Deep Learning

**Autores:** (Nombres de los integrantes del Grupo 2)

**Fecha:** 16 de Noviembre de 2025

---

### **Resumen (Abstract)**

Este estudio presenta una investigación exhaustiva sobre la predicción de hotspots de criminalidad para el delito de "Robo Agravado" en Lima, Perú. Partiendo de un objetivo inicial de clasificación, el proyecto fue redefinido hacia un problema de regresión espacio-temporal de mayor impacto práctico. Se implementó un pipeline de datos que, tras un análisis exploratorio a nivel nacional, se focalizó en 64,812 denuncias del departamento de Lima. Se evaluó sistemáticamente el rendimiento de una amplia gama de modelos, incluyendo `RandomForestRegressor`, `KNeighborsRegressor`, `SVR`, y arquitecturas de Deep Learning como `LSTM` y `Transformer`. La experimentación demostró que los modelos `LSTM` superaban consistentemente a los enfoques clásicos. El modelo campeón, un `LSTM` optimizado, alcanzó un Coeficiente de Determinación (R²) de **0.6970** y el RMSE más bajo de **1.5858**, estableciendo el mejor balance entre poder explicativo y precisión. Adicionalmente, se investigó el uso de características espaciales y arquitecturas `Transformer` con secuencias temporales largas, lo que proporcionó valiosos aprendizajes sobre la relación entre la complejidad del modelo y el rendimiento para este conjunto de datos específico. El estudio concluye que, si bien la predicción cuantitativa absoluta sigue siendo un desafío, el modelo `LSTM` desarrollado es una herramienta robusta y de alto valor para la predicción de riesgo relativo, permitiendo una asignación de recursos de seguridad más eficiente y proactiva.

---

### **1. Introducción**

La seguridad ciudadana es una de las principales preocupaciones en grandes metrópolis como Lima. La capacidad de anticipar y prevenir la actividad delictiva es fundamental para una gestión policial eficiente. Tradicionalmente, la asignación de recursos de seguridad se ha basado en la experiencia y en análisis históricos estáticos. Sin embargo, el avance en el campo del Machine Learning ofrece la oportunidad de crear modelos predictivos dinámicos que pueden identificar patrones complejos y anticipar la aparición de "hotspots" o zonas de alta concentración de crímenes.

Este proyecto, enmarcado en la Práctica Calificada 3 (PC3), se propuso inicialmente clasificar la modalidad de las denuncias. Sin embargo, se identificó que un objetivo de mayor valor práctico era la predicción de la densidad criminal. Por ello, el objetivo se redefinió para **desarrollar y evaluar un modelo de regresión capaz de predecir la cantidad de robos agravados por semana en una cuadrícula geográfica definida sobre el departamento de Lima.** Este cambio de enfoque permitió transformar un ejercicio académico en una solución con un claro valor operacional potencial.

### **2. Metodología General**

El proceso se dividió en fases principales: preparación de datos, ingeniería de características, y una serie de experimentos iterativos de modelado y evaluación.

#### **2.1. Fuente de Datos y Preparación**
-   **Fuente:** Se utilizó la base de datos `denuncias_peru`, que contiene registros históricos de denuncias policiales.
-   **Limpieza:** Se procesaron las columnas de coordenadas (`lat_hecho`, `long_hecho`) y fecha (`fecha_hora_hecho`) para asegurar su validez y formato.

#### **2.2. Ingeniería de Características Espacio-Temporales**
-   **Definición de Grilla Geográfica:** Se dividió el área geográfica en una cuadrícula de celdas de 0.005 grados (aprox. 550x550 metros).
-   **Agregación Temporal:** Los crímenes se agruparon por celda y semana para contar el `crime_count`, nuestra variable objetivo.
-   **Características de Lag (Históricas):** Se calcularon los conteos de crímenes de las 4 semanas anteriores (`crime_count_lag_1` a `crime_count_lag_4`).

#### **2.3. Estrategia de Evaluación**
-   **División Cronológica:** Se utilizó una división estricta del 80% de los datos más antiguos para entrenamiento y el 20% más reciente para prueba, simulando un escenario de predicción real.
-   **Métricas de Evaluación:** Se utilizaron el Error Absoluto Medio (MAE), la Raíz del Error Cuadrático Medio (RMSE) y el Coeficiente de Determinación (R²).

### **3. Experimentación y Resultados**

Se condujo una serie de experimentos para determinar el mejor enfoque y modelo.

#### **3.1. Experimento 1: Modelado a Nivel Nacional**
Inicialmente, se entrenaron modelos (`RandomForest`, `KNN`, `SVR`) con datos de todo el país. `RandomForestRegressor` resultó ser el mejor, pero su rendimiento fue modesto (R² ≈ 0.66). Se concluyó que la alta variabilidad geográfica entre departamentos introducía ruido, lo que justificó la decisión de enfocar el análisis exclusivamente en Lima.

#### **3.2. Experimento 2: Enfoque en Lima con Modelos Clásicos**
Al re-entrenar los mismos modelos solo con datos de Lima, el rendimiento del `RandomForestRegressor` mejoró, estableciendo un benchmark sólido de **R² = 0.6935**. Esto confirmó que la focalización geográfica era una estrategia correcta.

#### **3.3. Experimento 3: Exploración de Deep Learning (LSTM)**
Para capturar patrones temporales más complejos, se introdujeron redes neuronales recurrentes.
-   Un primer modelo `LSTM` simple ya era competitivo.
-   Un segundo **`LSTM` optimizado** (más profundo y con regularización) superó al `RandomForest`, alcanzando un **R² de 0.6970** y un **RMSE de 1.5858**.

#### **3.4. Experimento 4: Ingeniería de Características Espaciales**
Se enriqueció el dataset añadiendo `spatial_lag_crime_count` (conteo de crímenes en celdas vecinas). Al re-entrenar el `LSTM` con esta nueva característica, se logró el **MAE más bajo de todo el proyecto (0.6589)**, aunque con un ligero detrimento en el R² (0.6925). Esto demostró el valor de la información espacial para reducir el error promedio.

#### **3.5. Experimento 5: Arquitectura Transformer**
Se investigó si una arquitectura de vanguardia como el `Transformer` podría superar a los `LSTM`.
-   **Secuencias Cortas:** Con los datos de lag de 4 semanas, el `Transformer` tuvo un rendimiento muy pobre (R² ≈ 0.64).
-   **Secuencias Largas:** Se generó un nuevo dataset con secuencias de 24 semanas de historial para darle al `Transformer` los datos que teóricamente necesita. El resultado fue fascinante: el modelo alcanzó el MAE más bajo de todos los experimentos (0.5636), pero con un **R² negativo (-0.3007)**, indicando que el modelo no generalizó correctamente y fue incapaz de explicar la variabilidad de los datos.

### **4. Análisis y Selección del Modelo Final**

La siguiente tabla resume los resultados de los modelos más relevantes entrenados con datos de Lima:

| Modelo | MAE en Test | RMSE en Test | R² en Test |
| :--- | :---: | :---: | :---: |
| `RandomForestRegressor` | 0.7268 | 1.5949 | 0.6935 |
| **`LSTM` (Optimizado)** | 0.6690 | **1.5858** | **0.6970** |
| `LSTM` (Espacial) | **0.6589** | 1.5975 | 0.6925 |
| `Transformer` (Sec. Largas) | 0.5636 | 0.8129 | -0.3007 |

Tras una exhaustiva comparación, el **`LSTM` (Optimizado)** del Experimento 3 se selecciona como el **modelo campeón general**. Aunque otros modelos lograron el MAE más bajo, este `LSTM` presentó el **mejor balance de métricas**, con el **R² más alto** y el **RMSE más bajo**. Estas dos últimas métricas son más cruciales para un problema de predicción de riesgos, ya que un R² alto indica un mayor poder explicativo y un RMSE bajo asegura que el modelo no comete errores de gran magnitud, que serían los más costosos en un escenario operacional.

El análisis visual de los mapas de calor del modelo campeón confirma una **excelente precisión geográfica** (sabe *dónde* ocurrirán los crímenes) pero una **pobre calibración cuantitativa** (tiende a sobreestimar *cuántos* crímenes ocurrirán).

### **5. Discusión**

Este proyecto ha sido un viaje iterativo que subraya varias lecciones clave en la ciencia de datos. Primero, la importancia de la **definición del problema**, donde el pivote de clasificación a regresión de hotspots fue fundamental para generar valor. Segundo, la validación de que la **focalización del análisis** (de nacional a Lima) mejora la calidad del modelo. Tercero, y más importante, la demostración empírica de que **la complejidad no siempre es mejor**. A pesar de su sofisticación teórica, la arquitectura `Transformer` no fue la adecuada para este problema, mientras que un `LSTM` bien ajustado encontró el equilibrio perfecto entre capacidad y generalización.

El modelo final debe ser interpretado como un **predictor de riesgo relativo**. No nos dice el número exacto de crímenes, pero sí nos dice, con alta fiabilidad, qué zonas son más peligrosas que otras, lo cual es la información primordial para la asignación de recursos.

### **6. Conclusiones y Trabajo Futuro**

Se ha desarrollado con éxito un modelo `LSTM` que predice hotspots de robo agravado en Lima con un R² de 0.6970, superando a modelos clásicos y a arquitecturas más complejas como el `Transformer`. El modelo es una herramienta viable y robusta para la optimización de la seguridad ciudadana.

Como trabajo futuro, se proponen las siguientes líneas:
-   **Recalibración del Modelo:** Implementar técnicas de post-procesamiento para escalar la salida del modelo y que se ajuste mejor a los conteos reales.
-   **Análisis de Texto (NLP):** Aplicar modelos de NLP, como los descritos por Azmee et al. (2024), para extraer características del campo `observacion_hecho` y añadir una nueva capa de información al modelo.
-   **Modelos Específicos:** Replicar este pipeline para otros tipos de delitos de alto impacto.
