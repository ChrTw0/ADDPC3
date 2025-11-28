# Paper PC3 - Parte 2: Experimentación, Resultados y Conclusiones

**Autores:** (Nombres de los integrantes del Grupo 2)

**Fecha:** 16 de Noviembre de 2025

---

### **4. Experimentación y Resultados**

Con los conjuntos de datos preparados como se describió en la Parte 1, se condujo una serie de experimentos iterativos para determinar el enfoque de modelado más efectivo.

#### **4.1. Experimento 1: Benchmark con Modelos Clásicos (Enfoque Lima)**
Utilizando las características de lag temporal (Enfoque 1), se entrenaron tres modelos clásicos para establecer un rendimiento base.

-   **Modelos Evaluados:** `RandomForestRegressor`, `KNeighborsRegressor`, `SVR`.
-   **Resultados Clave:** `RandomForestRegressor` emergió como el claro ganador entre los modelos clásicos, estableciendo un benchmark sólido de **R² = 0.6935**. `SVR` mostró un MAE competitivo, pero su R² fue inferior.
-   **Conclusión:** El `RandomForest` demostró ser un modelo robusto y eficiente, convirtiéndose en el punto de referencia a superar.

#### **4.2. Experimento 2: Exploración de Deep Learning (LSTM)**
Bajo la hipótesis de que una red neuronal diseñada para secuencias podría capturar mejor los patrones temporales, se procedió a evaluar modelos `LSTM`.

-   **Modelos Evaluados:** Se probó un `LSTM` simple y luego un **`LSTM` optimizado** (más profundo, con 2 capas y regularización `Dropout`).
-   **Resultados Clave:** El `LSTM` optimizado superó al `RandomForest` en todas las métricas clave, alcanzando un **R² de 0.6970** y un **RMSE de 1.5858**.
-   **Conclusión:** Se validó que el enfoque de Deep Learning era prometedor, y el `LSTM` optimizado se convirtió en el nuevo modelo campeón.

#### **4.3. Experimento 3: Impacto de Características Espaciales**
Se investigó si añadir contexto geográfico podría mejorar el rendimiento del `LSTM`.

-   **Modelo Evaluado:** El `LSTM` optimizado, re-entrenado con el dataset que incluía `spatial_lag_crime_count` (Enfoque 2).
-   **Resultados Clave:** Este modelo logró el **MAE más bajo de todo el proyecto (0.6589)**. Sin embargo, su R² (0.6925) y RMSE (1.5975) fueron ligeramente inferiores a los del `LSTM` optimizado sin características espaciales.
-   **Conclusión:** Las características espaciales son valiosas para reducir el error promedio, pero pueden no mejorar necesariamente el poder explicativo general del modelo en este caso.

#### **4.4. Experimento 4: El Desafío del Transformer**
Finalmente, se probó una arquitectura de vanguardia, el `Transformer`, para determinar si su mecanismo de auto-atención podía ofrecer una ventaja.

-   **Sub-experimento A (Secuencias Cortas):** Utilizando las características de lag de 4 semanas, el `Transformer` tuvo un rendimiento muy pobre (R² ≈ 0.64), inferior incluso al `RandomForest`.
-   **Sub-experimento B (Secuencias Largas):** Para darle una oportunidad justa, se utilizó el dataset de secuencias de 24 semanas (Enfoque 3). El resultado fue revelador:
    -   **MAE:** 0.5636 (el más bajo de todos los experimentos).
    -   **R²:** -0.3007 (un valor negativo, indicando un fallo en la generalización).
-   **Conclusión:** A pesar de su capacidad teórica, la arquitectura `Transformer` no fue adecuada para este problema. Aunque logró un MAE bajo, su incapacidad para explicar la varianza de los datos (R² negativo) lo descarta como un modelo fiable.

### **5. Análisis y Selección del Modelo Final**

La siguiente tabla consolida los resultados de los modelos más relevantes, permitiendo una selección final basada en evidencia.

| Modelo | MAE en Test | RMSE en Test | R² en Test | Justificación |
| :--- | :---: | :---: | :---: | :--- |
| `RandomForestRegressor` | 0.7268 | 1.5949 | 0.6935 | Mejor modelo clásico, buen benchmark. |
| **`LSTM` (Optimizado)** | 0.6690 | **1.5858** | **0.6970** | **CAMPEÓN:** El mejor balance general. R² más alto y RMSE más bajo. |
| `LSTM` (Espacial) | **0.6589** | 1.5975 | 0.6925 | Logra el MAE más bajo, pero a costa de R² y RMSE. |
| `Transformer` (Sec. Largas) | 0.5636 | 0.8129 | -0.3007 | MAE bajo pero R² negativo. Modelo no fiable. |

Tras una exhaustiva comparación, el **`LSTM` (Optimizado)** del Experimento 2 se selecciona como el **modelo campeón general**. Aunque otros modelos lograron el MAE más bajo, este `LSTM` presentó el **mejor balance de métricas**, con el **R² más alto** y el **RMSE más bajo**. Estas dos últimas métricas son más cruciales para un problema de predicción de riesgos, ya que un R² alto indica un mayor poder explicativo y un RMSE bajo asegura que el modelo no comete errores de gran magnitud, que serían los más costosos en un escenario operacional.

El análisis visual de los mapas de calor del modelo campeón confirma una **excelente precisión geográfica** (sabe *dónde* ocurrirán los crímenes) pero una **pobre calibración cuantitativa** (tiende a sobreestimar *cuántos* crímenes ocurrirán).

### **6. Discusión**

Este proyecto ha sido un viaje iterativo que subraya varias lecciones clave en la ciencia de datos. Primero, la importancia de la **definición del problema**, donde el pivote de clasificación a regresión de hotspots fue fundamental para generar valor. Segundo, la validación de que la **focalización del análisis** (de nacional a Lima) mejora la calidad del modelo. Tercero, y más importante, la demostración empírica de que **la complejidad no siempre es mejor**. A pesar de su sofisticación teórica, la arquitectura `Transformer` no fue la adecuada para este problema, mientras que un `LSTM` bien ajustado encontró el equilibrio perfecto entre capacidad y generalización.

El modelo final debe ser interpretado como un **predictor de riesgo relativo**. No nos dice el número exacto de crímenes, pero sí nos dice, con alta fiabilidad, qué zonas son más peligrosas que otras, lo cual es la información primordial para la asignación de recursos.

### **7. Conclusiones y Trabajo Futuro**

Se ha desarrollado con éxito un modelo `LSTM` que predice hotspots de robo agravado en Lima con un R² de 0.6970, superando a modelos clásicos y a arquitecturas más complejas como el `Transformer`. El modelo es una herramienta viable y robusta para la optimización de la seguridad ciudadana.

Como trabajo futuro, se proponen las siguientes líneas:
-   **Recalibración del Modelo:** Implementar técnicas de post-procesamiento (como `Isotonic Regression`) para escalar la salida del modelo y que se ajuste mejor a los conteos reales, mejorando su utilidad cuantitativa.
-   **Análisis de Texto (NLP):** Aplicar modelos de NLP, como los descritos por Azmee et al. (2024), para extraer características del campo `observacion_hecho` y añadir una nueva capa de información al modelo.
-   **Modelos Específicos:** Replicar este pipeline para otros tipos de delitos de alto impacto, como "Extorsión" o "Hurto de Vehículos", para crear un sistema de predicción de seguridad ciudadana más completo.
