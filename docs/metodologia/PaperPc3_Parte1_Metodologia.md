# Paper PC3 - Parte 1: Clasificación de Hotspots de Criminalidad - Metodología

**Autores:** (Nombres de los integrantes del Grupo 2)

**Fecha:** 27 de Enero de 2025

**Capítulo:** 3 - Classification (Hands-On Machine Learning)

---

### **Resumen (Abstract)**

Este estudio presenta una implementación exhaustiva de técnicas de clasificación aplicadas a la predicción de hotspots de criminalidad para los delitos de **HURTO** y **EXTORSIÓN** en Lima, Perú. Respondiendo a la necesidad operacional de sistemas de decisión para la asignación de recursos de seguridad, se desarrollaron **tres problemas de clasificación complementarios** con valor práctico claro: (1) **Nivel de Riesgo** (clasificación multiclase de 4 niveles) para zonificación y asignación proporcional de recursos, (2) **Hotspot Crítico** (clasificación binaria) para decisiones de intervención inmediata, y (3) **Tendencia de Riesgo** (clasificación multiclase de 3 niveles) como sistema de alerta temprana para identificar zonas en deterioro. Se implementaron **7 algoritmos de clasificación** del Capítulo 3 (SGD, Logistic Regression, Random Forest, Gradient Boosting, KNN, Decision Tree, AdaBoost) aplicados a los 3 problemas y 2 delitos, resultando en **42 modelos de clasificación**. Un análisis de validación preliminar confirma la idoneidad de los datos, revelando autocorrelación temporal fuerte (r = 0.802 en lag-1), concentración espacial muy marcada (Índice de Gini = 0.771), y persistencia de hotspots (correlación espacial = 0.881). Esta sección fundamenta el trabajo experimental de clasificación que se detallará en la Parte 2, estableciendo un marco riguroso para la evaluación y comparación de modelos con enfoque en métricas operacionales (Precision, Recall, F1-Score).

---

### **1. Introducción**

La seguridad ciudadana es una de las principales preocupaciones en grandes metrópolis como Lima. La capacidad de anticipar y prevenir la actividad delictiva es fundamental para una gestión policial eficiente. Tradicionalmente, la asignación de recursos de seguridad se ha basado en la experiencia y en análisis históricos estáticos. Sin embargo, el avance en técnicas de **clasificación supervisada** (Capítulo 3, Hands-On Machine Learning) ofrece la oportunidad de crear sistemas de decisión automáticos que pueden identificar patrones complejos y clasificar zonas según su nivel de riesgo criminal.

Este proyecto, enmarcado en la Práctica Calificada 3 (PC3) y alineado con el **Capítulo 3: Classification**, desarrolla un sistema integral de clasificación de hotspots criminales. En lugar de predecir cantidades exactas de crímenes (regresión), el enfoque se centra en **clasificar zonas geográficas en categorías de riesgo** que permitan tomar decisiones operacionales concretas. Este enfoque de clasificación es más robusto ante la variabilidad inherente de los datos criminales y genera outputs directamente accionables para la asignación de recursos.

#### **1.1. Proceso de Selección de Delitos y Enfoque Metodológico**

La selección de **HURTO** y **EXTORSIÓN** como delitos objetivo, así como la decisión de adoptar un enfoque exclusivo de clasificación (en lugar de regresión), fue el resultado de un **análisis exploratorio exhaustivo** previo que evaluó todos los delitos reportados en Lima durante el período 2020-2025.

**Análisis Exploratorio Inicial - Evaluación de Candidatos:**

Se ejecutaron tres scripts de análisis crítico para fundamentar estas decisiones metodológicas clave:

1. **`analisis_critico_problema.py`** - Evaluación cuantitativa de todos los delitos:
   - Analizó 7.4M registros de denuncias en Lima
   - Calculó un score compuesto considerando: volumen de datos, concentración espacial (Gini), autocorrelación temporal, y tendencia
   - **Resultado clave:** HURTO obtuvo el score más alto de predictibilidad (**74.06 puntos**), superando significativamente a Robo Agravado (candidato inicial)
   - Recomendación del análisis: "**Clasificación sobre regresión**" debido a la naturaleza multimodal de la distribución criminal

2. **`analisis_tendencias_contexto.py`** - Análisis temporal y contexto socio-político:
   - Evaluó tendencias 2020-2025 para todos los tipos delictivos
   - **HURTO:** Tendencia creciente sostenida (+18.5%), score de estabilidad **71.18** (mejor del dataset)
   - **EXTORSIÓN:** Crecimiento explosivo de **+755.6%** en 5 años, convirtiéndose en prioridad nacional de seguridad
   - **Robo Agravado:** Tendencia decreciente marcada (-40.1%), descartando su uso como variable objetivo
   - Validación de suficiencia de datos post-COVID para modelamiento robusto

3. **`validacion_metodologia_mysql.py`** - Validación de idoneidad técnica:
   - Confirmó alta autocorrelación temporal (r = 0.802 en lag-1)
   - Concentración espacial extrema (Gini = 0.7712)
   - Persistencia de hotspots entre períodos (correlación espacial = 0.881)
   - **Conclusión:** Los datos presentan patrones predecibles suficientes para justificar modelos de ML

**Decisiones Fundamentadas:**

Basándose en esta evidencia empírica, se tomaron las siguientes decisiones metodológicas:

| Decisión | Justificación Basada en Datos |
|----------|-------------------------------|
| **HURTO como delito primario** | Score predictibilidad 74.06 (máximo), 213,019 registros, tendencia estable +18.5%, concentración espacial Gini=0.806 |
| **EXTORSIÓN como delito secundario** | Relevancia socio-política crítica (+755.6% crecimiento), suficiencia de datos (32,021 registros), urgencia de sistema de alerta temprana |
| **Descarte de Robo Agravado** | Tendencia decreciente (-40.1%), score inferior a HURTO, menor concentración espacial |
| **Clasificación sobre regresión** | Distribución multimodal del crimen, outputs accionables (decisión binaria/categórica), robustez ante variabilidad |

Esta fundamentación asegura que el trabajo no es una aplicación arbitraria de técnicas de ML, sino el resultado de un **proceso de investigación riguroso** que evalúa alternativas y selecciona el enfoque óptimo basándose en evidencia cuantitativa.

Para detalles completos del análisis exploratorio, consulte los scripts mencionados en el repositorio del proyecto.

**Objetivo Principal:** Desarrollar y evaluar **tres sistemas de clasificación complementarios** aplicados a los delitos de HURTO y EXTORSIÓN en Lima:

1. **Clasificación de Nivel de Riesgo (Multiclase - 4 niveles):**
   - Pregunta operacional: *"¿Qué nivel de recursos necesita esta zona?"*
   - Clases: Bajo (0-2 crímenes), Medio (3-5), Alto (6-10), Muy Alto (>10)
   - Valor: Zonificación para asignación proporcional de recursos

2. **Clasificación de Hotspot Crítico (Binaria):**
   - Pregunta operacional: *"¿Debo intervenir en esta zona esta semana?"*
   - Clases: Normal (≤5 crímenes), Crítico (>5 crímenes)
   - Valor: Decisión binaria clara para despliegue de operativos especiales

3. **Clasificación de Tendencia de Riesgo (Multiclase - 3 niveles):**
   - Pregunta operacional: *"¿Esta zona está mejorando o empeorando?"*
   - Clases: Descenso, Estable, Escalada
   - Valor: Sistema de alerta temprana para identificar zonas en deterioro

La selección de estos dos delitos específicos se fundamenta en:
- **HURTO:** Alto volumen de datos (213,019 registros → 709,678 registros procesados) que permite entrenamiento robusto, tendencia creciente (+18.5%), alta concentración espacial (Gini=0.806)
- **EXTORSIÓN:** Relevancia crítica actual (+755.6% crecimiento 2020-2025 → 107,907 registros procesados), delito prioritario en agenda nacional de seguridad

Este enfoque de clasificación transforma datos criminales en **sistemas de decisión operacionales**, cumpliendo con los requisitos del Capítulo 3 del libro (clasificación binaria y multiclase) y generando valor práctico real para la seguridad ciudadana.

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

#### **2.3. Estrategia de Evaluación para Clasificación**

**División Cronológica:** Para simular un escenario de predicción real, todos los conjuntos de datos se dividieron de forma cronológica: el 80% de los datos más antiguos se usó para entrenamiento (2020-2024) y el 20% más reciente para prueba (2024-2025).

**Métricas de Evaluación para Clasificación (Capítulo 3):**

Siguiendo las mejores prácticas del Capítulo 3, se utilizaron las siguientes métricas estándar para problemas de clasificación:

1. **Accuracy (Exactitud):** Proporción de predicciones correctas sobre el total. Útil como métrica general, pero puede ser engañosa con clases desbalanceadas.

2. **Precision (Precisión):**
   - Fórmula: TP / (TP + FP)
   - Interpretación operacional: "De las zonas que clasificamos como peligrosas, ¿qué porcentaje realmente lo es?"
   - Crítico para evitar falsas alarmas que desperdicien recursos

3. **Recall (Sensibilidad):**
   - Fórmula: TP / (TP + FN)
   - Interpretación operacional: "De todas las zonas realmente peligrosas, ¿qué porcentaje detectamos?"
   - Crítico para no perder hotspots que requieren intervención

4. **F1-Score:**
   - Fórmula: 2 × (Precision × Recall) / (Precision + Recall)
   - **Métrica principal** para selección de modelos
   - Balancea Precision y Recall, ideal para datos desbalanceados

5. **Confusion Matrix:** Visualización de TP, TN, FP, FN para análisis detallado de errores

**Criterios de Éxito:**
- F1-Score > 0.85: Modelo listo para producción
- F1-Score 0.70-0.85: Modelo funcional, considerar optimización
- F1-Score < 0.70: Requiere mejora significativa

**Manejo de Desbalance de Clases:**
- HURTO: 82.78% clase Bajo, 1.67% clase Muy Alto
- EXTORSIÓN: 82.94% clase Bajo, 1.81% clase Muy Alto
- Estrategia: F1-Score weighted para considerar distribución de clases
- Validación: Análisis de Precision/Recall por clase individual

### **3. Ingeniería de Características y Targets**

#### **3.1. Creación de Features Predictivas (X)**

**Pipeline de Feature Engineering:**

1. **Discretización Espacio-Temporal:**
   - Grid geográfico: Celdas de 0.005° (~555m × 555m)
   - Agregación temporal: Conteo de crímenes por celda y semana
   - Resultado HURTO: 709,678 observaciones (celda-semana)
   - Resultado EXTORSIÓN: 107,907 observaciones (celda-semana)

2. **Features Temporales (Lags):**
   - `crime_count_lag_1`: Crímenes semana anterior
   - `crime_count_lag_2`: Crímenes 2 semanas atrás
   - `crime_count_lag_3`: Crímenes 3 semanas atrás
   - `crime_count_lag_4`: Crímenes 4 semanas atrás
   - Justificación: Autocorrelación temporal fuerte (r = 0.802 en lag-1)

3. **Features de Calendario:**
   - `mes`: Estacionalidad mensual (1-12)
   - `dia_semana`: Patrón semanal (0-6)

**Feature Vector Final:** 6 features por observación (4 lags + 2 temporales)

#### **3.2. Creación de Targets de Clasificación (y)**

Esta es la **innovación metodológica clave**: transformar conteos continuos (`crime_count`) en categorías discretas con significado operacional.

**Target 1: Nivel de Riesgo (4 clases)**

```python
def crear_target_nivel_riesgo(crime_counts):
    bins = [0, 2, 5, 10, inf]
    labels = [0, 1, 2, 3]  # Bajo, Medio, Alto, Muy Alto
    return pd.cut(crime_counts, bins=bins, labels=labels)
```

Distribución (HURTO):
- Clase 0 (Bajo): 587,493 (82.78%)
- Clase 1 (Medio): 89,831 (12.66%)
- Clase 2 (Alto): 20,478 (2.89%)
- Clase 3 (Muy Alto): 11,876 (1.67%)

**Target 2: Hotspot Crítico (2 clases - Binaria)**

```python
def crear_target_hotspot_critico(crime_counts, umbral=5):
    return (crime_counts > umbral).astype(int)
```

Distribución (HURTO):
- Clase 0 (Normal): 677,324 (95.44%)
- Clase 1 (Crítico): 32,354 (4.56%)

**Target 3: Tendencia de Riesgo (3 clases)**

```python
def crear_target_tendencia(df):
    # Promedio histórico por celda (últimas 4 semanas)
    promedio = rolling_mean(crime_count, window=4)
    ratio = crime_count_actual / promedio

    if ratio < 0.7: return 0    # Descenso
    elif ratio <= 1.3: return 1  # Estable
    else: return 2               # Escalada
```

Distribución (HURTO):
- Clase 0 (Descenso): 63,960 (9.01%)
- Clase 1 (Estable): 594,750 (83.81%)
- Clase 2 (Escalada): 50,968 (7.18%)

**Justificación de Umbrales:**

Los umbrales fueron establecidos basándose en:
1. Análisis cuantil de distribución de crímenes
2. Capacidad operacional de recursos policiales
3. Literatura criminológica sobre definición de hotspots
4. Balance entre clases para entrenamiento efectivo

### **4. Algoritmos de Clasificación Implementados**

Siguiendo el **Capítulo 3: Classification** del libro "Hands-On Machine Learning", se implementaron **7 algoritmos de clasificación supervisada**:

#### **4.1. SGD Classifier (Stochastic Gradient Descent)**
- **Familia:** Clasificador lineal
- **Características:** Entrenamiento eficiente con grandes datasets, manejo online de datos
- **Aplicabilidad:** Excelente para 709,678 registros de HURTO
- **Parámetros:** max_iter=1000, random_state=42

#### **4.2. Logistic Regression**
- **Familia:** Clasificador lineal probabilístico
- **Características:** Salida probabilística, interpretable
- **Ventaja:** Probabilidades de pertenencia a clase útiles para ranking de zonas
- **Parámetros:** max_iter=1000, solver='lbfgs'

#### **4.3. Random Forest Classifier**
- **Familia:** Ensemble (Bagging)
- **Características:** Robusto a overfitting, maneja no-linealidad
- **Ventaja:** Importancia de features, funciona bien con datos desbalanceados
- **Parámetros:** n_estimators=100, n_jobs=-1

#### **4.4. Gradient Boosting Classifier**
- **Familia:** Ensemble (Boosting)
- **Características:** Construcción secuencial, alta precisión
- **Ventaja:** Típicamente el mejor rendimiento en competencias ML
- **Parámetros:** n_estimators=100, learning_rate=0.1

#### **4.5. K-Nearest Neighbors (KNN) Classifier**
- **Familia:** Instance-based learning
- **Características:** No entrena modelo explícito, decisiones por vecindad
- **Ventaja:** Captura patrones espaciales naturalmente
- **Parámetros:** n_neighbors=10, metric='euclidean'

#### **4.6. Decision Tree Classifier**
- **Familia:** Árbol de decisión individual
- **Características:** Altamente interpretable, reglas explícitas
- **Ventaja:** Fácil comunicación de lógica a stakeholders
- **Parámetros:** max_depth=20, criterion='gini'

#### **4.7. AdaBoost Classifier**
- **Familia:** Ensemble (Boosting adaptativo)
- **Características:** Enfoque iterativo en instancias difíciles
- **Ventaja:** Mejora clasificadores débiles
- **Parámetros:** n_estimators=100, learning_rate=1.0

#### **4.8. Cobertura Experimental Completa**

**Total de modelos:** 7 algoritmos × 3 tipos de clasificación × 2 delitos = **42 modelos**

| Tipo Clasificación | HURTO | EXTORSIÓN | Total por Tipo |
|-------------------|-------|-----------|----------------|
| Nivel de Riesgo (4 clases) | 7 | 7 | 14 |
| Hotspot Crítico (binaria) | 7 | 7 | 14 |
| Tendencia (3 clases) | 7 | 7 | 14 |
| **Total por Delito** | **21** | **21** | **42** |

Esta cobertura exhaustiva permite:
1. Comparación rigurosa de algoritmos
2. Identificación del mejor modelo por tipo de problema
3. Análisis de sensibilidad a volumen de datos (HURTO vs EXTORSIÓN)
4. Cumplimiento amplio del requisito PC3 (20+ modelos)

---

### **5. Conclusión de Metodología**

Se ha establecido un marco metodológico robusto para clasificación de hotspots criminales que:

✓ **Cumple requisitos académicos:** Capítulo 3 Classification, 42 modelos > 20 requeridos

✓ **Genera valor operacional:** 3 sistemas de decisión con preguntas concretas

✓ **Utiliza métricas apropiadas:** F1-Score, Precision, Recall para datos desbalanceados

✓ **Validación rigurosa:** Split temporal, autocorrelación confirmada, hotspots persistentes

Los resultados de la implementación de estos 42 modelos se detallan en la **Parte 2: Resultados y Análisis Comparativo**.
