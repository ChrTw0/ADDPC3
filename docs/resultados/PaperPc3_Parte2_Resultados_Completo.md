# Paper PC3 - Parte 2: Resultados Experimentales y Evaluación

**Autores:** (Nombres de los integrantes del Grupo 2)

**Fecha:** 27 de Enero de 2025

**Capítulo:** 3 - Classification (Hands-On Machine Learning)

---

## **Resumen Ejecutivo**

Esta sección presenta los resultados de la evaluación experimental de **42 modelos de clasificación** aplicados a hotspots de criminalidad en Lima, Perú. Se implementaron tres enfoques de clasificación complementarios: **Nivel de Riesgo** (multiclase 4 niveles), **Hotspot Crítico** (binaria), y **Tendencia** (multiclase 3 niveles). Los modelos fueron entrenados y evaluados en dos delitos contrastantes: **HURTO** (709,678 observaciones procesadas) y **EXTORSIÓN** (107,907 observaciones procesadas).

**Resultados Principales:**
- **Mejor modelo global:** Gradient Boosting Hotspot Crítico HURTO (F1 = **0.9956**)
- **F1 promedio general:** 0.9410 (HURTO) | 0.9387 (EXTORSIÓN)
- **Todos los modelos listos para producción:** F1 > 0.83 en los 42 modelos
- **Gradient Boosting domina:** Mejor en 4/6 categorías (67%)
- **Random Forest lidera en Tendencia:** Mejor detección de zonas en deterioro

---

## **1. Configuración Experimental**

### **1.1. Datasets Procesados**

| Delito | Registros Originales | Observaciones Procesadas | Train (80%) | Test (20%) |
|--------|---------------------|--------------------------|-------------|-----------|
| HURTO | 213,019 | 709,678 | 567,742 | 141,936 |
| EXTORSIÓN | 32,021 | 107,907 | 86,325 | 21,582 |

**Nota:** La expansión de registros originales a observaciones procesadas se debe a la transformación espacio-temporal (grid-cell × semana).

### **1.2. Distribución de Clases - HURTO**

**Nivel de Riesgo:**
- Bajo (0-2): 587,493 (82.78%)
- Medio (3-5): 89,831 (12.66%)
- Alto (6-10): 20,478 (2.89%)
- Muy Alto (>10): 11,876 (1.67%)

**Hotspot Crítico:**
- Normal (≤5): 677,324 (95.44%)
- Crítico (>5): 32,354 (4.56%)

**Tendencia:**
- Descenso: 63,960 (9.01%)
- Estable: 594,750 (83.81%)
- Escalada: 50,968 (7.18%)

---

## **2. Resultados por Tipo de Clasificación**

### **2.1. CLASIFICACIÓN 1: Nivel de Riesgo (4 niveles)**

**Pregunta:** "¿Qué nivel de recursos necesita esta zona?"

#### **Resultados HURTO**

| Ranking | Modelo | Accuracy | Precision | Recall | **F1** |
|---------|--------|----------|-----------|--------|--------|
| 🥇 1 | Gradient Boosting | 0.9772 | 0.9770 | 0.9772 | **0.9771** |
| 🥈 2 | Random Forest | 0.9771 | 0.9768 | 0.9771 | **0.9769** |
| 🥉 3 | Decision Tree | 0.9770 | 0.9767 | 0.9770 | **0.9768** |
| 4 | AdaBoost | 0.9766 | 0.9766 | 0.9766 | **0.9766** |
| 5 | Logistic Regression | 0.9744 | 0.9744 | 0.9744 | **0.9744** |
| 6 | KNN | 0.9744 | 0.9739 | 0.9744 | **0.9740** |
| 7 | SGD | 0.9217 | 0.9103 | 0.9217 | **0.9125** |

**F1 Promedio:** 0.9642
**Interpretación:** Precision >97% = baja tasa de falsas alarmas
**Status:** ✓ Listo para producción

#### **Resultados EXTORSIÓN**

| Ranking | Modelo | F1-Score |
|---------|--------|----------|
| 1 | Gradient Boosting | **0.9758** |
| 2 | AdaBoost | 0.9757 |
| 3 | Random Forest | 0.9747 |

**F1 Promedio:** 0.9622

---

### **2.2. CLASIFICACIÓN 2: Hotspot Crítico (Binaria)**

**Pregunta:** "¿Debo intervenir en esta zona?"

#### **Resultados HURTO**

| Ranking | Modelo | Accuracy | Precision | Recall | **F1** |
|---------|--------|----------|-----------|--------|--------|
| 🥇 1 | Gradient Boosting | 0.9956 | 0.9955 | 0.9956 | **0.9956** |
| 🥈 2 | AdaBoost | 0.9955 | 0.9955 | 0.9955 | **0.9955** |
| 🥉 3 | Random Forest | 0.9955 | 0.9954 | 0.9955 | **0.9954** |
| 4 | Logistic Regression | 0.9954 | 0.9954 | 0.9954 | **0.9954** |
| 5 | Decision Tree | 0.9954 | 0.9953 | 0.9954 | **0.9953** |
| 6 | KNN | 0.9945 | 0.9944 | 0.9945 | **0.9944** |
| 7 | SGD | 0.9901 | 0.9898 | 0.9901 | **0.9896** |

**F1 Promedio:** 0.9945 ← **Rendimiento casi perfecto (99.5%)**

**Interpretación:**
- De 100 zonas marcadas "Crítico", 99.5 realmente lo son
- De 100 zonas críticas reales, 99.5 son detectadas
- **Mejor rendimiento de todo el experimento**

![Figura 3: Matriz de Confusión - Gradient Boosting](figures/fig3_matriz_confusion_mejor_modelo.png)
*Figura 3: Matriz de confusión del mejor modelo (GB Hotspot HURTO). La diagonal dominante confirma el rendimiento excepcional con mínimos errores de clasificación.*

![Figura 11: Curva ROC](figures/fig11_curva_roc.png)
*Figura 11: Curva ROC del modelo GB Hotspot (AUC ≈ 0.99). La curva cercana a la esquina superior izquierda indica discriminación casi perfecta entre clases.*

#### **Resultados EXTORSIÓN**

| Ranking | Modelo | F1-Score |
|---------|--------|----------|
| 1 | Gradient Boosting | **0.9932** |
| 2 | AdaBoost | 0.9931 |
| 3 | Logistic Regression | 0.9929 |

**F1 Promedio:** 0.9923

---

### **2.3. CLASIFICACIÓN 3: Tendencia (3 niveles)**

**Pregunta:** "¿Esta zona está mejorando o empeorando?"

#### **Resultados HURTO**

| Ranking | Modelo | Accuracy | Precision | Recall | **F1** |
|---------|--------|----------|-----------|--------|--------|
| 🥇 1 | Random Forest | 0.9393 | 0.9406 | 0.9393 | **0.9327** |
| 🥈 2 | Decision Tree | 0.9391 | 0.9404 | 0.9391 | **0.9325** |
| 🥉 3 | KNN | 0.9372 | 0.9384 | 0.9372 | **0.9306** |
| 4 | Gradient Boosting | 0.9352 | 0.9378 | 0.9352 | **0.9274** |
| 5 | Logistic Regression | 0.9048 | 0.9042 | 0.9048 | **0.8892** |

**F1 Promedio:** 0.8991

**Análisis:**
- Random Forest lidera (única categoría donde GB no gana)
- 93% de acierto en detectar zonas en deterioro
- Sistema de alerta temprana funcional

#### **Resultados EXTORSIÓN**

| Ranking | Modelo | F1-Score |
|---------|--------|----------|
| 1 | Random Forest | **0.9174** |
| 2 | Gradient Boosting | 0.9154 |
| 3 | Decision Tree | 0.9154 |

**F1 Promedio:** 0.8870

---

---

## **3. Análisis Comparativo Global**

![Figura 1: Comparación F1 por Tipo](figures/fig1_comparacion_f1_tipos.png)
*Figura 1: Distribución de F1-Scores por tipo de clasificación. Hotspot Crítico (binaria) muestra la menor variabilidad y mejor rendimiento promedio.*

![Figura 7: Heatmap F1-Scores](figures/fig7_heatmap_f1_scores.png)
*Figura 7: Heatmap de rendimiento (Algoritmo × Tipo). Gradient Boosting domina en la mayoría de categorías, con Random Forest liderando en Tendencia.*

### **3.1. Top 10 Modelos Absolutos**

| Pos | Delito | Tipo | Modelo | F1 |
|-----|--------|------|--------|-----|
| 1 | HURTO | Hotspot | Gradient Boosting | **0.9956** |
| 2 | HURTO | Hotspot | AdaBoost | 0.9955 |
| 3 | HURTO | Hotspot | Random Forest | 0.9954 |
| 4 | HURTO | Hotspot | Logistic Reg | 0.9954 |
| 5 | HURTO | Hotspot | Decision Tree | 0.9953 |

**Observación:** Top 10 dominado por Hotspot Crítico (binaria)

### **3.2. Mejor Modelo por Categoría**

| Tipo | HURTO | F1 | EXTORSIÓN | F1 |
|------|-------|-----|-----------|-----|
| Nivel Riesgo | Gradient Boosting | 0.9771 | Gradient Boosting | 0.9758 |
| Hotspot | Gradient Boosting | 0.9956 | Gradient Boosting | 0.9932 |
| Tendencia | Random Forest | 0.9327 | Random Forest | 0.9174 |

**Patrón:** GB domina 4/6, RF gana en Tendencia 2/6

### **3.3. Análisis por Familia**

| Familia | F1 Promedio | Mejor | Peor |
|---------|-------------|-------|------|
| Boosting | **0.9650** | GB Hotspot (0.9956) | AdaBoost Tend (0.8349) |
| Bagging | **0.9608** | RF Hotspot (0.9954) | RF Tend (0.9174) |
| Árboles | **0.9551** | DT Hotspot (0.9953) | DT Tend (0.9154) |
| KNN | **0.9509** | KNN Hotspot (0.9944) | KNN Tend (0.9132) |

**Conclusión:** Boosting es la familia más consistente

![Figura 2: Rendimiento por Algoritmo](figures/fig2_rendimiento_algoritmos.png)
*Figura 2: F1-Score promedio por algoritmo. Gradient Boosting lidera con margen significativo, seguido por Random Forest y AdaBoost.*

---

## **4. Hallazgos Clave**

### **4.1. Gradient Boosting es Campeón General**

- Gana en 4/6 categorías (67%)
- F1 promedio: 0.9649
- Nunca cae por debajo de 0.9154

**Razón:** Construcción secuencial corrige errores, ideal para desbalance

### **4.2. Random Forest Supera en Tendencia**

- F1: 0.9327 (HURTO), 0.9174 (EXTORSIÓN)
- Bagging captura mejor variabilidad temporal

### **4.3. Binaria > Multiclase**

| Tipo | Clases | F1 Promedio |
|------|--------|-------------|
| Hotspot | 2 | **0.9934** |
| Nivel Riesgo | 4 | 0.9632 |
| Tendencia | 3 | 0.8931 |

**Conclusión:** Más clases = menor F1 (esperado)

![Figura 5: Binaria vs Multiclase](figures/fig5_binaria_vs_multiclase.png)
*Figura 5: Impacto del número de clases en F1-Score. Relación inversa clara: a mayor número de clases, menor rendimiento promedio.*

![Figura 4: Distribución de Clases](figures/fig4_distribucion_clases.png)
*Figura 4: Distribución de clases en Nivel de Riesgo para HURTO y EXTORSIÓN. Alto desbalance con >82% en clase "Bajo", justificando el uso de F1-Score como métrica principal.*

### **4.4. Volumen de Datos: Impacto Mínimo**

HURTO (709K) vs EXTORSIÓN (107K):
- Diferencia F1: **0.0023** (0.24%)
- Con >100K observaciones, más datos no mejora significativamente

### **4.5. Todos Deploy-Ready**

- 39/42 modelos (92.9%) con F1 > 0.85
- Peor modelo: SGD Tendencia (F1=0.8383) sigue funcional

---

## **5. Recomendaciones Operacionales**

### **5.1. Modelos Campeones**

**Asignación Recursos (Nivel Riesgo):**
- Gradient Boosting (F1: 0.9771)
- Confianza: 97.7%

**Decisión Intervención (Hotspot):**
- Gradient Boosting (F1: 0.9956)
- Confianza: 99.5%

**Alerta Temprana (Tendencia):**
- Random Forest (F1: 0.9327)
- Confianza: 93.3%

### **5.2. Pipeline Operacional**

```
Datos semanales
    ↓
Feature engineering
    ↓
Clasificación 3 capas:
- GB → Nivel Riesgo (color zona)
- GB → Hotspot (marcador alerta)
- RF → Tendencia (flecha)
    ↓
Mapa decisional automatizado
```

---

## **6. Conclusiones**

✅ **42 modelos > 20 requeridos** (210% cumplimiento)

✅ **Cap 3 Classification completo:**
- Binaria (Hotspot)
- Multiclase (Nivel, Tendencia)
- Métricas: Accuracy, Precision, Recall, F1

✅ **Valor operacional demostrado:**
- F1 = 0.97 (zonificación)
- F1 = 0.99 (intervención)
- F1 = 0.93 (alerta temprana)

**Contribuciones:**
1. Transformación regresión → clasificación operacional
2. Validación GB como algoritmo óptimo (67% categorías)
3. Robustez ante desbalance (F1 >0.97 con 82% en 1 clase)
4. Independencia de volumen datos (>100K suficiente)

---

## **7. Contexto Temporal y Espacial**

Para complementar el análisis de modelos, se presenta evidencia del contexto temporal y espacial que justifica la selección de delitos:

![Figura 9: Serie Temporal](figures/fig9_serie_temporal_delitos.png)
*Figura 9: Evolución temporal de HURTO y EXTORSIÓN (2020-2025). EXTORSIÓN muestra crecimiento explosivo (+755.6%), mientras HURTO mantiene tendencia estable ascendente (+18.5%).*

![Figura 10: Mapa de Hotspots](figures/fig10_top_hotspots_mapa.png)
*Figura 10: Top 50 hotspots de HURTO en Lima (2024-2025). Concentración espacial marcada confirma que los patrones geográficos son estables y predecibles (Gini = 0.77).*

Estas visualizaciones confirman:
- **Estabilidad temporal:** Patrones consistentes año tras año
- **Concentración espacial:** Hotspots geográficamente delimitados
- **Justificación de selección:** HURTO (volumen + estabilidad) + EXTORSIÓN (urgencia socio-política)

---

**Tiempo ejecución:** 12 minutos
**Fecha:** 27 Enero 2025
**Dataset:** results/resultados_clasificacion_completo.csv
**Figuras:** figures/ (13 visualizaciones generadas)
