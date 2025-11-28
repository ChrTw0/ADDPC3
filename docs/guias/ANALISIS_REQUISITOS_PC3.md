# Análisis Crítico: ¿Qué necesitas REALMENTE para PC3?

## 📋 Requisitos del PC3

### Grupo 2: Capítulo 3 - Classification
- **Tema:** Delincuencia
- **Requisito:** Mínimo **20 modelos** implementados
- **Capítulo:** Classification (pero tú cambiaste a regresión)

---

## 🤔 PROBLEMA: Clasificación vs Regresión

### Lo que el PC3 pide:
**CLASIFICACIÓN** (Cap. 3)
- Predecir categorías/clases
- Ejemplos: Clasificar tipo de delito, riesgo alto/medio/bajo

### Lo que tú estás haciendo:
**REGRESIÓN**
- Predecir cantidad de crímenes (número continuo)
- Modelos: Random Forest Regressor, KNN Regressor, SVR, LSTM

---

## ✅ SOLUCIÓN: Hacer AMBOS enfoques

### Enfoque 1: REGRESIÓN (lo que ya tienes)
**Problema:** Predecir **cantidad** de crímenes por semana en cada celda

**Modelos para HURTO y EXTORSIÓN:**
1. ✅ Random Forest Regressor
2. ✅ KNN Regressor
3. ✅ SVR
4. ✅ LSTM Simple
5. ✅ LSTM Optimizado
6. ⚠️ **FALTAN:** Variaciones y otros regresores

### Enfoque 2: CLASIFICACIÓN (cumplir PC3)
**Problema:** Predecir **nivel de riesgo** (Bajo/Medio/Alto/Muy Alto)

**Modelos de Clasificación necesarios:**
1. ❌ Logistic Regression
2. ❌ SGD Classifier
3. ❌ Random Forest Classifier
4. ❌ KNN Classifier
5. ❌ SVM Classifier
6. ❌ Decision Tree Classifier
7. ❌ Gradient Boosting Classifier
8. ❌ Extra Trees Classifier
9. ❌ Naive Bayes
10. ❌ Neural Network Classifier

---

## 🎯 RECOMENDACIÓN: Enfoque Híbrido

### Opción A: Solo Regresión (más simple)
**Modelos necesarios para llegar a 20:**

**Modelos Base (x2 delitos = 10 modelos):**
1-2. Random Forest Regressor (HURTO + EXTORSIÓN)
3-4. KNN Regressor (k=5, k=10)
5-6. SVR (kernel rbf, kernel poly)
7-8. Linear Regression + Ridge
9-10. Lasso + ElasticNet

**Modelos Deep Learning (x2 delitos = 10 modelos):**
11-12. LSTM Simple
13-14. LSTM Optimizado (2 capas)
15-16. LSTM con features espaciales
17-18. GRU (variante de LSTM)
19-20. Dense Network (MLP)

**Total: 20 modelos ✅**

### Opción B: Híbrido Regresión + Clasificación (más completo)

**Regresión (10 modelos):**
- HURTO: RF, KNN, SVR, LSTM, Dense (5)
- EXTORSIÓN: RF, KNN, SVR, LSTM, Dense (5)

**Clasificación (10 modelos):**
- Convertir `crime_count` a categorías: Bajo (0-2), Medio (3-5), Alto (6-10), Muy Alto (>10)
- HURTO: Logistic, RF, KNN, SVM, GradBoost (5)
- EXTORSIÓN: Logistic, RF, KNN, SVM, GradBoost (5)

**Total: 20 modelos ✅**

---

## 📊 Análisis de Datos: ¿Qué cargar?

### Datos ACTUALES que estás cargando:

```sql
SELECT
    id, lat_hecho, long_hecho, fecha_hora_hecho,
    modalidad_hecho, distrito_hecho
FROM denuncias
WHERE departamento_hecho = 'LIMA'
    AND modalidad_hecho = 'HURTO'  -- o EXTORSIÓN
```

### ¿Son necesarias TODAS estas columnas?

| Columna | ¿Necesaria? | Uso |
|---------|-------------|-----|
| `id` | ❌ NO | No se usa en modelo |
| `lat_hecho` | ✅ SÍ | Para crear grid espacial |
| `long_hecho` | ✅ SÍ | Para crear grid espacial |
| `fecha_hora_hecho` | ✅ SÍ | Para lags temporales |
| `modalidad_hecho` | ❌ NO | Ya filtrado en WHERE |
| `distrito_hecho` | ⚠️ OPCIONAL | Podría ser feature adicional |

### Columnas REALMENTE ÚTILES que NO estás usando:

```sql
SELECT
    lat_hecho,
    long_hecho,
    fecha_hora_hecho,
    turno_hecho,        -- ✅ ÚTIL: Mañana/Tarde/Noche
    periodo_dia,        -- ✅ ÚTIL: Madrugada/Día/Tarde/Noche
    distrito_hecho      -- ✅ ÚTIL: Feature categórica
FROM denuncias
WHERE departamento_hecho = 'LIMA'
    AND modalidad_hecho IN ('HURTO', 'EXTORSION')
```

### Query OPTIMIZADA:

```sql
-- Solo lo ESENCIAL
SELECT
    lat_hecho,
    long_hecho,
    YEAR(fecha_hora_hecho) as año,
    WEEK(fecha_hora_hecho) as semana,
    HOUR(fecha_hora_hecho) as hora,
    DAYOFWEEK(fecha_hora_hecho) as dia_semana
FROM denuncias
WHERE departamento_hecho = 'LIMA'
    AND modalidad_hecho = 'HURTO'
    AND fecha_hora_hecho >= '2020-01-01'  -- Solo últimos 5 años
    AND lat_hecho IS NOT NULL
    AND long_hecho IS NOT NULL
```

**Ventajas:**
- 🚀 Más rápido (menos datos transferidos)
- 💾 Menos memoria
- ⚡ Procesamiento más eficiente

---

## 🎯 MI RECOMENDACIÓN FINAL

### Para cumplir PC3 eficientemente:

#### 1. **Modelos de Regresión** (12 modelos)

**HURTO (6 modelos):**
1. Random Forest Regressor
2. KNN Regressor (k=10)
3. SVR (rbf kernel)
4. LSTM Optimizado
5. Dense Network
6. Gradient Boosting Regressor

**EXTORSIÓN (6 modelos):**
7-12. Mismos modelos que HURTO

#### 2. **Modelos de Clasificación** (8 modelos)

Target: Nivel de Riesgo (Bajo/Medio/Alto/Muy Alto)

**HURTO (4 modelos):**
13. Logistic Regression
14. Random Forest Classifier
15. KNN Classifier
16. SVM Classifier

**EXTORSIÓN (4 modelos):**
17-20. Mismos modelos que HURTO

**Total: 20 modelos ✅**

---

## ⚡ SIGUIENTE PASO

¿Qué prefieres?

### Opción 1: Mantener solo Regresión
- Agrego: Gradient Boosting, Extra Trees, MLP, GRU
- Más simple, menos cambios

### Opción 2: Agregar Clasificación
- Convierto problema a categorías de riesgo
- Agrego clasificadores clásicos
- MÁS COMPLETO, cumple mejor con PC3 (Cap 3: Classification)

### Opción 3: Optimizar datos + mantener actual
- Simplifico query SQL
- Solo cargo lo necesario
- Mejoro velocidad

**¿Cuál eliges?** 🎯
