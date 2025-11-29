# Arquitectura del Proyecto ADDPC3

## Visión General

Este proyecto implementa una **arquitectura modular y escalable** para la clasificación de hotspots criminales, siguiendo las mejores prácticas de ingeniería de software y machine learning.

---

## Principios de Diseño

### 1. **Separación de Responsabilidades**
Cada módulo tiene una función específica y bien definida:
- **Config:** Configuración centralizada
- **Models:** Lógica de ML
- **Utils:** Utilidades compartidas
- **Scripts:** Orquestación y ejecución

### 2. **Modularidad**
- Componentes independientes y reutilizables
- Fácil testing unitario
- Bajo acoplamiento, alta cohesión

### 3. **Escalabilidad**
- Fácil agregar nuevos modelos
- Extensible a nuevos delitos
- Soporte para múltiples tipos de clasificación

---

## Estructura Detallada

```
ADDPC3/
│
├── 📁 config/                    # Configuración Central
│   ├── __init__.py
│   └── config.py                # ⚙️ Hiperparámetros, constantes, delitos
│
├── 📁 models/                    # Modelos de Machine Learning
│   ├── __init__.py
│   ├── common.py                # 🔌 Conexión DB, funciones compartidas
│   ├── classification_models.py # 🤖 7 algoritmos de clasificación
│   └── best_models/             # 💾 Modelos entrenados (.joblib)
│
├── 📁 utils/                     # Utilidades Compartidas
│   ├── __init__.py
│   ├── data_preparation.py      # 📊 Extracción y preparación
│   ├── feature_engineering.py   # 🔧 Ingeniería de características
│   ├── target_engineering.py    # 🎯 Creación de targets
│   └── model_evaluation.py      # 📈 Evaluación y persistencia
│
└── 📁 scripts/                   # Scripts Ejecutables
    ├── ejecutar_todos_modelos.py # ⭐ Orquestador principal
    ├── analisis/                 # 🔬 Análisis exploratorios
    └── visualizacion/            # 📊 Gráficos y mapas
```

---

## Flujo de Ejecución

### Pipeline Principal

```
┌─────────────────────────────────────┐
│  1. Inicio (ejecutar_todos_modelos) │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  2. Cargar Configuración (config.py) │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  3. Preparar Datos (data_preparation)    │
│     • Extraer desde MySQL                │
│     • Crear grid espacial (0.005°)       │
│     • Features temporales (lags)         │
│     • Split train/test (80/20)           │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  4. Crear Targets (target_engineering)   │
│     • Nivel de Riesgo (4 clases)         │
│     • Hotspot Crítico (binario)          │
│     • Tendencia (3 clases)               │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  5. Entrenar Modelos (classification)    │
│     • 7 algoritmos × 3 tipos             │
│     • Optimización opcional (GridSearch) │
│     • Evaluación (F1, Precision, Recall) │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  6. Guardar Resultados (model_evaluation)│
│     • Mejores modelos (.joblib)          │
│     • Métricas (CSV)                     │
│     • Recomendaciones operacionales      │
└──────────────────────────────────────────┘
```

---

## Módulos Clave

### 📦 **config/config.py**

**Responsabilidad:** Configuración centralizada del proyecto

```python
# Delitos a procesar
DELITOS = {
    'hurto': 'HURTO',
    'extorsion': 'EXTORSION'
}

# Tipos de clasificación
TIPOS_CLASIFICACION = {
    'nivel_riesgo': {...},
    'hotspot_critico': {...},
    'tendencia': {...}
}

# Modelos a entrenar
MODELOS_CLASIFICACION = [
    'sgd', 'logistic', 'random_forest',
    'gradient_boosting', 'knn', 
    'decision_tree', 'adaboost'
]

# Hiperparámetros para optimización
HIPERPARAMETROS_CLASIFICACION = {...}
```

**Ventajas:**
- ✅ Punto único de configuración
- ✅ Fácil modificar parámetros
- ✅ Evita "magic numbers"

---

### 🔌 **models/common.py**

**Responsabilidad:** Conexión a base de datos y funciones compartidas

```python
def get_db_connection():
    """Establece conexión con MySQL usando credenciales .env"""
    load_dotenv()
    engine = create_engine(...)
    return engine
```

**Ventajas:**
- ✅ Reutilización de conexión
- ✅ Manejo seguro de credenciales
- ✅ Abstracción de DB

---

### 📊 **utils/data_preparation.py**

**Responsabilidad:** Preparación y transformación de datos

**Funciones principales:**
1. `extraer_datos_delito()` - Extrae desde MySQL
2. `crear_grid_espacial()` - Discretización geográfica
3. `crear_features_temporales()` - Mes, día de semana
4. `preparar_datos_completo()` - Pipeline completo

**Pipeline de datos:**
```
SQL → DataFrame → Grid Espacial → Features → Lags → Train/Test Split
```

---

### 🎯 **utils/target_engineering.py**

**Responsabilidad:** Creación de variables objetivo (targets)

**Funciones principales:**
1. `crear_target_nivel_riesgo()` - 4 clases (Bajo/Medio/Alto/Muy Alto)
2. `crear_target_hotspot_critico()` - Binario (Normal/Crítico)
3. `crear_target_tendencia()` - 3 clases (Descenso/Estable/Escalada)
4. `crear_todos_los_targets()` - Genera los 3 tipos

**Ejemplo:**
```python
# Nivel de Riesgo
bins = [0, 2, 5, 10, inf]
labels = [0, 1, 2, 3]  # Bajo, Medio, Alto, Muy Alto
```

---

### 🤖 **models/classification_models.py**

**Responsabilidad:** Implementación de algoritmos de clasificación

**Funciones principales:**
1. `obtener_modelo_clasificacion()` - Instancia el modelo
2. `entrenar_modelo_clasificacion()` - Entrena y evalúa
3. Métricas: Accuracy, Precision, Recall, F1-Score

**Soporte de optimización:**
```python
if optimizar:
    search = RandomizedSearchCV(
        modelo,
        param_distributions=hiperparametros,
        n_iter=20,
        cv=3
    )
```

---

### 📈 **utils/model_evaluation.py**

**Responsabilidad:** Evaluación y persistencia de modelos

**Funciones principales:**
1. `guardar_mejores_modelos()` - Serializa con joblib
2. `generar_resumen_resultados()` - DataFrame con métricas
3. `mostrar_mejores_por_delito()` - Top 5 modelos
4. `generar_recomendaciones_operacionales()` - Insights
5. `guardar_resultados_csv()` - Exporta resultados

---

### ⭐ **scripts/ejecutar_todos_modelos.py**

**Responsabilidad:** Orquestador principal del pipeline

**Flujo:**
1. Menu interactivo (CLI)
2. Selección de delito(s)
3. Optimización (Sí/No)
4. Procesar delito completo
5. Generar reportes finales

---

## Diseño de Datos

### Features (X)

```python
FEATURE_COLS = [
    'crime_count_lag_1',  # Crímenes semana anterior
    'crime_count_lag_2',  # 2 semanas atrás
    'crime_count_lag_3',  # 3 semanas atrás
    'crime_count_lag_4',  # 4 semanas atrás
    'mes',                # Estacionalidad mensual
    'dia_semana'          # Patrón semanal
]
```

### Targets (y)

| Target | Tipo | Clases | Uso Operacional |
|--------|------|--------|-----------------|
| `nivel_riesgo` | Multiclase | 4 | Asignación de recursos |
| `hotspot_critico` | Binario | 2 | Decisión de intervención |
| `tendencia` | Multiclase | 3 | Sistema de alerta |

---

## Ventajas de la Arquitectura Actual

### ✅ **Mantenibilidad**
- Cada archivo tiene una responsabilidad única
- Fácil encontrar y modificar código
- Reducción de acoplamiento

### ✅ **Reutilización**
- Funciones pueden usarse en otros proyectos
- Modelos independientes entre sí
- Utilidades compartidas

### ✅ **Testing**
- Cada módulo puede testearse por separado
- Mock de dependencias más sencillo
- Unit tests más focalizados

### ✅ **Escalabilidad**
- Fácil añadir nuevos modelos
- Agregar nuevos tipos de features
- Extender evaluaciones

### ✅ **Colaboración**
- Múltiples personas pueden trabajar sin conflictos
- Cambios localizados en archivos específicos
- Git diffs más legibles

---

## Cómo Extender el Proyecto

### Añadir un Nuevo Modelo de Clasificación

**Paso 1:** Agregar a `config/config.py`
```python
MODELOS_CLASIFICACION.append('nuevo_modelo')

HIPERPARAMETROS_CLASIFICACION['nuevo_modelo'] = {
    'param1': [val1, val2],
    'param2': [val3, val4]
}
```

**Paso 2:** Implementar en `models/classification_models.py`
```python
def obtener_modelo_clasificacion(nombre_modelo):
    modelos = {
        # ...modelos existentes...
        'nuevo_modelo': NuevoClasificador(params)
    }
```

### Añadir un Nuevo Tipo de Clasificación

**Paso 1:** Definir en `config/config.py`
```python
TIPOS_CLASIFICACION['nuevo_tipo'] = {
    'nombre': 'Nuevo Tipo',
    'descripcion': '...',
    'clases': ['Clase1', 'Clase2'],
    'pregunta': '¿Pregunta operacional?'
}
```

**Paso 2:** Crear función en `utils/target_engineering.py`
```python
def crear_target_nuevo_tipo(crime_counts):
    # Lógica de clasificación
    return target_array
```

### Añadir Nuevas Features

**Paso 1:** Implementar en `utils/feature_engineering.py`
```python
def crear_feature_nueva(df):
    # Cálculo de nueva característica
    return df
```

**Paso 2:** Integrar en `utils/data_preparation.py`
```python
def preparar_datos_completo(delito_key):
    # ...código existente...
    df = crear_feature_nueva(df)
    # ...continuar pipeline...
```

---

## Comparación: Antes vs. Después

| Aspecto | Antes (Monolítico) | Después (Modular) |
|---------|-------------------|-------------------|
| **Archivos** | 1 archivo de 659 líneas | 11 archivos organizados |
| **Mantenibilidad** | ❌ Difícil | ✅ Fácil |
| **Testing** | ❌ Complejo | ✅ Simple |
| **Colaboración** | ❌ Conflictos frecuentes | ✅ Trabajo paralelo |
| **Extensibilidad** | ❌ Rígido | ✅ Flexible |
| **Legibilidad** | ❌ Confuso | ✅ Claro |

---

## Mejores Prácticas Implementadas

1. ✅ **DRY (Don't Repeat Yourself)** - Reutilización de código
2. ✅ **Single Responsibility** - Un propósito por módulo
3. ✅ **Configuration Management** - Centralización de parámetros
4. ✅ **Separation of Concerns** - Lógica separada
5. ✅ **Explicit Imports** - Claridad en dependencias
6. ✅ **Error Handling** - Manejo robusto de excepciones
7. ✅ **Documentation** - Docstrings en funciones clave

---

## Tecnologías y Patrones

### Patrones de Diseño
- **Factory Pattern** - Creación de modelos
- **Pipeline Pattern** - Flujo de datos
- **Strategy Pattern** - Selección de algoritmos

### Herramientas
- **SQLAlchemy** - ORM para base de datos
- **Joblib** - Serialización de modelos
- **Scikit-learn** - Framework ML
- **Pandas** - Manipulación de datos

---

## Futuras Mejoras

### 🔮 Roadmap

1. **Testing Suite**
   - Unit tests con pytest
   - Integration tests
   - Coverage > 80%

2. **CI/CD Pipeline**
   - GitHub Actions
   - Automated testing
   - Deployment automation

3. **API REST**
   - FastAPI o Flask
   - Endpoints para predicción
   - Documentación con Swagger

4. **Containerización**
   - Dockerfile
   - Docker Compose
   - Despliegue en la nube

5. **Monitoring**
   - MLflow para tracking
   - Model drift detection
   - Performance monitoring

---

**Última actualización:** 28 de Noviembre de 2025
