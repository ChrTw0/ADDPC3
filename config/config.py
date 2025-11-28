"""
Configuración Central del Proyecto - SOLO CLASIFICACIÓN
========================================================
Proyecto enfocado 100% en clasificación (Capítulo 3 del libro).

Total de modelos: 7 algoritmos × 3 tipos de clasificación × 2 delitos = 42 modelos
"""

# ============================================================================
# DELITOS A PROCESAR
# ============================================================================

DELITOS = {
    'hurto': 'HURTO',
    'extorsion': 'EXTORSION'
}

# ============================================================================
# TIPOS DE CLASIFICACIÓN (3 PROBLEMAS CON VALOR OPERACIONAL)
# ============================================================================

TIPOS_CLASIFICACION = {
    'nivel_riesgo': {
        'nombre': 'Nivel de Riesgo',
        'descripcion': 'Clasificación multiclase (4 niveles)',
        'clases': ['Bajo', 'Medio', 'Alto', 'Muy Alto'],
        'pregunta': '¿Qué nivel de recursos necesita esta zona?'
    },
    'hotspot_critico': {
        'nombre': 'Hotspot Crítico',
        'descripcion': 'Clasificación binaria',
        'clases': ['Normal', 'Crítico'],
        'pregunta': '¿Debo intervenir en esta zona esta semana?'
    },
    'tendencia': {
        'nombre': 'Tendencia de Riesgo',
        'descripcion': 'Clasificación multiclase (3 niveles)',
        'clases': ['Descenso', 'Estable', 'Escalada'],
        'pregunta': '¿Esta zona está mejorando o empeorando?'
    }
}

# ============================================================================
# MODELOS DE CLASIFICACIÓN (7 ALGORITMOS)
# ============================================================================

MODELOS_CLASIFICACION = [
    'sgd',
    'logistic',
    'random_forest',
    'gradient_boosting',
    'knn',
    'decision_tree',
    'adaboost'
]

# ============================================================================
# HIPERPARÁMETROS PARA CLASIFICACIÓN
# ============================================================================

HIPERPARAMETROS_CLASIFICACION = {
    'sgd': {
        'loss': ['hinge', 'log_loss', 'modified_huber'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000]
    },
    'logistic': {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'decision_tree': {
        'max_depth': [5, 10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    },
    'adaboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }
}

# ============================================================================
# CONFIGURACIÓN DE FEATURES
# ============================================================================

GRID_SIZE = 0.005  # Tamaño de grid espacial (~555m × 555m)
NUM_LAGS = 4       # Número de lags temporales
TRAIN_TEST_SPLIT = 0.8  # Proporción de datos para entrenamiento

FEATURE_COLS = [
    'crime_count_lag_1',
    'crime_count_lag_2',
    'crime_count_lag_3',
    'crime_count_lag_4',
    'mes',
    'dia_semana'
]

# ============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN
# ============================================================================

RANDOMIZED_SEARCH_ITERATIONS = 20
CROSS_VALIDATION_FOLDS = 3
RANDOM_STATE = 42

# ============================================================================
# RUTAS DE SALIDA
# ============================================================================

MODELS_OUTPUT_DIR = 'models/best_models'
RESULTS_OUTPUT_DIR = 'results'
VISUALIZATIONS_OUTPUT_DIR = 'visualizations'
