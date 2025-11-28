"""
Modelos de Clasificación - Capítulo 3
======================================
Implementación de 7 algoritmos de clasificación aplicados a 3 problemas.

Total: 7 algoritmos × 3 clasificaciones × 2 delitos = 42 modelos
"""

import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from config.config import (
    HIPERPARAMETROS_CLASIFICACION, RANDOMIZED_SEARCH_ITERATIONS,
    CROSS_VALIDATION_FOLDS, RANDOM_STATE
)
from utils.target_engineering import obtener_descripcion_target


def obtener_modelo_clasificacion(nombre_modelo):
    """
    Retorna una instancia del modelo de clasificación solicitado.
    
    Args:
        nombre_modelo: Nombre del modelo
        
    Returns:
        Instancia del modelo
    """
    modelos = {
        'sgd': SGDClassifier(
            random_state=RANDOM_STATE, 
            max_iter=1000
        ),
        'logistic': LogisticRegression(
            random_state=RANDOM_STATE, 
            max_iter=1000
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, 
            random_state=RANDOM_STATE
        ),
        'knn': KNeighborsClassifier(n_neighbors=10),
        'decision_tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE, 
            max_depth=20
        ),
        'adaboost': AdaBoostClassifier(
            n_estimators=100, 
            random_state=RANDOM_STATE
        )
    }
    
    return modelos.get(nombre_modelo)


def entrenar_modelo_clasificacion(nombre_modelo, tipo_clasificacion, X_train, y_train, X_test, y_test, optimizar=False):
    """
    Entrena y evalúa un modelo de clasificación.
    
    Args:
        nombre_modelo: Nombre del modelo
        tipo_clasificacion: 'nivel_riesgo', 'hotspot_critico', 'tendencia'
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        optimizar: Si True, usa RandomizedSearchCV
        
    Returns:
        Dict con resultados del modelo
    """
    desc = obtener_descripcion_target(tipo_clasificacion)
    
    print(f"      Entrenando: {nombre_modelo} [{desc['nombre']}]{'  [OPTIMIZANDO...]' if optimizar else ''}...")
    
    model = obtener_modelo_clasificacion(nombre_modelo)
    if model is None:
        raise ValueError(f"Modelo {nombre_modelo} no reconocido")
    
    mejores_params = None
    
    # Optimización de hiperparámetros
    if optimizar and nombre_modelo in HIPERPARAMETROS_CLASIFICACION:
        param_grid = HIPERPARAMETROS_CLASIFICACION[nombre_modelo]
        
        n_iter = min(RANDOMIZED_SEARCH_ITERATIONS, len(list(param_grid.values())[0]) * 3)
        
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,
            cv=CROSS_VALIDATION_FOLDS,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        mejores_params = search.best_params_
        
        print(f"         [OPTIMIZADO] Mejores params: {mejores_params}")
    else:
        model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"         Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    
    # Interpretación operacional
    interpretacion = interpretar_resultados_operacionales(
        tipo_clasificacion, acc, prec, rec, f1
    )
    
    return {
        'modelo': nombre_modelo,
        'tipo_clasificacion': tipo_clasificacion,
        'nombre_clasificacion': desc['nombre'],
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'mejores_params': mejores_params,
        'interpretacion': interpretacion,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'model_obj': model
    }


def interpretar_resultados_operacionales(tipo_clasificacion, acc, prec, rec, f1):
    """
    Genera interpretación operacional de las métricas.
    
    Args:
        tipo_clasificacion: Tipo de clasificación
        acc, prec, rec, f1: Métricas
        
    Returns:
        Lista de interpretaciones operacionales
    """
    interpretaciones = []
    
    if tipo_clasificacion == 'nivel_riesgo':
        if f1 > 0.85:
            interpretaciones.append("✓ Excelente zonificación de recursos")
        if prec > 0.85:
            interpretaciones.append("✓ Baja tasa de falsas alarmas")
        if rec < 0.75:
            interpretaciones.append("⚠ Puede perder zonas de riesgo real")
    
    elif tipo_clasificacion == 'hotspot_critico':
        if rec > 0.85:
            interpretaciones.append("✓ Detecta mayoría de hotspots críticos")
        else:
            interpretaciones.append("⚠ Puede perder hotspots que requieren intervención")
        
        if prec > 0.80:
            interpretaciones.append("✓ Evita despliegues innecesarios")
        else:
            interpretaciones.append("⚠ Muchos falsos positivos = recursos desperdiciados")
    
    elif tipo_clasificacion == 'tendencia':
        if acc > 0.80:
            interpretaciones.append("✓ Identifica bien tendencias de deterioro")
        if f1 > 0.75:
            interpretaciones.append("✓ Buen sistema de alerta temprana")
    
    # Interpretaciones generales
    if f1 < 0.60:
        interpretaciones.append("⚠ Modelo poco confiable para decisiones operacionales")
    
    return interpretaciones
