"""
Ingeniería de Targets de Clasificación
=======================================
Define múltiples targets de clasificación con valor operacional claro.
"""

import pandas as pd
import numpy as np


# ============================================================================
# CLASIFICACIÓN 1: NIVEL DE RIESGO (MULTICLASE - 4 CATEGORÍAS)
# ============================================================================

def crear_target_nivel_riesgo(crime_counts):
    """
    Clasifica zonas según nivel de riesgo para asignación de recursos.
    
    Categorías:
    - 0 = BAJO (0-2 crímenes)     → Zona Verde    → Patrullaje rutinario
    - 1 = MEDIO (3-5 crímenes)    → Zona Amarilla → Patrullaje reforzado
    - 2 = ALTO (6-10 crímenes)    → Zona Naranja  → Operativo focalizado
    - 3 = MUY ALTO (>10 crímenes) → Zona Roja     → Intervención especial
    
    Pregunta que responde:
    "¿Qué nivel de recursos necesita esta zona?"
    
    Args:
        crime_counts: Array con cantidad de crímenes
        
    Returns:
        Array categórico (0, 1, 2, 3)
    """
    bins = [0, 2, 5, 10, float('inf')]
    labels = [0, 1, 2, 3]  # Bajo, Medio, Alto, Muy Alto
    
    return pd.cut(
        crime_counts, 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    ).astype(int)


# ============================================================================
# CLASIFICACIÓN 2: HOTSPOT CRÍTICO (BINARIA)
# ============================================================================

def crear_target_hotspot_critico(crime_counts, umbral=5):
    """
    Identifica zonas que requieren intervención inmediata.
    
    Categorías:
    - 0 = NO (<= 5 crímenes)  → Vigilancia normal
    - 1 = SÍ (> 5 crímenes)   → Desplegar operativo especial
    
    Pregunta que responde:
    "¿Debo intervenir en esta zona esta semana?"
    
    Args:
        crime_counts: Array con cantidad de crímenes
        umbral: Número de crímenes para considerar hotspot crítico
        
    Returns:
        Array binario (0, 1)
    """
    return (crime_counts > umbral).astype(int)


# ============================================================================
# CLASIFICACIÓN 3: TENDENCIA DE RIESGO (MULTICLASE - 3 CATEGORÍAS)
# ============================================================================

def crear_target_tendencia(df_completo, group_col='grid_cell'):
    """
    Clasifica zonas según si están mejorando, estables o empeorando.
    
    Categorías:
    - 0 = DESCENSO  → Crime_actual < Promedio_histórico * 0.7  → Mejorando
    - 1 = ESTABLE   → Entre 0.7 y 1.3 del promedio             → Estable
    - 2 = ESCALADA  → Crime_actual > Promedio_histórico * 1.3  → Empeorando (ALERTA)
    
    Pregunta que responde:
    "¿Esta zona está mejorando o empeorando?"
    
    Valor operacional:
    - Identificar zonas emergentes (antes seguras, ahora peligrosas)
    - Validar si estrategias están funcionando
    - Alertas tempranas de deterioro
    
    Args:
        df_completo: DataFrame con columnas ['grid_cell', 'año_semana', 'crime_count']
        group_col: Columna para agrupar (default: 'grid_cell')
        
    Returns:
        Array categórico (0, 1, 2)
    """
    # Calcular promedio histórico por celda (excluyendo semana actual)
    df_sorted = df_completo.sort_values([group_col, 'año_semana']).copy()
    
    # Promedio móvil de las últimas 4 semanas (excluyendo la actual)
    df_sorted['crime_promedio_historico'] = (
        df_sorted.groupby(group_col)['crime_count']
        .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
    )
    
    # Calcular ratio
    df_sorted['ratio'] = df_sorted['crime_count'] / (df_sorted['crime_promedio_historico'] + 0.1)
    
    # Clasificar tendencia
    def clasificar_tendencia(ratio):
        if ratio < 0.7:
            return 0  # Descenso
        elif ratio <= 1.3:
            return 1  # Estable
        else:
            return 2  # Escalada
    
    tendencias = df_sorted['ratio'].apply(clasificar_tendencia)
    
    return tendencias.values


# ============================================================================
# FUNCIÓN PRINCIPAL: CREAR TODOS LOS TARGETS
# ============================================================================

def crear_todos_los_targets(df_completo):
    """
    Crea los 3 tipos de targets de clasificación.
    
    Args:
        df_completo: DataFrame con ['grid_cell', 'año_semana', 'crime_count']
        
    Returns:
        Tupla: (y_nivel_riesgo, y_hotspot_critico, y_tendencia)
    """
    crime_counts = df_completo['crime_count'].values
    
    # Clasificación 1: Nivel de Riesgo
    y_nivel_riesgo = crear_target_nivel_riesgo(crime_counts)
    
    # Clasificación 2: Hotspot Crítico
    y_hotspot_critico = crear_target_hotspot_critico(crime_counts, umbral=5)
    
    # Clasificación 3: Tendencia
    y_tendencia = crear_target_tendencia(df_completo)
    
    return y_nivel_riesgo, y_hotspot_critico, y_tendencia


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def obtener_descripcion_target(tipo_clasificacion):
    """
    Retorna descripción del target de clasificación.
    
    Args:
        tipo_clasificacion: 'nivel_riesgo', 'hotspot_critico', 'tendencia'
        
    Returns:
        Dict con descripción, clases y valor operacional
    """
    descripciones = {
        'nivel_riesgo': {
            'nombre': 'Nivel de Riesgo',
            'pregunta': '¿Qué nivel de recursos necesita esta zona?',
            'clases': {
                0: 'BAJO (0-2) → Patrullaje rutinario',
                1: 'MEDIO (3-5) → Patrullaje reforzado',
                2: 'ALTO (6-10) → Operativo focalizado',
                3: 'MUY ALTO (>10) → Intervención especial'
            },
            'num_clases': 4,
            'tipo': 'multiclase'
        },
        'hotspot_critico': {
            'nombre': 'Hotspot Crítico',
            'pregunta': '¿Debo intervenir en esta zona esta semana?',
            'clases': {
                0: 'NO (≤5) → Vigilancia normal',
                1: 'SÍ (>5) → Desplegar operativo especial'
            },
            'num_clases': 2,
            'tipo': 'binaria'
        },
        'tendencia': {
            'nombre': 'Tendencia de Riesgo',
            'pregunta': '¿Esta zona está mejorando o empeorando?',
            'clases': {
                0: 'DESCENSO → Zona mejorando',
                1: 'ESTABLE → Zona estable',
                2: 'ESCALADA → Zona empeorando (ALERTA)'
            },
            'num_clases': 3,
            'tipo': 'multiclase'
        }
    }
    
    return descripciones.get(tipo_clasificacion)


def imprimir_distribucion_target(y_target, nombre_target):
    """
    Imprime la distribución de clases del target.
    
    Args:
        y_target: Array con las clases
        nombre_target: Nombre del target
    """
    unique, counts = np.unique(y_target, return_counts=True)
    total = len(y_target)
    
    print(f"\n   Distribución de {nombre_target}:")
    for clase, count in zip(unique, counts):
        pct = (count / total) * 100
        print(f"      Clase {clase}: {count:,} ({pct:.2f}%)")
