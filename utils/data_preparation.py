"""
Preparación y Extracción de Datos - CLASIFICACIÓN
==================================================
Funciones para extraer datos de MySQL y prepararlos para clasificación.
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler

from models.common import get_db_connection
from utils.target_engineering import crear_todos_los_targets, imprimir_distribucion_target
from config.config import (
    DELITOS, GRID_SIZE, NUM_LAGS, TRAIN_TEST_SPLIT, FEATURE_COLS
)


def extraer_datos_delito(delito_key):
    """
    Extrae datos de un delito desde MySQL.
    
    Args:
        delito_key: Clave del delito ('hurto', 'extorsion')
        
    Returns:
        DataFrame con los datos extraídos
    """
    delito_sql = DELITOS[delito_key]
    
    print(f"[1] Extrayendo datos de {delito_sql}...")
    
    engine = get_db_connection()
    if engine is None:
        print("[ERROR] No se pudo conectar a la base de datos")
        return None
    
    query = text("""
        SELECT
            lat_hecho,
            long_hecho,
            fecha_hora_hecho
        FROM denuncias
        WHERE departamento_hecho = 'LIMA'
            AND modalidad_hecho = :delito
            AND lat_hecho IS NOT NULL
            AND long_hecho IS NOT NULL
            AND fecha_hora_hecho >= '2020-01-01'
    """)
    
    df = pd.read_sql(query, engine, params={'delito': delito_sql})
    print(f"   {len(df):,} registros cargados")
    
    return df


def crear_grid_espacial(df):
    """
    Crea un grid espacial para agrupar crímenes.
    """
    df['grid_lat'] = (df['lat_hecho'] // GRID_SIZE) * GRID_SIZE
    df['grid_long'] = (df['long_hecho'] // GRID_SIZE) * GRID_SIZE
    df['grid_cell'] = df['grid_lat'].astype(str) + '_' + df['grid_long'].astype(str)
    
    return df


def crear_features_temporales(df):
    """
    Crea features temporales (mes, día de semana, año-semana).
    """
    df['fecha'] = pd.to_datetime(df['fecha_hora_hecho'])
    df['año_semana'] = df['fecha'].dt.strftime('%Y-%U')
    df['mes'] = df['fecha'].dt.month
    df['dia_semana'] = df['fecha'].dt.dayofweek
    
    return df


def preparar_datos_completo(delito_key):
    """
    Pipeline completo de preparación de datos para CLASIFICACIÓN.
    
    Args:
        delito_key: Clave del delito ('hurto', 'extorsion')
        
    Returns:
        Dict con todos los datos preparados para los 3 tipos de clasificación
    """
    # 1. Extraer datos
    df = extraer_datos_delito(delito_key)
    if df is None:
        return None
    
    # 2. Crear grid espacial
    print(f"[2] Creando features espaciales...")
    df = crear_grid_espacial(df)
    
    # 3. Crear features temporales
    df = crear_features_temporales(df)
    
    # 4. Agrupar por semana y grid
    hotspot_counts = df.groupby(['grid_cell', 'año_semana']).size().reset_index(name='crime_count')
    
    # 5. Añadir features temporales al agregado
    merged = hotspot_counts.merge(
        df[['año_semana', 'mes', 'dia_semana']].drop_duplicates(),
        on='año_semana',
        how='left'
    )
    
    # 6. Crear lags
    print(f"[3] Creando features de lags temporales...")
    merged = merged.sort_values(['grid_cell', 'año_semana'])
    for lag in range(1, NUM_LAGS + 1):
        merged[f'crime_count_lag_{lag}'] = merged.groupby('grid_cell')['crime_count'].shift(lag)
    
    merged = merged.dropna()
    print(f"   {len(merged):,} registros con features completas")
    
    # 7. Crear los 3 tipos de targets de clasificación
    print(f"[4] Creando targets de clasificación...")
    y_nivel_riesgo, y_hotspot_critico, y_tendencia = crear_todos_los_targets(merged)
    
    # Mostrar distribución de clases
    imprimir_distribucion_target(y_nivel_riesgo, "Nivel de Riesgo")
    imprimir_distribucion_target(y_hotspot_critico, "Hotspot Crítico")
    imprimir_distribucion_target(y_tendencia, "Tendencia")
    
    # 8. Separar features
    X = merged[FEATURE_COLS].values
    
    # 9. Split temporal
    print(f"\n[5] Dividiendo datos (train/test: {TRAIN_TEST_SPLIT:.0%}/{1-TRAIN_TEST_SPLIT:.0%})...")
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    # Splits para cada tipo de clasificación
    y_nivel_riesgo_train = y_nivel_riesgo[:split_idx]
    y_nivel_riesgo_test = y_nivel_riesgo[split_idx:]
    
    y_hotspot_train = y_hotspot_critico[:split_idx]
    y_hotspot_test = y_hotspot_critico[split_idx:]
    
    y_tendencia_train = y_tendencia[:split_idx]
    y_tendencia_test = y_tendencia[split_idx:]
    
    # 10. Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Retornar todo en un diccionario organizado
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'targets': {
            'nivel_riesgo': {
                'train': y_nivel_riesgo_train,
                'test': y_nivel_riesgo_test
            },
            'hotspot_critico': {
                'train': y_hotspot_train,
                'test': y_hotspot_test
            },
            'tendencia': {
                'train': y_tendencia_train,
                'test': y_tendencia_test
            }
        },
        'scaler': scaler,
        'df_completo': merged
    }
