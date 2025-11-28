"""
Feature Engineering
===================
Funciones para crear y transformar features.
"""

import pandas as pd
import numpy as np
from config.config import GRID_SIZE, NUM_LAGS


def crear_features_espaciales(lat, long):
    """
    Crea features espaciales a partir de coordenadas.
    
    Args:
        lat: Latitud
        long: Longitud
        
    Returns:
        Tupla: (grid_lat, grid_long, grid_cell)
    """
    grid_lat = (lat // GRID_SIZE) * GRID_SIZE
    grid_long = (long // GRID_SIZE) * GRID_SIZE
    grid_cell = f"{grid_lat}_{grid_long}"
    
    return grid_lat, grid_long, grid_cell


def crear_features_lags(df, target_col, group_col, num_lags=NUM_LAGS):
    """
    Crea features de lags temporales.
    
    Args:
        df: DataFrame
        target_col: Columna objetivo para crear lags
        group_col: Columna para agrupar (ej: 'grid_cell')
        num_lags: Número de lags a crear
        
    Returns:
        DataFrame con columnas de lags añadidas
    """
    df = df.sort_values([group_col, 'año_semana'])
    
    for lag in range(1, num_lags + 1):
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    
    return df


def agregar_features_estadisticas(df, value_col, group_col):
    """
    Agrega features estadísticas (media, std, etc.).
    
    Args:
        df: DataFrame
        value_col: Columna con valores
        group_col: Columna para agrupar
        
    Returns:
        DataFrame con features estadísticas
    """
    stats = df.groupby(group_col)[value_col].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    return df.merge(stats, on=group_col, how='left')
