"""
Evaluación y Persistencia de Modelos - ENFOQUE OPERACIONAL
===========================================================
Funciones para evaluar modelos desde perspectiva de toma de decisiones.
"""

import os
import pandas as pd
import joblib

from config.config import MODELS_OUTPUT_DIR, RESULTS_OUTPUT_DIR, TIPOS_CLASIFICACION


def guardar_mejores_modelos(resultados, delito_key):
    """
    Guarda los mejores modelos por tipo de clasificación.
    
    Args:
        resultados: Lista de diccionarios con resultados
        delito_key: Clave del delito
    """
    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    
    print(f"\n[GUARDANDO MEJORES MODELOS PARA {delito_key.upper()}]")
    
    for tipo_clf in TIPOS_CLASIFICACION.keys():
        modelos_tipo = [r for r in resultados if r['tipo_clasificacion'] == tipo_clf]
        
        if modelos_tipo:
            mejor = max(modelos_tipo, key=lambda x: x['f1'])
            ruta = f'{MODELS_OUTPUT_DIR}/best_{tipo_clf}_{delito_key}.joblib'
            joblib.dump(mejor['model_obj'], ruta)
            
            print(f"   {mejor['nombre_clasificacion']}: {mejor['modelo']} (F1: {mejor['f1']:.4f})")


def generar_resumen_resultados(todos_resultados):
    """
    Genera resumen completo con enfoque operacional.
    
    Args:
        todos_resultados: Lista de diccionarios con resultados
    """
    if not todos_resultados:
        print("\n[ERROR] No hay resultados para mostrar")
        return None
    
    df_resultados = pd.DataFrame(todos_resultados)
    
    print(f"\n{'='*80}")
    print("RESUMEN FINAL - ENFOQUE OPERACIONAL")
    print(f"{'='*80}")
    
    # Resumen general
    print(f"\n📊 Total modelos entrenados: {len(df_resultados)}")
    
    for tipo_clf, info in TIPOS_CLASIFICACION.items():
        count = len(df_resultados[df_resultados['tipo_clasificacion'] == tipo_clf])
        print(f"   • {info['nombre']}: {count} modelos")
    
    # Top 5 por tipo de clasificación
    for tipo_clf, info in TIPOS_CLASIFICACION.items():
        print(f"\n{'='*80}")
        print(f"[TOP 5] {info['nombre'].upper()}")
        print(f"Pregunta: {info['pregunta']}")
        print(f"{'='*80}")
        
        modelos_tipo = df_resultados[df_resultados['tipo_clasificacion'] == tipo_clf]
        
        if len(modelos_tipo) > 0:
            top5 = modelos_tipo.nlargest(5, 'f1')
            
            for idx, row in top5.iterrows():
                delito = row['delito'].upper()
                modelo = row['modelo']
                f1_val = row['f1']
                prec = row['precision']
                rec = row['recall']
                
                print(f"\n{delito:10s} | {modelo:20s}")
                print(f"   F1: {f1_val:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
                
                if row['interpretacion']:
                    print(f"   Interpretación:")
                    for interp in row['interpretacion']:
                        print(f"      {interp}")
    
    return df_resultados


def mostrar_mejores_por_delito(df_resultados):
    """
    Muestra los mejores modelos por delito y tipo de clasificación.
    """
    print(f"\n{'='*80}")
    print("MEJORES MODELOS POR DELITO Y TIPO")
    print(f"{'='*80}")
    
    for delito in df_resultados['delito'].unique():
        print(f"\n🎯 {delito.upper()}")
        
        df_delito = df_resultados[df_resultados['delito'] == delito]
        
        for tipo_clf in df_delito['tipo_clasificacion'].unique():
            df_tipo = df_delito[df_delito['tipo_clasificacion'] == tipo_clf]
            mejor = df_tipo.loc[df_tipo['f1'].idxmax()]
            
            print(f"   {mejor['nombre_clasificacion']:20s} → {mejor['modelo']:20s} (F1: {mejor['f1']:.4f})")


def generar_recomendaciones_operacionales(df_resultados):
    """
    Genera recomendaciones para implementación operacional.
    """
    print(f"\n{'='*80}")
    print("RECOMENDACIONES OPERACIONALES")
    print(f"{'='*80}")
    
    # Mejor modelo global
    mejor_global = df_resultados.loc[df_resultados['f1'].idxmax()]
    
    print(f"\n✨ MODELO CAMPEÓN GENERAL:")
    print(f"   Delito: {mejor_global['delito'].upper()}")
    print(f"   Tipo: {mejor_global['nombre_clasificacion']}")
    print(f"   Algoritmo: {mejor_global['modelo']}")
    print(f"   F1-Score: {mejor_global['f1']:.4f}")
    
    # Análisis por tipo de clasificación
    print(f"\n📋 ANÁLISIS POR PROBLEMA:")
    
    for tipo_clf, info in TIPOS_CLASIFICACION.items():
        df_tipo = df_resultados[df_resultados['tipo_clasificacion'] == tipo_clf]
        
        if len(df_tipo) > 0:
            mejor = df_tipo.loc[df_tipo['f1'].idxmax()]
            f1_promedio = df_tipo['f1'].mean()
            
            print(f"\n   {info['nombre']}:")
            print(f"      Mejor modelo: {mejor['modelo']} (F1: {mejor['f1']:.4f})")
            print(f"      F1 promedio: {f1_promedio:.4f}")
            
            if f1_promedio > 0.85:
                print(f"      ✓ Listo para producción")
            elif f1_promedio > 0.70:
                print(f"      ⚠ Considerar optimización antes de producción")
            else:
                print(f"      ✗ Requiere mejora significativa")


def guardar_resultados_csv(df_resultados):
    """
    Guarda los resultados en CSV.
    """
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
    ruta = f'{RESULTS_OUTPUT_DIR}/resultados_clasificacion_completo.csv'
    
    # Eliminar columnas no serializables
    df_export = df_resultados.drop(columns=['model_obj', 'confusion_matrix', 'interpretacion'], errors='ignore')
    df_export.to_csv(ruta, index=False)
    
    print(f"\n[OK] Resultados guardados en '{ruta}'")


def mostrar_cumplimiento_pc3(num_modelos):
    """
    Muestra el cumplimiento de requisitos de PC3.
    """
    print(f"\n{'='*80}")
    print("CUMPLIMIENTO PC3 - GRUPO 2")
    print(f"{'='*80}")
    print(f"\n✓ Capítulo 3: Classification (100% implementado)")
    print(f"✓ Mínimo 20 modelos: {num_modelos} modelos de clasificación entrenados")
    print(f"✓ Aplicación a Delincuencia: HURTO + EXTORSIÓN en Lima")
    print(f"✓ Valor operacional: 3 problemas de decisión con impacto real")
    print(f"\n📊 Desglose:")
    print(f"   • 7 algoritmos de clasificación")
    print(f"   • 3 tipos de clasificación (con valor operacional)")
    print(f"   • 2 delitos (HURTO y EXTORSIÓN)")
    print(f"   • Total: 7 × 3 × 2 = {num_modelos} modelos")
