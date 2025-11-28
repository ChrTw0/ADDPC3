"""
Pipeline Completo - SOLO CLASIFICACIÓN (Capítulo 3)
====================================================
Implementación de 42 modelos de clasificación con valor operacional.

Modelos implementados:
- 7 algoritmos de clasificación
- 3 tipos de problemas (nivel_riesgo, hotspot_critico, tendencia)
- 2 delitos (HURTO y EXTORSIÓN)

Total: 7 × 3 × 2 = 42 modelos de clasificación
"""

import warnings
warnings.filterwarnings('ignore')

from config.config import DELITOS, MODELOS_CLASIFICACION, TIPOS_CLASIFICACION
from utils.data_preparation import preparar_datos_completo
from models.classification_models import entrenar_modelo_clasificacion
from utils.model_evaluation import (
    guardar_mejores_modelos, generar_resumen_resultados,
    mostrar_mejores_por_delito, generar_recomendaciones_operacionales,
    guardar_resultados_csv, mostrar_cumplimiento_pc3
)


def procesar_delito_completo(delito_key, optimizar_hiperparametros=False):
    """
    Procesa un delito con TODOS los modelos de clasificación.
    
    Args:
        delito_key: Nombre del delito ('hurto', 'extorsion')
        optimizar_hiperparametros: Si True, busca mejores hiperparámetros
        
    Returns:
        Lista de resultados de todos los modelos
    """
    delito_sql = DELITOS[delito_key]
    
    print(f"\n{'='*80}")
    print(f"PROCESANDO: {delito_sql}{' [CON OPTIMIZACIÓN]' if optimizar_hiperparametros else ''}")
    print(f"{'='*80}")
    
    # 1. PREPARAR DATOS
    datos = preparar_datos_completo(delito_key)
    if datos is None:
        return None
    
    X_train = datos['X_train']
    X_test = datos['X_test']
    targets = datos['targets']
    
    # 2. ENTRENAR MODELOS
    print(f"\n[6] Entrenando modelos de clasificación...")
    resultados = []
    
    # Iterar sobre los 3 tipos de clasificación
    for tipo_clf, info in TIPOS_CLASIFICACION.items():
        print(f"\n   === {info['nombre'].upper()} ===")
        print(f"   Pregunta: {info['pregunta']}")
        
        y_train = targets[tipo_clf]['train']
        y_test = targets[tipo_clf]['test']
        
        # Entrenar los 7 algoritmos para este tipo
        for modelo in MODELOS_CLASIFICACION:
            try:
                resultado = entrenar_modelo_clasificacion(
                    modelo, tipo_clf,
                    X_train, y_train, X_test, y_test,
                    optimizar=optimizar_hiperparametros
                )
                resultado['delito'] = delito_key
                resultados.append(resultado)
            except Exception as e:
                print(f"         [ERROR] {modelo}: {e}")
    
    # 3. GUARDAR MEJORES MODELOS
    print(f"\n[OK] {len(resultados)} modelos entrenados para {delito_sql}")
    guardar_mejores_modelos(resultados, delito_key)
    
    return resultados


def main():
    """
    Función principal del pipeline.
    """
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║    PIPELINE CLASIFICACIÓN - CAPÍTULO 3 (42 modelos)                   ║
    ║                                                                        ║
    ║  ENFOQUE: 100% Clasificación con Valor Operacional                    ║
    ║                                                                        ║
    ║  3 TIPOS DE CLASIFICACIÓN:                                            ║
    ║    1. Nivel de Riesgo (4 clases)                                      ║
    ║       → ¿Qué recursos necesita esta zona?                             ║
    ║                                                                        ║
    ║    2. Hotspot Crítico (binaria)                                       ║
    ║       → ¿Intervenir esta zona?                                        ║
    ║                                                                        ║
    ║    3. Tendencia (3 clases)                                            ║
    ║       → ¿Zona mejorando o empeorando?                                 ║
    ║                                                                        ║
    ║  7 ALGORITMOS × 3 TIPOS × 2 DELITOS = 42 MODELOS                     ║
    ║                                                                        ║
    ║  Cumple PC3 Grupo 2: Cap. 3 Classification + 20+ modelos             ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Selección de delito
    print("\nOpciones de delito:")
    print("  1. Solo HURTO (213K registros)")
    print("  2. Solo EXTORSIÓN (32K registros)")
    print("  3. AMBOS (recomendado para 42 modelos)")
    
    opcion = input("\nOpción (1/2/3): ").strip()
    
    procesar_hurto = opcion in ['1', '3']
    procesar_extorsion = opcion in ['2', '3']
    
    # Optimización de hiperparámetros
    print("\n¿Optimizar hiperparámetros?")
    print("  1. NO - Entrenamiento rápido (~10-15 min)")
    print("  2. SÍ - Búsqueda de mejores parámetros (~30-45 min)")
    
    opcion_opt = input("\nOpción (1/2): ").strip()
    optimizar = opcion_opt == '2'
    
    if optimizar:
        print("\n[INFO] Optimización activada. Esto puede tardar 30-45 minutos.")
    else:
        print("\n[INFO] Entrenamiento rápido con parámetros por defecto.")
    
    # Procesar delitos
    todos_resultados = []
    
    if procesar_hurto:
        resultados = procesar_delito_completo('hurto', optimizar_hiperparametros=optimizar)
        if resultados:
            todos_resultados.extend(resultados)
    
    if procesar_extorsion:
        resultados = procesar_delito_completo('extorsion', optimizar_hiperparametros=optimizar)
        if resultados:
            todos_resultados.extend(resultados)
    
    # RESUMEN FINAL
    if todos_resultados:
        df_resultados = generar_resumen_resultados(todos_resultados)
        mostrar_mejores_por_delito(df_resultados)
        generar_recomendaciones_operacionales(df_resultados)
        guardar_resultados_csv(df_resultados)
        mostrar_cumplimiento_pc3(len(df_resultados))
        
        print(f"\n{'='*80}")
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
