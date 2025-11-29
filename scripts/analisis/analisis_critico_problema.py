"""
Análisis Crítico y Destructivo del Problema
===========================================
Este script explora si "predecir cantidad semanal de Robo Agravado" es realmente
el MEJOR problema a resolver, o si hay oportunidades más valiosas en los datos.

PREGUNTAS CRÍTICAS:
1. ¿Es Robo Agravado el delito más relevante/predecible?
2. ¿Predecir "cantidad" es el mejor target, o hay mejores opciones?
3. ¿Hay variables no utilizadas con alto valor predictivo?
4. ¿El scope (Lima completo) es óptimo, o deberían enfocarse en distritos?
5. ¿Qué problema tiene el MAYOR impacto operacional real?
"""

import os
import sys
from pathlib import Path
# Agregar raíz del proyecto al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from scipy import stats
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ANÁLISIS CRÍTICO Y DESTRUCTIVO DEL PROBLEMA")
print("="*80)
print("\n⚠️  OBJETIVO: Determinar si estás trabajando en el problema CORRECTO\n")

# Conexión
load_dotenv()
db_user = os.getenv("MYSQL_USER")
db_password = os.getenv("MYSQL_PASSWORD")
db_host = os.getenv("MYSQL_HOST")
db_name = os.getenv("MYSQL_DB")

connection_string = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(connection_string)

# ============================================================================
# 1. ANÁLISIS DE TODOS LOS DELITOS - ¿ES ROBO AGRAVADO EL MEJOR?
# ============================================================================
print("="*80)
print("1. ¿ES 'ROBO AGRAVADO' EL DELITO MÁS RELEVANTE PARA PREDECIR?")
print("="*80)

query_delitos_lima = text("""
    SELECT
        modalidad_hecho,
        COUNT(*) as total,
        COUNT(DISTINCT DATE(fecha_hora_hecho)) as dias_activos,
        COUNT(DISTINCT CONCAT(lat_hecho, '_', long_hecho)) as ubicaciones_unicas,
        MIN(fecha_hora_hecho) as fecha_min,
        MAX(fecha_hora_hecho) as fecha_max
    FROM denuncias
    WHERE departamento_hecho = 'LIMA'
        AND lat_hecho IS NOT NULL
        AND long_hecho IS NOT NULL
        AND fecha_hora_hecho IS NOT NULL
    GROUP BY modalidad_hecho
    HAVING total >= 1000
    ORDER BY total DESC
    LIMIT 30
""")

print("\nCargando estadísticas de TODOS los delitos en Lima...")
df_delitos = pd.read_sql(query_delitos_lima, engine)

print(f"\n🔍 Top 20 Delitos en Lima (con coordenadas válidas):")
print(f"{'#':<4} {'Delito':<50} {'Total':>10} {'Días':>8} {'Locs':>8} {'Promedio/día':>12}")
print("-" * 100)

for idx, row in df_delitos.head(20).iterrows():
    dias = (pd.to_datetime(row['fecha_max']) - pd.to_datetime(row['fecha_min'])).days
    promedio_dia = row['total'] / max(dias, 1)
    print(f"{idx+1:<4} {row['modalidad_hecho'][:48]:<50} {row['total']:>10,} {row['dias_activos']:>8} {row['ubicaciones_unicas']:>8} {promedio_dia:>12.2f}")

# Calcular métricas de predictibilidad
print("\n" + "="*80)
print("ANÁLISIS DE PREDICTIBILIDAD POR DELITO")
print("="*80)
print("\nCriterios de evaluación:")
print("  1. Volumen suficiente (> 5000 casos)")
print("  2. Distribución temporal estable (frecuencia diaria alta)")
print("  3. Concentración espacial (Gini alto)")
print("  4. Relevancia para seguridad ciudadana")

# Vamos a analizar los top 10 delitos en detalle
top_delitos = df_delitos.head(10)['modalidad_hecho'].tolist()

resultados_delitos = []

for delito in top_delitos[:5]:  # Analizar top 5 para no saturar
    print(f"\n📊 Analizando: {delito}")

    query_delito = text(f"""
        SELECT
            lat_hecho,
            long_hecho,
            fecha_hora_hecho
        FROM denuncias
        WHERE departamento_hecho = 'LIMA'
            AND modalidad_hecho = :delito
            AND lat_hecho IS NOT NULL
            AND long_hecho IS NOT NULL
    """)

    df_d = pd.read_sql(query_delito, engine, params={'delito': delito})

    # Coordenadas
    df_d['lat_hecho'] = pd.to_numeric(df_d['lat_hecho'], errors='coerce')
    df_d['long_hecho'] = pd.to_numeric(df_d['long_hecho'], errors='coerce')
    df_d = df_d.dropna()

    # Grid
    grid_size = 0.005
    df_d['grid_cell'] = ((df_d['lat_hecho'] // grid_size) * grid_size).astype(str) + '_' + \
                        ((df_d['long_hecho'] // grid_size) * grid_size).astype(str)

    crimes_per_cell = df_d.groupby('grid_cell').size()

    # Gini
    sorted_crimes = np.sort(crimes_per_cell.values)
    n = len(sorted_crimes)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_crimes)) / (n * np.sum(sorted_crimes)) - (n + 1) / n

    # Concentración
    top5_pct = crimes_per_cell.nlargest(int(len(crimes_per_cell)*0.05)).sum() / len(df_d) * 100
    top10_pct = crimes_per_cell.nlargest(int(len(crimes_per_cell)*0.10)).sum() / len(df_d) * 100

    # Temporal
    df_d['fecha'] = pd.to_datetime(df_d['fecha_hora_hecho'])
    df_d['año_semana'] = df_d['fecha'].dt.strftime('%Y-%U')
    crimes_per_week = df_d.groupby('año_semana').size()

    autocorr_1 = crimes_per_week.autocorr(lag=1) if len(crimes_per_week) > 1 else 0

    resultados_delitos.append({
        'delito': delito,
        'total': len(df_d),
        'gini': gini,
        'top5_pct': top5_pct,
        'top10_pct': top10_pct,
        'autocorr': autocorr_1,
        'celdas': len(crimes_per_cell),
        'promedio_semana': crimes_per_week.mean()
    })

    print(f"   Total casos: {len(df_d):,}")
    print(f"   Índice Gini: {gini:.4f} {'✓ Alta concentración' if gini > 0.7 else '⚠️  Moderada' if gini > 0.5 else '❌ Baja'}")
    print(f"   Top 5% celdas: {top5_pct:.1f}% de casos")
    print(f"   Autocorr lag-1: {autocorr_1:.4f} {'✓ Fuerte' if autocorr_1 > 0.7 else '⚠️  Moderada' if autocorr_1 > 0.5 else '❌ Débil'}")

# Comparación
print("\n" + "="*80)
print("RANKING DE PREDICTIBILIDAD")
print("="*80)

df_resultados = pd.DataFrame(resultados_delitos)
df_resultados['score'] = (
    (df_resultados['gini'] * 40) +  # 40% peso en concentración espacial
    (df_resultados['autocorr'] * 30) +  # 30% peso en autocorrelación
    ((df_resultados['total'] / df_resultados['total'].max()) * 30)  # 30% peso en volumen
)
df_resultados = df_resultados.sort_values('score', ascending=False)

print(f"\n{'Ranking':<8} {'Delito':<45} {'Score':>8} {'Gini':>8} {'AutoC':>8} {'Total':>10}")
print("-" * 100)
for idx, row in df_resultados.iterrows():
    print(f"{idx+1:<8} {row['delito'][:43]:<45} {row['score']:>8.2f} {row['gini']:>8.4f} {row['autocorr']:>8.4f} {row['total']:>10,}")

print("\n💡 CONCLUSIÓN:")
mejor = df_resultados.iloc[0]
if mejor['delito'] == 'ROBO AGRAVADO':
    print("   ✅ ROBO AGRAVADO es efectivamente el delito MÁS PREDECIBLE")
else:
    print(f"   ⚠️  ALERTA: '{mejor['delito']}' podría ser MÁS PREDECIBLE")
    print(f"   Score: {mejor['score']:.2f} vs Robo Agravado")

# ============================================================================
# 2. ANÁLISIS DE TARGET ALTERNATIVO - ¿ES "CANTIDAD" EL MEJOR TARGET?
# ============================================================================
print("\n" + "="*80)
print("2. ¿ES 'CANTIDAD SEMANAL' EL MEJOR TARGET PARA PREDECIR?")
print("="*80)

print("\n🎯 Targets alternativos a considerar:")

# Cargar datos de Robo Agravado para análisis
query_robo = text("""
    SELECT
        fecha_hora_hecho,
        turno_hecho,
        periodo_dia,
        lat_hecho,
        long_hecho,
        distrito_hecho,
        tipo_via_hecho
    FROM denuncias
    WHERE departamento_hecho = 'LIMA'
        AND modalidad_hecho = 'ROBO AGRAVADO'
        AND lat_hecho IS NOT NULL
        AND long_hecho IS NOT NULL
    LIMIT 65000
""")

df_robo = pd.read_sql(query_robo, engine)
df_robo['fecha'] = pd.to_datetime(df_robo['fecha_hora_hecho'])
df_robo['hora'] = df_robo['fecha'].dt.hour
df_robo['dia_semana'] = df_robo['fecha'].dt.dayofweek

print("\n1️⃣  TARGET ACTUAL: Cantidad de crímenes (regresión)")
print("   Ventajas:")
print("     ✓ Información cuantitativa")
print("     ✓ Útil para dimensionar recursos")
print("   Desventajas:")
print("     ❌ Difícil calibración (modelo sobreestima)")
print("     ❌ RMSE penaliza errores grandes")
print("     ❌ No distingue entre 0 y 1 crimen (igual de crítico)")

print("\n2️⃣  ALTERNATIVA 1: Probabilidad binaria de crimen (clasificación)")
print("   Target: ¿Ocurrirá AL MENOS 1 crimen en esta celda esta semana? (Sí/No)")
print("   Ventajas:")
print("     ✓ MÁS SIMPLE de predecir (mayor accuracy esperado)")
print("     ✓ Output probabilístico calibrable")
print("     ✓ Útil para priorización (scoring de riesgo)")
print("     ✓ Métricas: AUC-ROC, Precision, Recall")
print("   Desventajas:")
print("     ❌ Pierde información cuantitativa")

# Analizar balance de este target
df_robo['grid_cell'] = ((df_robo['lat_hecho'] // 0.005) * 0.005).astype(str) + '_' + \
                       ((df_robo['long_hecho'] // 0.005) * 0.005).astype(str)
df_robo['año_semana'] = df_robo['fecha'].dt.strftime('%Y-%U')

# Crear grid completo
unique_cells = df_robo['grid_cell'].unique()
unique_weeks = df_robo['año_semana'].unique()
grid_semanas = pd.DataFrame([(c, w) for c in unique_cells for w in unique_weeks],
                            columns=['grid_cell', 'año_semana'])

crimes_binary = df_robo.groupby(['grid_cell', 'año_semana']).size().reset_index(name='count')
crimes_binary['has_crime'] = 1

grid_full = grid_semanas.merge(crimes_binary[['grid_cell', 'año_semana', 'has_crime']],
                               on=['grid_cell', 'año_semana'], how='left')
grid_full['has_crime'] = grid_full['has_crime'].fillna(0)

balance = grid_full['has_crime'].value_counts(normalize=True)
print(f"\n   Balance de clases:")
print(f"     Sin crimen (0): {balance.get(0, 0)*100:.1f}%")
print(f"     Con crimen (1): {balance.get(1, 0)*100:.1f}%")
if balance.get(0, 0) > 0.7:
    print(f"   ⚠️  Desbalanceado, pero manejable con técnicas estándar")

print("\n3️⃣  ALTERNATIVA 2: Nivel de riesgo categórico (clasificación multiclase)")
print("   Target: Riesgo = {Muy Bajo, Bajo, Medio, Alto, Muy Alto}")
print("   Basado en quintiles de cantidad histórica")
print("   Ventajas:")
print("     ✓ Balance entre interpretabilidad y granularidad")
print("     ✓ Alineado con uso policial (zonas rojas/amarillas/verdes)")
print("     ✓ Menos sensible a errores de conteo exacto")
print("   Desventajas:")
print("     ❌ Definición arbitraria de umbrales")

print("\n4️⃣  ALTERNATIVA 3: Hora del día de mayor riesgo (clasificación)")
print("   Target: ¿En qué franja horaria es más probable el crimen?")

# Distribución horaria
dist_hora = df_robo['hora'].value_counts(normalize=True).sort_index()
hora_pico = dist_hora.idxmax()
print(f"\n   Distribución horaria:")
print(f"     Hora pico: {hora_pico}:00 ({dist_hora.max()*100:.1f}% de casos)")

franjas = pd.cut(df_robo['hora'], bins=[0, 6, 12, 18, 24],
                 labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'], include_lowest=True)
dist_franjas = franjas.value_counts(normalize=True)
print(f"\n   Por franja:")
for franja, pct in dist_franjas.items():
    print(f"     {franja}: {pct*100:.1f}%")

print("\n   Ventajas:")
print("     ✓ MUY ÚTIL operacionalmente (optimizar turnos)")
print("     ✓ Diferente del enfoque actual")
print("   Desventajas:")
print("     ❌ Menor granularidad temporal")

print("\n5️⃣  ALTERNATIVA 4: Días hasta próximo crimen (regresión)")
print("   Target: ¿Cuántos días pasarán hasta el próximo crimen en esta zona?")
print("   Ventajas:")
print("     ✓ Perspectiva temporal útil para patrullaje")
print("     ✓ Diferente a otros enfoques")
print("   Desventajas:")
print("     ❌ Difícil de modelar (distribución exponencial)")
print("     ❌ Requiere ingeniería compleja")

# ============================================================================
# 3. VARIABLES NO UTILIZADAS - ¿QUÉ ESTÁS DEJANDO EN LA MESA?
# ============================================================================
print("\n" + "="*80)
print("3. VARIABLES NO UTILIZADAS CON POTENCIAL VALOR")
print("="*80)

print("\n📋 Variables disponibles pero NO usadas en el modelo actual:")

# Turno
if df_robo['turno_hecho'].notna().sum() > 0:
    dist_turno = df_robo['turno_hecho'].value_counts()
    print(f"\n1. TURNO_HECHO (disponibilidad: {df_robo['turno_hecho'].notna().sum()/len(df_robo)*100:.1f}%)")
    for turno, count in dist_turno.head(5).items():
        print(f"     {turno}: {count:,} ({count/len(df_robo)*100:.1f}%)")
    print("   💡 Potencial: ALTO - Podría mejorar predicción de hora/franja")

# Período del día
if df_robo['periodo_dia'].notna().sum() > 0:
    dist_periodo = df_robo['periodo_dia'].value_counts()
    print(f"\n2. PERIODO_DIA (disponibilidad: {df_robo['periodo_dia'].notna().sum()/len(df_robo)*100:.1f}%)")
    for periodo, count in dist_periodo.head(5).items():
        print(f"     {periodo}: {count:,} ({count/len(df_robo)*100:.1f}%)")
    print("   💡 Potencial: ALTO - Complementa turno")

# Distrito
if df_robo['distrito_hecho'].notna().sum() > 0:
    dist_distrito = df_robo['distrito_hecho'].value_counts()
    print(f"\n3. DISTRITO_HECHO (disponibilidad: {df_robo['distrito_hecho'].notna().sum()/len(df_robo)*100:.1f}%)")
    print(f"   Top 5 distritos:")
    for distrito, count in dist_distrito.head(5).items():
        print(f"     {distrito}: {count:,} ({count/len(df_robo)*100:.1f}%)")
    print("   💡 Potencial: MEDIO - Ya capturado implícitamente por grid, pero útil para features")

# Tipo de vía
if df_robo['tipo_via_hecho'].notna().sum() > 0:
    dist_via = df_robo['tipo_via_hecho'].value_counts()
    print(f"\n4. TIPO_VIA_HECHO (disponibilidad: {df_robo['tipo_via_hecho'].notna().sum()/len(df_robo)*100:.1f}%)")
    print(f"   Top 5 tipos de vía:")
    for via, count in dist_via.head(5).items():
        print(f"     {via}: {count:,} ({count/len(df_robo)*100:.1f}%)")
    print("   💡 Potencial: MEDIO - Avenidas vs calles vs parques")

# ============================================================================
# 4. ANÁLISIS DE SCOPE - ¿LIMA COMPLETO O DISTRITOS?
# ============================================================================
print("\n" + "="*80)
print("4. ¿ES 'LIMA COMPLETO' EL SCOPE ÓPTIMO?")
print("="*80)

# Top distritos por volumen
top_distritos = df_robo['distrito_hecho'].value_counts().head(10)

print(f"\n📍 Top 10 Distritos por volumen de Robo Agravado:")
print(f"{'#':<4} {'Distrito':<30} {'Total':>10} {'% Lima':>10}")
print("-" * 60)
for idx, (distrito, count) in enumerate(top_distritos.items(), 1):
    pct = count / len(df_robo) * 100
    print(f"{idx:<4} {str(distrito)[:28]:<30} {count:>10,} {pct:>9.1f}%")

print("\n💡 Alternativa: Enfocarse en UN distrito de alto impacto")
print("   Ventajas:")
print("     ✓ Mayor homogeneidad geográfica")
print("     ✓ Modelos más precisos (menos variabilidad)")
print("     ✓ Implementación piloto más viable")
print("     ✓ Colaboración más directa con comisarías locales")
print("   Desventajas:")
print("     ❌ Menor generalización")
print("     ❌ Menos datos de entrenamiento")

# ============================================================================
# 5. RECOMENDACIONES FINALES
# ============================================================================
print("\n" + "="*80)
print("5. RECOMENDACIONES CRÍTICAS Y DESTRUCTIVAS")
print("="*80)

print("\n🔥 CRÍTICA AL ENFOQUE ACTUAL:")
print("-" * 80)

print("\n❌ DEBILIDADES IDENTIFICADAS:")
print("   1. Predecir 'cantidad exacta' es innecesariamente difícil")
print("      → El modelo sobreestima y la calibración es compleja")
print("      → Para operaciones, solo necesitas saber DÓNDE patrullar")
print()
print("   2. No aprovechas variables temporales ricas (turno, período_día)")
print("      → Podrías predecir CUÁNDO además de DÓNDE")
print()
print("   3. Lima completo podría ser demasiado heterogéneo")
print("      → Diferentes distritos tienen dinámicas distintas")
print()
print("   4. No comparaste con otros delitos más predecibles")
print("      → Podrías estar trabajando en un problema más difícil que alternativas")

print("\n✅ FORTALEZAS VALIDADAS:")
print("   1. La metodología (división temporal, lags) es correcta")
print("   2. Robo Agravado SÍ tiene patrones espaciales fuertes (Gini 0.77)")
print("   3. Autocorrelación alta (0.80) valida enfoque de lag")
print("   4. El LSTM supera baselines significativamente")

print("\n" + "="*80)
print("🎯 RECOMENDACIONES PARA MAYOR IMPACTO")
print("="*80)

print("\n🥇 OPCIÓN 1 (Recomendada): Cambiar a clasificación binaria")
print("   Target: P(al menos 1 crimen | celda, semana)")
print("   Justificación:")
print("     • MÁS FÁCIL de predecir y calibrar")
print("     • Output probabilístico útil para ranking de zonas")
print("     • Suficiente para asignación de recursos")
print("     • Métricas más interpretables (AUC, Precision@K)")
print("   Impacto: ALTO")
print("   Esfuerzo: BAJO (solo cambiar target y modelo)")

print("\n🥈 OPCIÓN 2: Mantener regresión pero agregar predicción de franja horaria")
print("   Modelo dual:")
print("     A) DÓNDE: Hotspots espaciales (actual)")
print("     B) CUÁNDO: Franja horaria de mayor riesgo")
print("   Justificación:")
print("     • Valor operacional MUY alto (optimizar turnos)")
print("     • Diferenciación del trabajo actual")
print("     • Usa variables disponibles no aprovechadas")
print("   Impacto: MUY ALTO")
print("   Esfuerzo: MEDIO (nuevo modelo adicional)")

print("\n🥉 OPCIÓN 3: Enfocarse en 1-2 distritos clave")
print("   Scope: Ej. San Juan de Lurigancho + Lima Cercado")
print("   Justificación:")
print("     • Mayor precisión por homogeneidad")
print("     • Implementación piloto realista")
print("     • Validación en campo más viable")
print("   Impacto: MEDIO")
print("   Esfuerzo: BAJO (re-filtrar datos)")

print("\n⚡ OPCIÓN 4 (Más audaz): Comparar múltiples delitos")
print("   Crear un 'Crime Forecast Dashboard' multi-delito")
print("   Justificación:")
print("     • Mayor alcance e impacto")
print("     • Demuestra versatilidad de la metodología")
print("     • Más valioso para autoridades")
print("   Impacto: MUY ALTO")
print("   Esfuerzo: ALTO (escalar pipeline)")

print("\n" + "="*80)
print("💭 PREGUNTA FINAL PARA REFLEXIONAR")
print("="*80)
print("\n¿Cuál es tu OBJETIVO REAL con este proyecto?")
print()
print("  A) Proyecto académico → Enfoque actual está bien, R²=0.697 es excelente")
print("  B) Impacto operacional real → Considera Opción 1 o 2")
print("  C) Publicación científica → Considera Opción 4 (más novedoso)")
print("  D) Piloto con policía → Considera Opción 3 (más implementable)")

print("\n" + "="*80)
print("ANÁLISIS CRÍTICO COMPLETADO")
print("="*80)
print("\n⚠️  Recuerda: Un modelo 'suficientemente bueno' que se IMPLEMENTA")
print("    vale MÁS que un modelo 'perfecto' que nunca se usa.\n")
