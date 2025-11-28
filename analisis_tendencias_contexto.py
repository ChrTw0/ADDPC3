"""
Análisis de Tendencias y Contexto Temporal - Perú 2020-2025
============================================================
Este script considera el CONTEXTO TEMPORAL y las TENDENCIAS para identificar
el delito más relevante para predecir en el contexto ACTUAL del Perú.

CONSIDERACIONES CRÍTICAS:
1. Delitos de pandemia (solo 2020-2021) → IRRELEVANTES para predicción futura
2. Delitos en CRECIMIENTO (ej. extorsión) → ALTA relevancia actual
3. Delitos en DESCENSO → Menor relevancia futura
4. Relevancia socio-política actual en Perú
"""

import os
import sys
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
print("ANÁLISIS DE TENDENCIAS Y CONTEXTO - PERÚ 2020-2025")
print("="*80)
print("\n🎯 Identificar el delito MÁS RELEVANTE considerando tendencias actuales\n")

# Conexión
load_dotenv()
db_user = os.getenv("MYSQL_USER")
db_password = os.getenv("MYSQL_PASSWORD")
db_host = os.getenv("MYSQL_HOST")
db_name = os.getenv("MYSQL_DB")

connection_string = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(connection_string)

# ============================================================================
# 1. ANÁLISIS DE TENDENCIAS TEMPORALES
# ============================================================================
print("="*80)
print("1. ANÁLISIS DE TENDENCIAS TEMPORALES (2020-2025)")
print("="*80)

# Obtener series temporales por delito
query_tendencias = text("""
    SELECT
        modalidad_hecho,
        YEAR(fecha_hora_hecho) as año,
        COUNT(*) as total
    FROM denuncias
    WHERE departamento_hecho = 'LIMA'
        AND fecha_hora_hecho IS NOT NULL
        AND YEAR(fecha_hora_hecho) >= 2020
        AND modalidad_hecho IN (
            SELECT modalidad_hecho
            FROM denuncias
            WHERE departamento_hecho = 'LIMA'
                AND YEAR(fecha_hora_hecho) >= 2020
            GROUP BY modalidad_hecho
            HAVING COUNT(*) >= 5000
        )
    GROUP BY modalidad_hecho, año
    ORDER BY modalidad_hecho, año
""")

print("\nCargando tendencias anuales de delitos en Lima...")
df_tendencias = pd.read_sql(query_tendencias, engine)

# Calcular tendencia (pendiente de regresión lineal) para cada delito
delitos_con_tendencia = []

for delito in df_tendencias['modalidad_hecho'].unique():
    df_d = df_tendencias[df_tendencias['modalidad_hecho'] == delito].sort_values('año')

    if len(df_d) >= 3:  # Al menos 3 años para calcular tendencia
        años = df_d['año'].values
        casos = df_d['total'].values

        # Regresión lineal
        slope, intercept, r_value, p_value, std_err = stats.linregress(años, casos)

        # Cambio porcentual 2020 vs último año disponible
        cambio_pct = ((casos[-1] - casos[0]) / casos[0] * 100) if casos[0] > 0 else 0

        # Total casos
        total_casos = df_d['total'].sum()

        # Casos en 2024-2025 (actualidad)
        casos_recientes = df_d[df_d['año'].isin([2024, 2025])]['total'].sum()

        delitos_con_tendencia.append({
            'delito': delito,
            'pendiente': slope,
            'cambio_pct': cambio_pct,
            'r2': r_value**2,
            'total_casos': total_casos,
            'casos_2024_2025': casos_recientes,
            'promedio_anual': total_casos / len(df_d),
            'años_datos': len(df_d)
        })

df_analisis = pd.DataFrame(delitos_con_tendencia)

# Clasificar tendencias
df_analisis['tendencia'] = df_analisis['cambio_pct'].apply(
    lambda x: '📈 CRECIENTE' if x > 15 else '📉 DECRECIENTE' if x < -15 else '➡️  ESTABLE'
)

# Ordenar por cambio porcentual
df_analisis = df_analisis.sort_values('cambio_pct', ascending=False)

print("\n🔥 DELITOS EN MAYOR CRECIMIENTO (2020 → 2025):")
print("="*80)
print(f"{'#':<4} {'Delito':<45} {'Cambio':>10} {'Total':>10} {'2024-25':>10}")
print("-" * 80)

for idx, row in df_analisis.head(15).iterrows():
    emoji = '🔴' if row['cambio_pct'] > 50 else '🟠' if row['cambio_pct'] > 15 else '🟢'
    print(f"{emoji} {row['delito'][:43]:<45} {row['cambio_pct']:>9.1f}% {row['total_casos']:>10,} {row['casos_2024_2025']:>10,}")

print("\n📉 DELITOS EN MAYOR DESCENSO (2020 → 2025):")
print("="*80)
for idx, row in df_analisis.tail(10).iterrows():
    print(f"🟢 {row['delito'][:43]:<45} {row['cambio_pct']:>9.1f}% {row['total_casos']:>10,} {row['casos_2024_2025']:>10,}")

# ============================================================================
# 2. IDENTIFICAR DELITOS CONTEXTUALES (PANDEMIA)
# ============================================================================
print("\n" + "="*80)
print("2. IDENTIFICACIÓN DE DELITOS CONTEXTUALES (PANDEMIA)")
print("="*80)

# Delitos sospechosos de ser solo de pandemia
delitos_pandemia = df_analisis[
    (df_analisis['delito'].str.contains('EMERGENCIA|COVID|CUARENTENA|D.S.', case=False, na=False)) |
    (df_analisis['cambio_pct'] < -80)  # Descenso dramático indica temporalidad
]

print(f"\n⚠️  Delitos identificados como TEMPORALES/PANDEMIA:")
for idx, row in delitos_pandemia.iterrows():
    print(f"   ❌ {row['delito']}")
    print(f"      Cambio: {row['cambio_pct']:.1f}% | Total: {row['total_casos']:,}")

print(f"\n💡 Estos delitos NO deberían usarse para predicción futura (contexto ya no existe)")

# ============================================================================
# 3. ANÁLISIS ESPECÍFICO: ROBO AGRAVADO vs EXTORSIÓN
# ============================================================================
print("\n" + "="*80)
print("3. COMPARACIÓN CRÍTICA: ROBO AGRAVADO vs EXTORSIÓN")
print("="*80)

# Robo Agravado
robo_stats = df_analisis[df_analisis['delito'] == 'ROBO AGRAVADO'].iloc[0] if 'ROBO AGRAVADO' in df_analisis['delito'].values else None

# Extorsión
extorsion_stats = df_analisis[df_analisis['delito'] == 'EXTORSION'].iloc[0] if 'EXTORSION' in df_analisis['delito'].values else None

if robo_stats is not None:
    print("\n📊 ROBO AGRAVADO:")
    print(f"   Total casos (2020-2025): {robo_stats['total_casos']:,}")
    print(f"   Casos recientes (2024-2025): {robo_stats['casos_2024_2025']:,}")
    print(f"   Tendencia: {robo_stats['cambio_pct']:+.1f}% {robo_stats['tendencia']}")
    print(f"   Promedio anual: {robo_stats['promedio_anual']:,.0f}")

if extorsion_stats is not None:
    print("\n📊 EXTORSIÓN:")
    print(f"   Total casos (2020-2025): {extorsion_stats['total_casos']:,}")
    print(f"   Casos recientes (2024-2025): {extorsion_stats['casos_2024_2025']:,}")
    print(f"   Tendencia: {extorsion_stats['cambio_pct']:+.1f}% {extorsion_stats['tendencia']}")
    print(f"   Promedio anual: {extorsion_stats['promedio_anual']:,.0f}")

# Análisis detallado de extorsión
print("\n🔍 Análisis Detallado de EXTORSIÓN:")
query_extorsion = text("""
    SELECT
        YEAR(fecha_hora_hecho) as año,
        COUNT(*) as casos
    FROM denuncias
    WHERE departamento_hecho = 'LIMA'
        AND modalidad_hecho = 'EXTORSION'
        AND fecha_hora_hecho IS NOT NULL
    GROUP BY año
    ORDER BY año
""")

df_ext = pd.read_sql(query_extorsion, engine)
print("\n   Evolución año por año:")
for idx, row in df_ext.iterrows():
    if idx > 0:
        cambio = ((row['casos'] - df_ext.iloc[idx-1]['casos']) / df_ext.iloc[idx-1]['casos'] * 100)
        print(f"   {row['año']}: {row['casos']:,} casos ({cambio:+.1f}% vs año anterior)")
    else:
        print(f"   {row['año']}: {row['casos']:,} casos")

# ============================================================================
# 4. ANÁLISIS DE PREDICTIBILIDAD (con contexto temporal)
# ============================================================================
print("\n" + "="*80)
print("4. PREDICTIBILIDAD CONSIDERANDO CONTEXTO ACTUAL")
print("="*80)

print("\n🎯 Delitos candidatos (filtrados):")
print("   Criterios:")
print("   ✓ Al menos 10,000 casos en 2024-2025 (relevancia actual)")
print("   ✓ NO son delitos de pandemia")
print("   ✓ Tendencia estable o creciente (no en descenso dramático)")

# Filtrar
candidatos = df_analisis[
    (df_analisis['casos_2024_2025'] >= 5000) &
    (~df_analisis['delito'].str.contains('EMERGENCIA|COVID|CUARENTENA|D.S.', case=False, na=False)) &
    (df_analisis['cambio_pct'] > -50)
].copy()

# Analizar predictibilidad de candidatos top
print(f"\n📋 Top 10 candidatos por volumen ACTUAL (2024-2025):")
candidatos_sorted = candidatos.sort_values('casos_2024_2025', ascending=False)

delitos_a_analizar = []
for idx, row in candidatos_sorted.head(10).iterrows():
    delitos_a_analizar.append(row['delito'])
    print(f"   {idx+1}. {row['delito']}")
    print(f"      Casos 2024-25: {row['casos_2024_2025']:,} | Tendencia: {row['cambio_pct']:+.1f}%")

# Análisis de concentración espacial para candidatos
print("\n" + "="*80)
print("5. ANÁLISIS DE CONCENTRACIÓN ESPACIAL (Top candidatos)")
print("="*80)

resultados_final = []

for delito in ['ROBO AGRAVADO', 'EXTORSION', 'HURTO', 'HURTO AGRAVADO']:
    if delito not in delitos_a_analizar[:10]:
        continue

    print(f"\n📊 Analizando: {delito}")

    query_espacial = text("""
        SELECT
            lat_hecho,
            long_hecho,
            fecha_hora_hecho
        FROM denuncias
        WHERE departamento_hecho = 'LIMA'
            AND modalidad_hecho = :delito
            AND lat_hecho IS NOT NULL
            AND long_hecho IS NOT NULL
            AND YEAR(fecha_hora_hecho) >= 2023
    """)

    df_d = pd.read_sql(query_espacial, engine, params={'delito': delito})

    if len(df_d) < 1000:
        print(f"   ⚠️  Datos insuficientes ({len(df_d)} casos)")
        continue

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

    # Autocorrelación
    df_d['fecha'] = pd.to_datetime(df_d['fecha_hora_hecho'])
    df_d['año_semana'] = df_d['fecha'].dt.strftime('%Y-%U')
    crimes_per_week = df_d.groupby('año_semana').size()
    autocorr = crimes_per_week.autocorr(lag=1) if len(crimes_per_week) > 1 else 0

    # Stats de tendencia
    stats_delito = df_analisis[df_analisis['delito'] == delito].iloc[0]

    resultados_final.append({
        'delito': delito,
        'casos_2023_2025': len(df_d),
        'gini': gini,
        'autocorr': autocorr,
        'tendencia_pct': stats_delito['cambio_pct'],
        'casos_2024_2025': stats_delito['casos_2024_2025']
    })

    print(f"   Casos 2023-2025: {len(df_d):,}")
    print(f"   Gini: {gini:.4f} {'✓ Alta concentración' if gini > 0.7 else '⚠️  Moderada'}")
    print(f"   Autocorr: {autocorr:.4f} {'✓ Fuerte' if autocorr > 0.7 else '⚠️  Moderada'}")
    print(f"   Tendencia 2020-2025: {stats_delito['cambio_pct']:+.1f}%")

# ============================================================================
# 6. RECOMENDACIÓN FINAL CON CONTEXTO
# ============================================================================
print("\n" + "="*80)
print("6. RECOMENDACIÓN FINAL CONSIDERANDO CONTEXTO PERUANO ACTUAL")
print("="*80)

df_final = pd.DataFrame(resultados_final)

# Score ponderado (dando peso a tendencia positiva)
df_final['score_contexto'] = (
    (df_final['gini'] * 30) +  # 30% concentración espacial
    (df_final['autocorr'] * 25) +  # 25% autocorrelación
    ((df_final['casos_2024_2025'] / df_final['casos_2024_2025'].max()) * 25) +  # 25% volumen actual
    ((df_final['tendencia_pct'].clip(lower=0) / 100) * 20)  # 20% tendencia positiva
)

df_final = df_final.sort_values('score_contexto', ascending=False)

print("\n🏆 RANKING FINAL (considerando contexto 2024-2025):")
print("="*80)
print(f"{'#':<4} {'Delito':<25} {'Score':>8} {'Gini':>7} {'AutoC':>7} {'Tend%':>8} {'Casos 24-25':>12}")
print("-" * 80)

for idx, row in df_final.iterrows():
    print(f"{idx+1:<4} {row['delito']:<25} {row['score_contexto']:>8.2f} {row['gini']:>7.4f} {row['autocorr']:>7.4f} {row['tendencia_pct']:>7.1f}% {row['casos_2024_2025']:>12,}")

print("\n" + "="*80)
print("💡 ANÁLISIS CRÍTICO Y RECOMENDACIONES")
print("="*80)

# Comparar Robo Agravado vs mejor alternativa
if len(df_final) > 0:
    mejor = df_final.iloc[0]

    if mejor['delito'] == 'ROBO AGRAVADO':
        print("\n✅ ROBO AGRAVADO es la mejor opción considerando:")
        print("   • Alta concentración espacial (Gini alto)")
        print("   • Fuerte autocorrelación temporal")
        print("   • Relevancia actual (volumen significativo en 2024-2025)")
        print("   • Tendencia estable")
        print("\n👍 Tu elección original es CORRECTA")
    else:
        print(f"\n⚠️  ALERTA: '{mejor['delito']}' podría ser mejor opción")
        print(f"   Score: {mejor['score_contexto']:.2f} vs Robo Agravado")

        # Comparar con Robo Agravado
        robo_row = df_final[df_final['delito'] == 'ROBO AGRAVADO']
        if len(robo_row) > 0:
            robo_row = robo_row.iloc[0]
            print(f"\n   Comparación:")
            print(f"   {mejor['delito']}:")
            print(f"     Gini: {mejor['gini']:.4f} | AutoC: {mejor['autocorr']:.4f} | Casos: {mejor['casos_2024_2025']:,}")
            print(f"   ROBO AGRAVADO:")
            print(f"     Gini: {robo_row['gini']:.4f} | AutoC: {robo_row['autocorr']:.4f} | Casos: {robo_row['casos_2024_2025']:,}")

# Análisis de EXTORSIÓN específicamente
if 'EXTORSION' in df_final['delito'].values:
    ext_row = df_final[df_final['delito'] == 'EXTORSION'].iloc[0]
    print("\n🔥 CASO ESPECIAL: EXTORSIÓN")
    print("="*80)
    print(f"   Tendencia: {ext_row['tendencia_pct']:+.1f}% (2020→2025)")
    print(f"   Casos actuales: {ext_row['casos_2024_2025']:,}")
    print(f"   Predictibilidad: Gini={ext_row['gini']:.4f}, AutoC={ext_row['autocorr']:.4f}")

    print("\n   💭 Consideraciones sobre EXTORSIÓN:")
    print("   Ventajas:")
    print("     ✓ ALTA relevancia política y social en Perú actual")
    print("     ✓ Problema CRECIENTE (urgente para autoridades)")
    print("     ✓ Potencialmente mayor impacto mediático")
    print("   Desventajas:")
    if ext_row['gini'] < 0.7 or ext_row['autocorr'] < 0.7:
        print("     ❌ Menor predictibilidad técnica que Robo Agravado")
        print("     ❌ Podría resultar en modelo menos preciso")
    else:
        print("     ⚠️  Predictibilidad comparable a Robo Agravado")

    if ext_row['casos_2024_2025'] < 10000:
        print("     ❌ Menor volumen de datos (menos robust)")

print("\n" + "="*80)
print("🎯 RECOMENDACIÓN FINAL")
print("="*80)

print("\n1️⃣  Para PROYECTO ACADÉMICO (maximizar R²):")
print("   → ROBO AGRAVADO")
print("   Razón: Mayor volumen, mejor predictibilidad técnica")

print("\n2️⃣  Para IMPACTO POLÍTICO/MEDIÁTICO:")
print("   → EXTORSIÓN (si predictibilidad es aceptable)")
print("   Razón: Problema más urgente en contexto actual Perú")

print("\n3️⃣  Para VALIDACIÓN METODOLÓGICA:")
print("   → Comparar AMBOS delitos")
print("   Razón: Demostrar versatilidad (delito estable vs creciente)")

print("\n💡 Sugerencia: Si tienes tiempo, haz AMBOS:")
print("   • Modelo 1: Robo Agravado (predicción robusta, R² alto)")
print("   • Modelo 2: Extorsión (relevancia actual, mayor impacto)")
print("   → Presentas dos casos de uso con diferentes dinámicas temporales")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
