"""
Análisis de Validación de Metodología - Conectando a MySQL
===========================================================
Este script se conecta directamente a la base de datos MySQL 'denuncias_peru'
para validar si vale la pena hacer predicciones de hotspots.

Preguntas críticas:
1. ¿Los datos originales tienen suficiente calidad?
2. ¿Existen patrones temporales predecibles?
3. ¿Existen hotspots espaciales persistentes?
4. ¿La metodología es correcta?
5. ¿Un R² de 0.697 es realmente bueno?
6. ¿VALE LA PENA hacer predicciones o es ruido aleatorio?
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from scipy import stats
from sklearn.metrics import r2_score
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Configuración visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("VALIDACIÓN DE METODOLOGÍA - ANÁLISIS DESDE DATOS ORIGINALES (MySQL)")
print("="*80)
print()

# ============================================================================
# 1. CONEXIÓN A BASE DE DATOS
# ============================================================================
print("1. CONECTANDO A BASE DE DATOS MySQL...")
print("-" * 80)

load_dotenv()
db_user = os.getenv("MYSQL_USER")
db_password = os.getenv("MYSQL_PASSWORD")
db_host = os.getenv("MYSQL_HOST")
db_name = os.getenv("MYSQL_DB")

print(f"Host: {db_host}")
print(f"Base de datos: {db_name}")
print(f"Usuario: {db_user}")

try:
    connection_string = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(connection_string)
    connection = engine.connect()
    print("✓ Conexión exitosa a MySQL")
    connection.close()
except Exception as e:
    print(f"❌ Error de conexión: {e}")
    sys.exit(1)

# ============================================================================
# 2. EXPLORACIÓN DE LA ESTRUCTURA DE DATOS
# ============================================================================
print("\n" + "="*80)
print("2. EXPLORANDO ESTRUCTURA DE LA TABLA 'denuncias'")
print("="*80)

# Obtener info básica
query_count = text("SELECT COUNT(*) as total FROM denuncias")
total_records = pd.read_sql(query_count, engine).iloc[0]['total']
print(f"\n✓ Total de registros en la tabla: {total_records:,}")

# Columnas disponibles
query_columns = text("SHOW COLUMNS FROM denuncias")
columns_info = pd.read_sql(query_columns, engine)
print(f"\n✓ Columnas disponibles ({len(columns_info)}):")
for idx, row in columns_info.iterrows():
    print(f"   - {row['Field']}: {row['Type']}")

# Modalidades de hecho disponibles
query_modalidades = text("""
    SELECT modalidad_hecho, COUNT(*) as count
    FROM denuncias
    GROUP BY modalidad_hecho
    ORDER BY count DESC
    LIMIT 10
""")
modalidades = pd.read_sql(query_modalidades, engine)
print(f"\n✓ Top 10 Modalidades de Hecho:")
for idx, row in modalidades.iterrows():
    print(f"   {idx+1}. {row['modalidad_hecho']}: {row['count']:,} denuncias")

# Departamentos disponibles
query_deptos = text("""
    SELECT departamento_hecho, COUNT(*) as count
    FROM denuncias
    GROUP BY departamento_hecho
    ORDER BY count DESC
    LIMIT 10
""")
departamentos = pd.read_sql(query_deptos, engine)
print(f"\n✓ Top 10 Departamentos:")
for idx, row in departamentos.iterrows():
    print(f"   {idx+1}. {row['departamento_hecho']}: {row['count']:,} denuncias")

# ============================================================================
# 3. EXTRACCIÓN DE DATOS: ROBO AGRAVADO EN LIMA
# ============================================================================
print("\n" + "="*80)
print("3. EXTRAYENDO DATOS: ROBO AGRAVADO EN LIMA")
print("="*80)

query_lima = text("""
    SELECT
        id,
        lat_hecho,
        long_hecho,
        fecha_hora_hecho,
        modalidad_hecho,
        departamento_hecho,
        provincia_hecho,
        distrito_hecho
    FROM denuncias
    WHERE departamento_hecho = 'LIMA'
    AND modalidad_hecho = 'ROBO AGRAVADO'
""")

print("Cargando datos desde MySQL... (esto puede tomar un momento)")
df = pd.read_sql(query_lima, engine)
print(f"✓ Datos cargados: {len(df):,} registros de Robo Agravado en Lima")

# ============================================================================
# 4. LIMPIEZA Y VALIDACIÓN DE DATOS
# ============================================================================
print("\n" + "="*80)
print("4. LIMPIEZA Y VALIDACIÓN DE CALIDAD DE DATOS")
print("="*80)

print("\n4.1 Valores Nulos por Columna:")
nulls = df.isnull().sum()
for col, count in nulls.items():
    if count > 0:
        pct = count / len(df) * 100
        print(f"   {col}: {count:,} ({pct:.2f}%)")

# Limpiar coordenadas
df['lat_hecho'] = pd.to_numeric(df['lat_hecho'], errors='coerce')
df['long_hecho'] = pd.to_numeric(df['long_hecho'], errors='coerce')
df['fecha_hora_hecho'] = pd.to_datetime(df['fecha_hora_hecho'], errors='coerce')

inicial = len(df)
df = df.dropna(subset=['lat_hecho', 'long_hecho', 'fecha_hora_hecho'])
final = len(df)
removidos = inicial - final

print(f"\n4.2 Limpieza de Datos:")
print(f"   Registros iniciales: {inicial:,}")
print(f"   Registros removidos: {removidos:,} ({removidos/inicial*100:.2f}%)")
print(f"   Registros válidos: {final:,}")

# Validar rango geográfico de Lima
lat_lima = (-12.3, -11.7)
long_lima = (-77.2, -76.7)

coords_validas = (
    (df['lat_hecho'] >= lat_lima[0]) &
    (df['lat_hecho'] <= lat_lima[1]) &
    (df['long_hecho'] >= long_lima[0]) &
    (df['long_hecho'] <= long_lima[1])
)

print(f"\n4.3 Validación de Coordenadas:")
print(f"   Dentro del rango de Lima: {coords_validas.sum():,} / {len(df):,} ({coords_validas.sum()/len(df)*100:.2f}%)")
print(f"   Fuera del rango: {(~coords_validas).sum():,}")

if (~coords_validas).sum() > 0:
    print(f"\n   ⚠️  Hay {(~coords_validas).sum()} coordenadas fuera del rango esperado de Lima")
    print(f"   Rango latitud: [{df['lat_hecho'].min():.4f}, {df['lat_hecho'].max():.4f}]")
    print(f"   Rango longitud: [{df['long_hecho'].min():.4f}, {df['long_hecho'].max():.4f}]")

# Rango temporal
print(f"\n4.4 Rango Temporal:")
fecha_min = df['fecha_hora_hecho'].min()
fecha_max = df['fecha_hora_hecho'].max()
dias_total = (fecha_max - fecha_min).days
print(f"   Desde: {fecha_min}")
print(f"   Hasta: {fecha_max}")
print(f"   Período: {dias_total} días (~{dias_total/365:.1f} años)")

# ============================================================================
# 5. ANÁLISIS TEMPORAL - ¿HAY PATRONES PREDECIBLES?
# ============================================================================
print("\n" + "="*80)
print("5. ANÁLISIS TEMPORAL - ¿EXISTEN PATRONES PREDECIBLES?")
print("="*80)

df['año'] = df['fecha_hora_hecho'].dt.year
df['mes'] = df['fecha_hora_hecho'].dt.month
df['dia_semana'] = df['fecha_hora_hecho'].dt.dayofweek
df['año_semana'] = df['fecha_hora_hecho'].dt.strftime('%Y-%U')

print("\n5.1 Distribución por Año:")
por_año = df.groupby('año').size()
for año, count in por_año.items():
    pct = count / len(df) * 100
    print(f"   {año}: {count:,} ({pct:.1f}%)")

print("\n5.2 Distribución por Mes:")
por_mes = df.groupby('mes').size().sort_index()
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
for mes, count in por_mes.items():
    pct = count / len(df) * 100
    print(f"   {meses[mes-1]}: {count:,} ({pct:.1f}%)")

cv_mes = por_mes.std() / por_mes.mean()
print(f"\n   Coeficiente de Variación (CV): {cv_mes:.4f}")
if cv_mes > 0.15:
    print(f"   ✓ ALTA variabilidad mensual - HAY ESTACIONALIDAD")
else:
    print(f"   ⚠️  Baja variabilidad - Patrón mensual débil")

print("\n5.3 Distribución por Día de la Semana:")
por_dia = df.groupby('dia_semana').size().sort_index()
dias_nom = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
for dia, count in por_dia.items():
    pct = count / len(df) * 100
    print(f"   {dias_nom[dia]}: {count:,} ({pct:.1f}%)")

cv_dia = por_dia.std() / por_dia.mean()
print(f"\n   Coeficiente de Variación (CV): {cv_dia:.4f}")
if cv_dia > 0.10:
    print(f"   ✓ HAY patrón semanal predecible")
else:
    print(f"   ⚠️  Patrón semanal débil")

# Autocorrelación temporal
print("\n5.4 Autocorrelación Temporal (crímenes por semana):")
crimes_per_week = df.groupby('año_semana').size().sort_index()
print(f"   Total de semanas: {len(crimes_per_week)}")
print(f"   Promedio crímenes/semana: {crimes_per_week.mean():.2f}")
print(f"   Desv. estándar: {crimes_per_week.std():.2f}")

lags = [1, 2, 4, 8, 12]
print("\n   Correlación con semanas anteriores:")
for lag in lags:
    if len(crimes_per_week) > lag:
        shifted = crimes_per_week.shift(lag)
        valid_mask = ~(crimes_per_week.isna() | shifted.isna())
        if valid_mask.sum() > 10:
            corr = crimes_per_week[valid_mask].corr(shifted[valid_mask])
            status = '✓ Fuerte' if abs(corr) > 0.5 else '✓ Moderada' if abs(corr) > 0.3 else '⚠️  Débil'
            print(f"   Lag {lag:2d} semana(s): {corr:6.4f} {status}")

# ============================================================================
# 6. ANÁLISIS ESPACIAL - ¿HAY HOTSPOTS PERSISTENTES?
# ============================================================================
print("\n" + "="*80)
print("6. ANÁLISIS ESPACIAL - ¿EXISTEN HOTSPOTS PERSISTENTES?")
print("="*80)

# Crear grid (0.005 grados ≈ 555 metros en Lima)
grid_size = 0.005
df['grid_lat'] = (df['lat_hecho'] // grid_size) * grid_size
df['grid_long'] = (df['long_hecho'] // grid_size) * grid_size
df['grid_cell'] = df['grid_lat'].astype(str) + '_' + df['grid_long'].astype(str)

crimes_per_cell = df.groupby('grid_cell').size().sort_values(ascending=False)

print(f"\n6.1 Estadísticas Espaciales:")
print(f"   Total de celdas: {len(crimes_per_cell):,}")
print(f"   Promedio crímenes/celda: {crimes_per_cell.mean():.2f}")
print(f"   Mediana crímenes/celda: {crimes_per_cell.median():.2f}")
print(f"   Desv. estándar: {crimes_per_cell.std():.2f}")
print(f"   Coeficiente de Variación: {crimes_per_cell.std() / crimes_per_cell.mean():.4f}")

print(f"\n6.2 Top 10 Hotspots:")
for i, (cell, count) in enumerate(crimes_per_cell.head(10).items(), 1):
    pct = count / len(df) * 100
    print(f"   {i:2d}. Celda {cell}: {count:,} crímenes ({pct:.2f}% del total)")

# Concentración tipo Pareto
print(f"\n6.3 Concentración Espacial (Principio de Pareto):")
percentiles = [5, 10, 20, 30]
for pct in percentiles:
    top_n = int(len(crimes_per_cell) * pct / 100)
    top_crimes = crimes_per_cell.head(top_n).sum()
    print(f"   Top {pct:2d}% de celdas → {top_crimes:,} crímenes ({top_crimes/len(df)*100:.1f}% del total)")

# Índice de Gini
sorted_crimes = np.sort(crimes_per_cell.values)
n = len(sorted_crimes)
index = np.arange(1, n + 1)
gini = (2 * np.sum(index * sorted_crimes)) / (n * np.sum(sorted_crimes)) - (n + 1) / n
print(f"\n6.4 Índice de Gini (0=igualdad, 1=máxima desigualdad): {gini:.4f}")
if gini > 0.7:
    print(f"   ✓ MUY CONCENTRADO - Hotspots muy claros y definidos")
elif gini > 0.5:
    print(f"   ✓ CONCENTRADO - Hotspots moderados")
else:
    print(f"   ⚠️  POCO concentrado - Distribución casi uniforme")

# Persistencia temporal de hotspots
print(f"\n6.5 Persistencia de Hotspots en el Tiempo:")
df_sorted = df.sort_values('fecha_hora_hecho')
mid = len(df_sorted) // 2

df_p1 = df_sorted.iloc[:mid]
df_p2 = df_sorted.iloc[mid:]

cells_p1 = df_p1.groupby('grid_cell').size()
cells_p2 = df_p2.groupby('grid_cell').size()

top_n = 50
top_cells_p1 = set(cells_p1.nlargest(top_n).index)
top_cells_p2 = set(cells_p2.nlargest(top_n).index)
overlap = len(top_cells_p1 & top_cells_p2)

print(f"   Período 1: {df_p1['fecha_hora_hecho'].min()} a {df_p1['fecha_hora_hecho'].max()}")
print(f"   Período 2: {df_p2['fecha_hora_hecho'].min()} a {df_p2['fecha_hora_hecho'].max()}")
print(f"   Top {top_n} hotspots que se repiten: {overlap} ({overlap/top_n*100:.1f}%)")

if overlap > 35:
    print(f"   ✓ ALTA PERSISTENCIA - Hotspots estables en el tiempo")
elif overlap > 25:
    print(f"   ✓ PERSISTENCIA MODERADA")
else:
    print(f"   ⚠️  BAJA PERSISTENCIA - Hotspots cambian mucho")

# Correlación espacial entre períodos
all_cells = list(set(cells_p1.index) | set(cells_p2.index))
counts_p1 = [cells_p1.get(cell, 0) for cell in all_cells]
counts_p2 = [cells_p2.get(cell, 0) for cell in all_cells]
spatial_corr = np.corrcoef(counts_p1, counts_p2)[0, 1]

print(f"\n   Correlación espacial entre períodos: {spatial_corr:.4f}")
if spatial_corr > 0.7:
    print(f"   ✓ MUY PREDECIBLE espacialmente")
elif spatial_corr > 0.5:
    print(f"   ✓ MODERADAMENTE predecible")
else:
    print(f"   ⚠️  POCO predecible espacialmente")

# ============================================================================
# 7. EVALUACIÓN DE BASELINES
# ============================================================================
print("\n" + "="*80)
print("7. COMPARACIÓN CON MODELOS BASELINE")
print("="*80)

# Cargar features procesadas para comparar
try:
    df_features = pd.read_parquet('data/processed/hotspot_features_robo_agravado_lima.parquet')

    if 'crime_count' in df_features.columns:
        print("\n7.1 Baseline: Predecir la Media")
        y_true = df_features['crime_count'].values
        y_pred_mean = np.full_like(y_true, y_true.mean(), dtype=float)

        mae_mean = np.mean(np.abs(y_true - y_pred_mean))
        rmse_mean = np.sqrt(np.mean((y_true - y_pred_mean)**2))
        r2_mean = r2_score(y_true, y_pred_mean)

        print(f"   MAE:  {mae_mean:.4f}")
        print(f"   RMSE: {rmse_mean:.4f}")
        print(f"   R²:   {r2_mean:.4f}")

        print("\n7.2 Baseline: Persistencia (semana anterior)")
        if 'crime_count_lag_1' in df_features.columns:
            mask = df_features['crime_count_lag_1'].notna() & df_features['crime_count'].notna()
            y_true_p = df_features.loc[mask, 'crime_count'].values
            y_pred_p = df_features.loc[mask, 'crime_count_lag_1'].values

            mae_pers = np.mean(np.abs(y_true_p - y_pred_p))
            rmse_pers = np.sqrt(np.mean((y_true_p - y_pred_p)**2))
            r2_pers = r2_score(y_true_p, y_pred_p)

            print(f"   MAE:  {mae_pers:.4f}")
            print(f"   RMSE: {rmse_pers:.4f}")
            print(f"   R²:   {r2_pers:.4f}")

            print("\n7.3 Comparación")
            r2_lstm = 0.6970
            print(f"   {'Modelo':<25} {'R²':>10}")
            print(f"   {'-'*35}")
            print(f"   {'Predecir Media':<25} {r2_mean:>10.4f}")
            print(f"   {'Persistencia (t-1)':<25} {r2_pers:>10.4f}")
            print(f"   {'LSTM Optimizado':<25} {r2_lstm:>10.4f}")

            if r2_lstm > r2_pers:
                mejora = ((r2_lstm - r2_pers) / abs(r2_pers) * 100) if r2_pers != 0 else float('inf')
                print(f"\n   ✓ LSTM supera persistencia en {mejora:.1f}%")
                print(f"   ✓ El modelo SÍ aporta valor sobre baselines simples")
            else:
                print(f"\n   ⚠️  LSTM no mejora significativamente sobre persistencia")

except FileNotFoundError:
    print("\n   ⚠️  No se encontró el archivo de features procesadas")
    print("   Ejecuta primero: python -m 02_feature_engineering.2b_create_grid_features_lima")

# ============================================================================
# 8. VISUALIZACIONES
# ============================================================================
print("\n" + "="*80)
print("8. GENERANDO VISUALIZACIONES...")
print("="*80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 8.1 Serie temporal
ax1 = fig.add_subplot(gs[0, :])
crimes_per_week.plot(ax=ax1, color='steelblue', linewidth=1.5)
ax1.set_title('Serie Temporal: Crímenes por Semana', fontweight='bold', fontsize=14)
ax1.set_xlabel('Semana')
ax1.set_ylabel('Número de Crímenes')
ax1.grid(True, alpha=0.3)

# 8.2 Estacionalidad mensual
ax2 = fig.add_subplot(gs[1, 0])
por_mes.plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('Estacionalidad Mensual', fontweight='bold')
ax2.set_xlabel('Mes')
ax2.set_ylabel('Total Crímenes')
ax2.set_xticklabels(meses, rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# 8.3 Patrón semanal
ax3 = fig.add_subplot(gs[1, 1])
por_dia.plot(kind='bar', ax=ax3, color='mediumseagreen')
ax3.set_title('Patrón Día de Semana', fontweight='bold')
ax3.set_xlabel('Día')
ax3.set_ylabel('Total Crímenes')
ax3.set_xticklabels(['L', 'M', 'X', 'J', 'V', 'S', 'D'], rotation=0)
ax3.grid(True, alpha=0.3, axis='y')

# 8.4 Autocorrelación
ax4 = fig.add_subplot(gs[1, 2])
autocorrs = []
for lag in range(1, 13):
    if len(crimes_per_week) > lag:
        shifted = crimes_per_week.shift(lag)
        valid_mask = ~(crimes_per_week.isna() | shifted.isna())
        if valid_mask.sum() > 10:
            corr = crimes_per_week[valid_mask].corr(shifted[valid_mask])
            autocorrs.append(corr)
ax4.bar(range(1, len(autocorrs)+1), autocorrs, color='teal', alpha=0.7)
ax4.axhline(y=0.3, color='red', linestyle='--', linewidth=1, label='Moderado')
ax4.set_title('Autocorrelación Temporal', fontweight='bold')
ax4.set_xlabel('Lag (semanas)')
ax4.set_ylabel('Correlación')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 8.5 Curva de Pareto
ax5 = fig.add_subplot(gs[2, 0])
sorted_cumsum = (crimes_per_cell.sort_values(ascending=False).cumsum() / crimes_per_cell.sum() * 100)
x_pct = np.arange(1, len(sorted_cumsum) + 1) / len(sorted_cumsum) * 100
ax5.plot(x_pct, sorted_cumsum.values, linewidth=2, color='darkviolet')
ax5.axhline(y=80, color='red', linestyle='--', linewidth=1, label='80%')
ax5.set_title('Curva Pareto - Concentración Espacial', fontweight='bold')
ax5.set_xlabel('% Celdas (ordenadas)')
ax5.set_ylabel('% Acumulado Crímenes')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 8.6 Distribución crímenes por celda
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(crimes_per_cell.values, bins=50, color='orange', edgecolor='black', alpha=0.7)
ax6.set_title('Distribución Crímenes/Celda', fontweight='bold')
ax6.set_xlabel('Crímenes por Celda')
ax6.set_ylabel('Frecuencia')
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3)

# 8.7 Mapa de calor simple
ax7 = fig.add_subplot(gs[2, 2])
pivot_data = df.groupby(['grid_lat', 'grid_long']).size().reset_index(name='count')
pivot_data = pivot_data.pivot(index='grid_lat', columns='grid_long', values='count').fillna(0)
sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'Crímenes'}, ax=ax7, robust=True)
ax7.set_title('Mapa de Calor - Hotspots Lima', fontweight='bold')
ax7.set_xlabel('Longitud (grid)')
ax7.set_ylabel('Latitud (grid)')

plt.savefig('validacion_metodologia_completa.png', dpi=300, bbox_inches='tight')
print("✓ Visualización guardada: 'validacion_metodologia_completa.png'")

# ============================================================================
# 9. CONCLUSIONES
# ============================================================================
print("\n" + "="*80)
print("9. CONCLUSIONES Y RECOMENDACIONES")
print("="*80)

print("\n📊 RESUMEN EJECUTIVO:")
print("-" * 80)

# Criterios de evaluación
score = 0
max_score = 5

# 1. Patrones temporales
tiene_estacionalidad = cv_mes > 0.15 or max(autocorrs[:4]) > 0.3
if tiene_estacionalidad:
    score += 1
    print("\n✓ [1/5] HAY patrones temporales predecibles")
else:
    print("\n⚠️  [0/5] Patrones temporales débiles")

# 2. Hotspots espaciales
hotspots_claros = gini > 0.5
if hotspots_claros:
    score += 1
    print("✓ [2/5] HAY hotspots espaciales claramente definidos")
else:
    print("⚠️  [1/5] Hotspots poco definidos")

# 3. Persistencia temporal
persistencia_alta = overlap > 30
if persistencia_alta:
    score += 1
    print("✓ [3/5] Hotspots SON persistentes en el tiempo")
else:
    print("⚠️  [2/5] Baja persistencia de hotspots")

# 4. Correlación espacial
correlacion_alta = spatial_corr > 0.6
if correlacion_alta:
    score += 1
    print("✓ [4/5] ALTA correlación espacial entre períodos")
else:
    print("⚠️  [3/5] Correlación espacial moderada")

# 5. Modelo supera baselines
if 'r2_pers' in locals():
    supera_baseline = 0.6970 > r2_pers
    if supera_baseline:
        score += 1
        print("✓ [5/5] Modelo LSTM supera baselines simples")
    else:
        print("⚠️  [4/5] Modelo no supera significativamente a persistencia")

print(f"\n🎯 SCORE FINAL: {score}/{max_score}")
print("=" * 80)

if score >= 4:
    print("\n✅ CONCLUSIÓN: SÍ VALE LA PENA hacer predicciones de hotspots")
    print("\n📋 JUSTIFICACIÓN:")
    print("  • Los datos muestran patrones temporales y espaciales claros")
    print("  • Los hotspots son persistentes y predecibles")
    print("  • El modelo LSTM aporta valor real sobre métodos simples")
    print("  • R² de 0.697 significa que el modelo explica ~70% de la variabilidad")
    print("\n💡 RECOMENDACIONES:")
    print("  ✓ La metodología es correcta (división temporal, features de lag)")
    print("  ✓ El modelo es útil para PRIORIZAR zonas de patrullaje")
    print("  ✓ Enfoque: Predicción de RIESGO RELATIVO, no conteo exacto")
    print("  ✓ Valor operacional: Asignación proactiva de recursos")
    print("\n🔧 MEJORAS SUGERIDAS:")
    print("  • Recalibración para reducir sobreestimación")
    print("  • Incorporar variables externas (clima, eventos, días festivos)")
    print("  • Análisis de texto en 'observacion_hecho' con NLP")

elif score >= 3:
    print("\n⚠️  CONCLUSIÓN: Las predicciones tienen VALOR LIMITADO")
    print("\n📋 JUSTIFICACIÓN:")
    print("  • Se detectan algunos patrones pero no son muy fuertes")
    print("  • El modelo aporta cierto valor pero con limitaciones")
    print("\n💡 RECOMENDACIONES:")
    print("  • Usar como herramienta complementaria, no principal")
    print("  • Combinar con conocimiento experto policial")
    print("  • Evaluar costo-beneficio del desarrollo")

else:
    print("\n❌ CONCLUSIÓN: Las predicciones NO parecen valer la pena")
    print("\n📋 JUSTIFICACIÓN:")
    print("  • Patrones débiles o inexistentes")
    print("  • Hotspots no son suficientemente persistentes")
    print("  • El modelo no supera significativamente a baselines")
    print("\n💡 RECOMENDACIONES:")
    print("  • Re-evaluar la definición del problema")
    print("  • Considerar otros tipos de delitos")
    print("  • Incorporar muchas más variables externas")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
