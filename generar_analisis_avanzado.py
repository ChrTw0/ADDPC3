"""
Análisis Avanzado - Gráficos Complementarios para Paper PC3
===========================================================
Genera visualizaciones de análisis temporal y espacial avanzado.

Gráficos adicionales:
1. Serie Temporal de Crímenes (2020-2025)
2. Top 10 Hotspots Críticos - Mapa de Calor
3. Curva ROC - Clasificación Binaria (Hotspot)
4. Precision-Recall Curve
5. Comparación Train vs Test Performance
6. Evolución Mensual de Delitos
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import joblib
import os
import mysql.connector
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración estética
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directorios
FIGURES_DIR = 'figures'
MODELS_DIR = 'models/best_models'
os.makedirs(FIGURES_DIR, exist_ok=True)


def conectar_db():
    """Conecta a la base de datos MySQL."""
    try:
        # Forzar TCP/IP en lugar de named pipe (Windows fix)
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', '127.0.0.1'),
            port=int(os.getenv('MYSQL_PORT', 3306)),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', '1234'),
            database=os.getenv('MYSQL_DB', 'denuncias_peru'),
            use_pure=True  # Fuerza TCP/IP en Windows
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Conexión DB: {e}")
        print(f"[INFO] Asegúrate de que MySQL esté corriendo en puerto 3306")
        print(f"[INFO] Comando: net start MySQL80 (o similar)")
        return None


def grafico_9_serie_temporal():
    """
    Gráfico 9: Serie Temporal de Crímenes 2020-2025
    Evolución mensual de HURTO y EXTORSIÓN.
    """
    conn = conectar_db()
    if not conn:
        print("[SKIP] Figura 9: No hay conexión a DB")
        return

    query = """
    SELECT
        DATE_FORMAT(fecha_hora_hecho, '%Y-%m') as mes,
        modalidad_hecho as tipo_delito,
        COUNT(*) as cantidad
    FROM denuncias
    WHERE departamento_hecho = 'LIMA'
        AND modalidad_hecho IN ('HURTO', 'EXTORSION')
        AND fecha_hora_hecho >= '2020-01-01'
        AND fecha_hora_hecho <= '2025-01-20'
    GROUP BY mes, modalidad_hecho
    ORDER BY mes, modalidad_hecho
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Convertir a datetime
    df['mes'] = pd.to_datetime(df['mes'] + '-01')

    # Pivot para graficar
    df_pivot = df.pivot(index='mes', columns='tipo_delito', values='cantidad')

    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot con área sombreada
    ax.fill_between(df_pivot.index, df_pivot['HURTO'], alpha=0.3, label='HURTO')
    ax.plot(df_pivot.index, df_pivot['HURTO'], linewidth=2, marker='o',
            markersize=4, label='HURTO (línea)')

    # Escala secundaria para EXTORSIÓN
    ax2 = ax.twinx()
    ax2.fill_between(df_pivot.index, df_pivot['EXTORSION'], alpha=0.3,
                     color='orange', label='EXTORSIÓN')
    ax2.plot(df_pivot.index, df_pivot['EXTORSION'], linewidth=2, marker='s',
            markersize=4, color='orangered', label='EXTORSIÓN (línea)')

    # Línea vertical COVID
    ax.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--',
               alpha=0.5, linewidth=2, label='Inicio COVID-19')

    ax.set_title('Evolución Temporal de HURTO y EXTORSIÓN en Lima (2020-2025)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Denuncias de HURTO', fontsize=12, color='blue')
    ax2.set_ylabel('Denuncias de EXTORSIÓN', fontsize=12, color='orange')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.grid(alpha=0.3)

    # Combinar leyendas
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig9_serie_temporal_delitos.png', dpi=300, bbox_inches='tight')
    print("[OK] Figura 9 guardada: fig9_serie_temporal_delitos.png")
    plt.close()


def grafico_10_top_hotspots_mapa():
    """
    Gráfico 10: Top 10 Hotspots Críticos
    Scatterplot geográfico de las zonas más peligrosas.
    """
    conn = conectar_db()
    if not conn:
        print("[SKIP] Figura 10: No hay conexión a DB")
        return

    query = """
    SELECT
        FLOOR(lat_hecho / 0.005) * 0.005 as lat_grid,
        FLOOR(long_hecho / 0.005) * 0.005 as long_grid,
        COUNT(*) as crime_count
    FROM denuncias
    WHERE departamento_hecho = 'LIMA'
        AND modalidad_hecho = 'HURTO'
        AND fecha_hora_hecho >= '2024-01-01'
        AND fecha_hora_hecho <= '2025-01-20'
        AND lat_hecho IS NOT NULL
        AND long_hecho IS NOT NULL
    GROUP BY lat_grid, long_grid
    ORDER BY crime_count DESC
    LIMIT 50
    """

    df = pd.read_sql(query, conn)
    conn.close()

    fig, ax = plt.subplots(figsize=(12, 10))

    # Scatter con tamaño proporcional
    scatter = ax.scatter(df['long_grid'], df['lat_grid'],
                        s=df['crime_count']*2,
                        c=df['crime_count'],
                        cmap='YlOrRd',
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5)

    # Top 10 con anotaciones
    top10 = df.nlargest(10, 'crime_count')
    for idx, row in top10.iterrows():
        ax.annotate(f"{int(row['crime_count'])}",
                   xy=(row['long_grid'], row['lat_grid']),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=8,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

    plt.colorbar(scatter, ax=ax, label='Número de Crímenes (2024-2025)')

    ax.set_title('Top 50 Hotspots de HURTO en Lima (2024-2025)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig10_top_hotspots_mapa.png', dpi=300, bbox_inches='tight')
    print("[OK] Figura 10 guardada: fig10_top_hotspots_mapa.png")
    plt.close()


def grafico_11_curva_roc():
    """
    Gráfico 11: Curva ROC - Clasificación Binaria
    Para Hotspot Crítico (mejor modelo).
    """
    try:
        from utils.data_preparation import preparar_datos_completo

        datos = preparar_datos_completo('hurto')
        if datos is None:
            print("[SKIP] Figura 11: No se pudo cargar datos")
            return

        X_test = datos['X_test']
        y_test = datos['targets']['hotspot_critico']['test']

        # Cargar modelo
        modelo = joblib.load(f'{MODELS_DIR}/best_hotspot_critico_hurto.joblib')

        # Predicciones probabilísticas
        if hasattr(modelo, 'predict_proba'):
            y_proba = modelo.predict_proba(X_test)[:, 1]
        elif hasattr(modelo, 'decision_function'):
            y_proba = modelo.decision_function(X_test)
        else:
            print("[SKIP] Figura 11: Modelo no soporta probabilidades")
            return

        # Calcular ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random classifier (AUC = 0.50)')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve - Gradient Boosting (Hotspot Crítico HURTO)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/fig11_curva_roc.png', dpi=300, bbox_inches='tight')
        print("[OK] Figura 11 guardada: fig11_curva_roc.png")
        plt.close()

    except Exception as e:
        print(f"[SKIP] Figura 11: {e}")


def grafico_12_precision_recall():
    """
    Gráfico 12: Precision-Recall Curve
    Para Hotspot Crítico (mejor modelo).
    """
    try:
        from utils.data_preparation import preparar_datos_completo

        datos = preparar_datos_completo('hurto')
        if datos is None:
            print("[SKIP] Figura 12: No se pudo cargar datos")
            return

        X_test = datos['X_test']
        y_test = datos['targets']['hotspot_critico']['test']

        # Cargar modelo
        modelo = joblib.load(f'{MODELS_DIR}/best_hotspot_critico_hurto.joblib')

        # Predicciones probabilísticas
        if hasattr(modelo, 'predict_proba'):
            y_proba = modelo.predict_proba(X_test)[:, 1]
        elif hasattr(modelo, 'decision_function'):
            y_proba = modelo.decision_function(X_test)
        else:
            print("[SKIP] Figura 12: Modelo no soporta probabilidades")
            return

        # Calcular Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(recall, precision, color='blue', lw=2,
               label=f'PR curve (AP = {avg_precision:.4f})')
        ax.axhline(y=y_test.mean(), color='red', linestyle='--',
                  label=f'Baseline (prevalencia = {y_test.mean():.4f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve - Gradient Boosting (Hotspot Crítico HURTO)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=12)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/fig12_precision_recall.png', dpi=300, bbox_inches='tight')
        print("[OK] Figura 12 guardada: fig12_precision_recall.png")
        plt.close()

    except Exception as e:
        print(f"[SKIP] Figura 12: {e}")


def grafico_13_evolucion_mensual():
    """
    Gráfico 13: Evolución Mensual de Delitos por Tipo
    Comparación año a año.
    """
    conn = conectar_db()
    if not conn:
        print("[SKIP] Figura 13: No hay conexión a DB")
        return

    query = """
    SELECT
        YEAR(fecha_hora_hecho) as año,
        MONTH(fecha_hora_hecho) as mes,
        modalidad_hecho as tipo_delito,
        COUNT(*) as cantidad
    FROM denuncias
    WHERE departamento_hecho = 'LIMA'
        AND modalidad_hecho IN ('HURTO', 'EXTORSION')
        AND YEAR(fecha_hora_hecho) >= 2020
    GROUP BY año, mes, modalidad_hecho
    ORDER BY modalidad_hecho, año, mes
    """

    df = pd.read_sql(query, conn)
    conn.close()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for idx, delito in enumerate(['HURTO', 'EXTORSION']):
        df_delito = df[df['tipo_delito'] == delito]

        # Pivot año x mes
        pivot = df_delito.pivot_table(
            values='cantidad',
            index='mes',
            columns='año',
            aggfunc='sum'
        )

        ax = axes[idx]
        pivot.plot(ax=ax, marker='o', linewidth=2)

        ax.set_title(f'{delito} - Patrón Estacional por Año',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Mes', fontsize=12)
        ax.set_ylabel('Número de Denuncias', fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                           'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
        ax.legend(title='Año', loc='best')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig13_evolucion_mensual.png', dpi=300, bbox_inches='tight')
    print("[OK] Figura 13 guardada: fig13_evolucion_mensual.png")
    plt.close()


def main():
    """Genera todos los gráficos avanzados."""
    print("\n" + "="*80)
    print("GENERACIÓN DE ANÁLISIS AVANZADO - GRÁFICOS COMPLEMENTARIOS")
    print("="*80 + "\n")

    print("[1/5] Generando Figura 9: Serie Temporal...")
    grafico_9_serie_temporal()

    print("[2/5] Generando Figura 10: Top Hotspots Mapa...")
    grafico_10_top_hotspots_mapa()

    print("[3/5] Generando Figura 11: Curva ROC...")
    grafico_11_curva_roc()

    print("[4/5] Generando Figura 12: Precision-Recall...")
    grafico_12_precision_recall()

    print("[5/5] Generando Figura 13: Evolución Mensual...")
    grafico_13_evolucion_mensual()

    print("\n" + "="*80)
    print("✅ ANÁLISIS AVANZADO COMPLETADO")
    print(f"   Figuras guardadas en: {FIGURES_DIR}/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
