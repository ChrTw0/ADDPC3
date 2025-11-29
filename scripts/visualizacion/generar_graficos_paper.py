"""
Generación de Gráficos para Paper PC3 - Clasificación de Hotspots
=================================================================
Genera visualizaciones profesionales para complementar el paper académico.

Gráficos generados:
1. Comparación F1-Score por Tipo de Clasificación
2. Rendimiento por Algoritmo y Delito
3. Matriz de Confusion - Mejor Modelo (GB Hotspot HURTO)
4. Distribución de Clases - Nivel de Riesgo
5. Comparación Binaria vs Multiclase
6. Importancia de Features (Random Forest)
7. Curvas de Aprendizaje - Mejor Modelo
8. Heatmap de F1-Scores (Algoritmo x Tipo)
"""

import sys
from pathlib import Path
# Agregar raíz del proyecto al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
import joblib
import os

# Configuración estética
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)
FIGSIZE_TALL = (10, 12)

# Directorios
RESULTS_DIR = 'results'
MODELS_DIR = 'models/best_models'
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def cargar_resultados():
    """Carga los resultados de clasificación."""
    df = pd.read_csv(f'{RESULTS_DIR}/resultados_clasificacion_completo.csv')

    # Mapeo de nombres legibles
    df['delito'] = df['delito'].str.upper()
    df['nombre_clasificacion'] = df['nombre_clasificacion'].str.title()

    return df


def grafico_1_comparacion_f1_por_tipo(df):
    """
    Gráfico 1: Comparación F1-Score por Tipo de Clasificación
    Boxplot mostrando distribución de F1 para cada tipo.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Preparar datos
    df_plot = df[['nombre_clasificacion', 'f1', 'delito']].copy()

    # Boxplot con violín
    sns.violinplot(data=df_plot, x='nombre_clasificacion', y='f1',
                   hue='delito', split=True, ax=ax)

    ax.set_title('Distribución de F1-Score por Tipo de Clasificación',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Tipo de Clasificación', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_ylim([0.8, 1.0])
    ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5,
               label='Umbral Producción (0.85)')
    ax.legend(title='Delito', loc='lower left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig1_comparacion_f1_tipos.png', dpi=300, bbox_inches='tight')
    print("[OK] Figura 1 guardada: fig1_comparacion_f1_tipos.png")
    plt.close()


def grafico_2_rendimiento_algoritmos(df):
    """
    Gráfico 2: Rendimiento Promedio por Algoritmo
    Barplot horizontal con F1 promedio de cada algoritmo.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calcular F1 promedio por modelo y delito
    df_grouped = df.groupby(['modelo', 'delito'])['f1'].mean().reset_index()
    df_pivot = df_grouped.pivot(index='modelo', columns='delito', values='f1')

    # Ordenar por promedio general
    df_pivot['Promedio'] = df_pivot.mean(axis=1)
    df_pivot = df_pivot.sort_values('Promedio', ascending=True)
    df_pivot = df_pivot.drop('Promedio', axis=1)

    # Barplot horizontal agrupado
    df_pivot.plot(kind='barh', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)

    ax.set_title('Rendimiento Promedio por Algoritmo (F1-Score)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('F1-Score Promedio', fontsize=12)
    ax.set_ylabel('Algoritmo', fontsize=12)
    ax.set_xlim([0.88, 0.98])
    ax.axvline(x=0.95, color='green', linestyle='--', alpha=0.5,
               label='Excelente (>0.95)')
    ax.legend(title='Delito', loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    # Anotaciones de valores
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig2_rendimiento_algoritmos.png', dpi=300, bbox_inches='tight')
    print("[OK] Figura 2 guardada: fig2_rendimiento_algoritmos.png")
    plt.close()


def grafico_3_matriz_confusion_mejor_modelo():
    """
    Gráfico 3: Matriz de Confusión del Mejor Modelo
    Gradient Boosting - Hotspot Crítico - HURTO
    """
    try:
        # Cargar mejor modelo
        from utils.data_preparation import preparar_datos_completo

        datos = preparar_datos_completo('hurto')
        if datos is None:
            print("[SKIP] Figura 3: No se pudo cargar datos para matriz de confusión")
            return

        X_test = datos['X_test']
        y_test = datos['targets']['hotspot_critico']['test']

        # Cargar modelo
        modelo = joblib.load(f'{MODELS_DIR}/best_hotspot_critico_hurto.joblib')
        y_pred = modelo.predict(X_test)

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=['Normal', 'Crítico'])
        disp.plot(ax=ax, cmap='Blues', values_format='d')

        ax.set_title('Matriz de Confusión - Gradient Boosting (Hotspot Crítico HURTO)\nF1 = 0.9956',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/fig3_matriz_confusion_mejor_modelo.png', dpi=300, bbox_inches='tight')
        print("[OK] Figura 3 guardada: fig3_matriz_confusion_mejor_modelo.png")
        plt.close()

    except Exception as e:
        print(f"[SKIP] Figura 3: {e}")


def grafico_4_distribucion_clases(df):
    """
    Gráfico 4: Distribución de Clases - Nivel de Riesgo
    Muestra el desbalance de clases en el problema multiclase.
    """
    # Distribución conocida de HURTO (del paper)
    distribucion_hurto = {
        'Bajo\n(0-2)': 587493,
        'Medio\n(3-5)': 89831,
        'Alto\n(6-10)': 20478,
        'Muy Alto\n(>10)': 11876
    }

    distribucion_extorsion = {
        'Bajo\n(0-2)': 89472,
        'Medio\n(3-5)': 14261,
        'Alto\n(6-10)': 2224,
        'Muy Alto\n(>10)': 1950
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # HURTO
    ax1 = axes[0]
    clases = list(distribucion_hurto.keys())
    valores = list(distribucion_hurto.values())
    porcentajes = [v/sum(valores)*100 for v in valores]

    bars1 = ax1.bar(clases, valores, color=['green', 'yellow', 'orange', 'red'],
                    edgecolor='black', linewidth=1.5)
    ax1.set_title('HURTO - Distribución de Clases (Nivel de Riesgo)',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Número de Observaciones', fontsize=12)
    ax1.set_xlabel('Nivel de Riesgo', fontsize=12)

    # Anotaciones
    for bar, pct in zip(bars1, porcentajes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # EXTORSIÓN
    ax2 = axes[1]
    valores2 = list(distribucion_extorsion.values())
    porcentajes2 = [v/sum(valores2)*100 for v in valores2]

    bars2 = ax2.bar(clases, valores2, color=['green', 'yellow', 'orange', 'red'],
                    edgecolor='black', linewidth=1.5)
    ax2.set_title('EXTORSIÓN - Distribución de Clases (Nivel de Riesgo)',
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Número de Observaciones', fontsize=12)
    ax2.set_xlabel('Nivel de Riesgo', fontsize=12)

    # Anotaciones
    for bar, pct in zip(bars2, porcentajes2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig4_distribucion_clases.png', dpi=300, bbox_inches='tight')
    print("[OK] Figura 4 guardada: fig4_distribucion_clases.png")
    plt.close()


def grafico_5_binaria_vs_multiclase(df):
    """
    Gráfico 5: Comparación Binaria vs Multiclase
    Scatterplot F1 vs número de clases.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Mapear número de clases
    clases_map = {
        'Hotspot Crítico': 2,
        'Tendencia De Riesgo': 3,
        'Nivel De Riesgo': 4
    }

    df['num_clases'] = df['nombre_clasificacion'].map(clases_map)

    # Scatterplot
    for delito in df['delito'].unique():
        df_delito = df[df['delito'] == delito]
        ax.scatter(df_delito['num_clases'], df_delito['f1'],
                  label=delito, alpha=0.6, s=100, edgecolor='black')

    # Línea de tendencia
    z = np.polyfit(df['num_clases'], df['f1'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(2, 4, 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2,
            label=f'Tendencia: F1 = {z[0]:.3f}×clases + {z[1]:.3f}')

    ax.set_title('Impacto del Número de Clases en F1-Score',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Número de Clases', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_xticks([2, 3, 4])
    ax.set_xticklabels(['2\n(Binaria)', '3\n(Tendencia)', '4\n(Nivel Riesgo)'])
    ax.set_ylim([0.83, 1.0])
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig5_binaria_vs_multiclase.png', dpi=300, bbox_inches='tight')
    print("[OK] Figura 5 guardada: fig5_binaria_vs_multiclase.png")
    plt.close()


def grafico_6_importancia_features():
    """
    Gráfico 6: Importancia de Features
    Random Forest - Hotspot Crítico - HURTO
    """
    try:
        # Cargar modelo Random Forest
        modelo = joblib.load(f'{MODELS_DIR}/best_hotspot_critico_hurto.joblib')

        # Nombres de features
        feature_names = [
            'crime_count_lag_1',
            'crime_count_lag_2',
            'crime_count_lag_3',
            'crime_count_lag_4',
            'mes',
            'dia_semana'
        ]

        # Si el modelo tiene feature_importances_
        if hasattr(modelo, 'feature_importances_'):
            importancias = modelo.feature_importances_

            fig, ax = plt.subplots(figsize=(10, 6))

            # Ordenar por importancia
            indices = np.argsort(importancias)[::-1]

            ax.barh(range(len(importancias)), importancias[indices],
                   color='steelblue', edgecolor='black', linewidth=1)
            ax.set_yticks(range(len(importancias)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importancia', fontsize=12)
            ax.set_title('Importancia de Features - Gradient Boosting (Hotspot HURTO)',
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            # Anotaciones
            for i, v in enumerate(importancias[indices]):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

            plt.tight_layout()
            plt.savefig(f'{FIGURES_DIR}/fig6_importancia_features.png', dpi=300, bbox_inches='tight')
            print("[OK] Figura 6 guardada: fig6_importancia_features.png")
            plt.close()
        else:
            print("[SKIP] Figura 6: Modelo no tiene feature_importances_")

    except Exception as e:
        print(f"[SKIP] Figura 6: {e}")


def grafico_7_heatmap_f1(df):
    """
    Gráfico 7: Heatmap de F1-Scores
    Algoritmo x Tipo de Clasificación
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for idx, delito in enumerate(['HURTO', 'EXTORSION']):
        df_delito = df[df['delito'] == delito]

        # Pivot para heatmap
        pivot = df_delito.pivot_table(
            values='f1',
            index='modelo',
            columns='nombre_clasificacion'
        )

        # Heatmap
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu',
                   ax=axes[idx], cbar_kws={'label': 'F1-Score'},
                   vmin=0.83, vmax=1.0, linewidths=0.5, linecolor='gray')

        axes[idx].set_title(f'{delito} - Heatmap de F1-Scores',
                          fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Tipo de Clasificación', fontsize=12)
        axes[idx].set_ylabel('Algoritmo', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig7_heatmap_f1_scores.png', dpi=300, bbox_inches='tight')
    print("[OK] Figura 7 guardada: fig7_heatmap_f1_scores.png")
    plt.close()


def grafico_8_metricas_comparadas(df):
    """
    Gráfico 8: Comparación Accuracy vs Precision vs Recall vs F1
    Radar chart para los mejores 5 modelos.
    """
    from math import pi

    # Top 5 modelos
    top5 = df.nlargest(5, 'f1')

    categorias = ['Accuracy', 'Precision', 'Recall', 'F1']
    num_categorias = len(categorias)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angulos = [n / float(num_categorias) * 2 * pi for n in range(num_categorias)]
    angulos += angulos[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(categorias, fontsize=12)
    ax.set_ylim(0.98, 1.0)

    for idx, row in top5.iterrows():
        valores = [row['accuracy'], row['precision'], row['recall'], row['f1']]
        valores += valores[:1]

        label = f"{row['modelo']} ({row['nombre_clasificacion'][:8]})"
        ax.plot(angulos, valores, 'o-', linewidth=2, label=label)
        ax.fill(angulos, valores, alpha=0.15)

    ax.set_title('Top 5 Modelos - Comparación Multimétrica',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig8_metricas_comparadas.png', dpi=300, bbox_inches='tight')
    print("[OK] Figura 8 guardada: fig8_metricas_comparadas.png")
    plt.close()


def main():
    """Genera todos los gráficos."""
    print("\n" + "="*80)
    print("GENERACIÓN DE GRÁFICOS PARA PAPER PC3")
    print("="*80 + "\n")

    # Cargar datos
    print("[1/8] Cargando resultados...")
    df = cargar_resultados()
    print(f"      Cargados {len(df)} modelos\n")

    # Generar gráficos
    print("[2/8] Generando Figura 1: Comparación F1 por Tipo...")
    grafico_1_comparacion_f1_por_tipo(df)

    print("[3/8] Generando Figura 2: Rendimiento por Algoritmo...")
    grafico_2_rendimiento_algoritmos(df)

    print("[4/8] Generando Figura 3: Matriz de Confusión...")
    grafico_3_matriz_confusion_mejor_modelo()

    print("[5/8] Generando Figura 4: Distribución de Clases...")
    grafico_4_distribucion_clases(df)

    print("[6/8] Generando Figura 5: Binaria vs Multiclase...")
    grafico_5_binaria_vs_multiclase(df)

    print("[7/8] Generando Figura 6: Importancia de Features...")
    grafico_6_importancia_features()

    print("[8/8] Generando Figura 7: Heatmap F1-Scores...")
    grafico_7_heatmap_f1(df)

    print("[BONUS] Generando Figura 8: Métricas Comparadas...")
    grafico_8_metricas_comparadas(df)

    print("\n" + "="*80)
    print("✅ GENERACIÓN COMPLETADA")
    print(f"   Figuras guardadas en: {FIGURES_DIR}/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
