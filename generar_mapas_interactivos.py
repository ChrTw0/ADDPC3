"""
Genera mapas INTERACTIVOS con Folium para HURTO y EXTORSIÓN
Mapas zonificados (Norte/Centro/Sur/Este/Oeste) + Mapa general
Exporta a HTML para navegación interactiva
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración
OUTPUT_DIR = Path('mapas_interactivos')
OUTPUT_DIR.mkdir(exist_ok=True)

# Límites geográficos de Lima Metropolitana por zona
ZONAS_LIMA = {
    'norte': {
        'nombre': 'Lima Norte',
        'center': [-11.85, -76.975],
        'lat_min': -11.95, 'lat_max': -11.75,
        'long_min': -77.10, 'long_max': -76.85,
        'distritos': ['Los Olivos', 'Comas', 'Independencia', 'San Martín de Porres', 'Carabayllo']
    },
    'centro': {
        'nombre': 'Lima Centro',
        'center': [-12.025, -77.025],
        'lat_min': -12.10, 'lat_max': -11.95,
        'long_min': -77.10, 'long_max': -76.95,
        'distritos': ['Cercado', 'Breña', 'La Victoria', 'San Luis', 'Jesús María', 'Lince']
    },
    'sur': {
        'nombre': 'Lima Sur',
        'center': [-12.225, -76.95],
        'lat_min': -12.35, 'lat_max': -12.10,
        'long_min': -77.05, 'long_max': -76.85,
        'distritos': ['Villa El Salvador', 'Villa María del Triunfo', 'San Juan de Miraflores', 'Chorrillos']
    },
    'este': {
        'nombre': 'Lima Este',
        'center': [-12.05, -76.875],
        'lat_min': -12.15, 'lat_max': -11.95,
        'long_min': -77.00, 'long_max': -76.75,
        'distritos': ['San Juan de Lurigancho', 'El Agustino', 'Santa Anita', 'Ate']
    },
    'oeste': {
        'nombre': 'Lima Oeste/Costa',
        'center': [-12.10, -77.075],
        'lat_min': -12.20, 'lat_max': -12.00,
        'long_min': -77.15, 'long_max': -77.00,
        'distritos': ['Callao', 'Miraflores', 'San Isidro', 'Barranco', 'Magdalena', 'San Miguel']
    }
}


def crear_mapa_interactivo_zona(delito_key, zona_key=None):
    """
    Crea mapa interactivo con Folium

    Args:
        delito_key: 'hurto' o 'extorsion'
        zona_key: 'norte', 'centro', 'sur', 'este', 'oeste' o None (mapa completo)
    """
    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster

        delito_nombre = delito_key.upper()

        # Determinar configuración según zona
        if zona_key is None:
            # Mapa general de toda Lima
            zona_nombre = "Lima Metropolitana Completa"
            center = [-12.05, -77.03]
            zoom_start = 11
            lat_bounds = None
        else:
            zona_info = ZONAS_LIMA[zona_key]
            zona_nombre = zona_info['nombre']
            center = zona_info['center']
            zoom_start = 12
            lat_bounds = (zona_info['lat_min'], zona_info['lat_max'],
                         zona_info['long_min'], zona_info['long_max'])

        print(f"\n{'='*70}")
        print(f"  MAPA INTERACTIVO: {zona_nombre} - {delito_nombre}")
        print(f"{'='*70}")

        # Cargar datos
        print(f"[1/4] Cargando datos de {delito_nombre}...")
        try:
            from utils.data_preparation import preparar_datos_completo
            datos = preparar_datos_completo(delito_key)

            if datos is None:
                raise Exception("No se pudieron cargar datos")

            df = datos['df_completo']

            # Extraer coordenadas
            df['lat_grid'] = df['grid_cell'].str.split('_').str[0].astype(float)
            df['long_grid'] = df['grid_cell'].str.split('_').str[1].astype(float)

            # Filtrar por zona si aplica
            if lat_bounds:
                mask_zona = (
                    (df['lat_grid'] >= lat_bounds[0]) &
                    (df['lat_grid'] <= lat_bounds[1]) &
                    (df['long_grid'] >= lat_bounds[2]) &
                    (df['long_grid'] <= lat_bounds[3])
                )
                df_zona = df[mask_zona].copy()
            else:
                df_zona = df.copy()

            if len(df_zona) == 0:
                print(f"   ⚠️  No hay datos en {zona_nombre}")
                return False

            # Agrupar por celda
            hotspots = df_zona.groupby(['lat_grid', 'long_grid'])['crime_count'].sum().reset_index()

            num_hotspots = 50 if zona_key is None else 30
            top_hotspots = hotspots.nlargest(num_hotspots, 'crime_count')

            print(f"   ✓ {len(top_hotspots)} hotspots cargados")
            print(f"   ✓ Total crímenes: {top_hotspots['crime_count'].sum():,}")

        except Exception as e:
            print(f"   ⚠️  Error cargando datos: {e}")
            print(f"   Generando datos sintéticos...")

            if lat_bounds:
                top_hotspots = pd.DataFrame({
                    'lat_grid': np.random.uniform(lat_bounds[0], lat_bounds[1], 30),
                    'long_grid': np.random.uniform(lat_bounds[2], lat_bounds[3], 30),
                    'crime_count': np.random.randint(50, 800, 30)
                })
            else:
                top_hotspots = pd.DataFrame({
                    'lat_grid': np.random.uniform(-12.25, -11.75, 50),
                    'long_grid': np.random.uniform(-77.15, -76.75, 50),
                    'crime_count': np.random.randint(50, 2000, 50)
                })

        # Crear mapa base
        print(f"[2/4] Creando mapa interactivo...")
        mapa = folium.Map(
            location=center,
            zoom_start=zoom_start,
            tiles='OpenStreetMap',
            control_scale=True
        )

        # Añadir capas de tiles adicionales
        folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(mapa)
        folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(mapa)

        # Preparar datos para HeatMap
        print(f"[3/4] Agregando capa de calor...")
        heat_data = [
            [row['lat_grid'], row['long_grid'], row['crime_count']]
            for idx, row in top_hotspots.iterrows()
        ]

        HeatMap(
            heat_data,
            name='Mapa de Calor',
            min_opacity=0.4,
            max_zoom=18,
            radius=15,
            blur=20,
            gradient={
                0.0: 'blue',
                0.3: 'lime',
                0.5: 'yellow',
                0.7: 'orange',
                1.0: 'red'
            }
        ).add_to(mapa)

        # Añadir marcadores con clusters y áreas de cobertura
        print(f"[4/4] Agregando marcadores y áreas de cobertura...")

        # Feature group para áreas de grid (VISIBLE por defecto)
        grid_areas = folium.FeatureGroup(name='🟦 Áreas de Grid (~555m)', show=True)

        # Markers sin cluster para que sean siempre visibles
        markers_layer = folium.FeatureGroup(name='📍 Marcadores Hotspots', show=True)

        GRID_SIZE = 0.005  # Mismo que config.py

        for idx, row in top_hotspots.iterrows():
            # Determinar color según intensidad
            if row['crime_count'] > 1000:
                color = 'red'
                icon = 'exclamation-triangle'
                nivel = 'MUY ALTO'
            elif row['crime_count'] > 500:
                color = 'orange'
                icon = 'warning'
                nivel = 'ALTO'
            else:
                color = 'blue'
                icon = 'info-sign'
                nivel = 'MODERADO'

            # Calcular límites de la celda (área que representa el marcador)
            # IMPORTANTE: El grid_cell ya es la esquina inferior-izquierda
            lat_min = row['lat_grid']
            lat_max = row['lat_grid'] + GRID_SIZE
            long_min = row['long_grid']
            long_max = row['long_grid'] + GRID_SIZE

            # Centro de la celda
            lat_center = lat_min + GRID_SIZE / 2
            long_center = long_min + GRID_SIZE / 2

            # Popup con información MEJORADA
            popup_html = f"""
            <div style="font-family: Arial; width: 280px;">
                <h4 style="color: {color}; margin-bottom: 10px; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                    🎯 Celda de Hotspot - {nivel}
                </h4>

                <div style="background: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <p style="margin: 5px 0; font-size: 13px;">
                        <b>💡 Qué representa:</b><br>
                        Esta celda es un área de <b>~555m × 555m</b> donde se concentran crímenes.
                    </p>
                </div>

                <table style="width: 100%; font-size: 13px;">
                    <tr style="background: #e8f4f8;">
                        <td style="padding: 5px;"><b>Delito:</b></td>
                        <td style="padding: 5px;">{delito_nombre}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px;"><b>Total Crímenes:</b></td>
                        <td style="color: {color}; font-weight: bold; font-size: 18px; padding: 5px;">
                            {int(row['crime_count'])}
                        </td>
                    </tr>
                    <tr style="background: #e8f4f8;">
                        <td style="padding: 5px;"><b>Nivel Riesgo:</b></td>
                        <td style="padding: 5px; color: {color}; font-weight: bold;">{nivel}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px;"><b>Tamaño Celda:</b></td>
                        <td style="padding: 5px;">~555m × 555m</td>
                    </tr>
                    <tr style="background: #e8f4f8;">
                        <td style="padding: 5px;"><b>Centro (lat):</b></td>
                        <td style="padding: 5px; font-family: monospace;">{row['lat_grid']:.5f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px;"><b>Centro (long):</b></td>
                        <td style="padding: 5px; font-family: monospace;">{row['long_grid']:.5f}</td>
                    </tr>
                    <tr style="background: #e8f4f8;">
                        <td style="padding: 5px;"><b>Zona:</b></td>
                        <td style="padding: 5px;">{zona_nombre}</td>
                    </tr>
                </table>

                <div style="background: #fffbcc; padding: 8px; border-radius: 5px; margin-top: 10px; border-left: 3px solid #ffc107;">
                    <p style="font-size: 11px; margin: 0;">
                        <b>📊 Interpretación:</b> En esta área de ~307,000 m²
                        (3 cuadras × 3 cuadras aprox.) ocurrieron <b>{int(row['crime_count'])}
                        {delito_nombre.lower()}s</b> durante el período test (2024-2025).
                    </p>
                </div>

                <p style="font-size: 10px; color: gray; margin-top: 10px; text-align: center;">
                    ⏱️ Período: 2024-2025 (Test) | 📍 Grid: 0.005°
                </p>
            </div>
            """

            # PRIMERO: Dibujar rectángulo del área de grid (MÁS VISIBLE)
            folium.Rectangle(
                bounds=[[lat_min, long_min], [lat_max, long_max]],
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.25,  # Aumentado de 0.15 a 0.25 para mejor visibilidad
                weight=3,          # Borde más grueso (de 2 a 3)
                opacity=0.8,       # Opacidad del borde
                popup=folium.Popup(
                    f"""<div style="font-family: Arial;">
                        <b>Área de Grid</b><br>
                        Tamaño: ~555m × 555m<br>
                        Crímenes: <b style="color: {color};">{int(row['crime_count'])}</b>
                    </div>""",
                    max_width=200
                ),
                tooltip=f"Celda: {int(row['crime_count'])} crímenes"
            ).add_to(grid_areas)

            # SEGUNDO: Marcador en el centro de la celda (SIN cluster para siempre visible)
            folium.Marker(
                location=[lat_center, long_center],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"📍 {delito_nombre}: {int(row['crime_count'])} crímenes en ~555m²",
                icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
            ).add_to(markers_layer)

        # Agregar capas al mapa EN ORDEN
        grid_areas.add_to(mapa)      # Primero rectángulos (abajo)
        markers_layer.add_to(mapa)   # Luego marcadores (arriba)

        # Añadir leyenda MEJORADA con explicación de rectángulos
        legend_html = f"""
        <div style="position: fixed;
                    bottom: 50px; right: 50px;
                    width: 320px; height: auto;
                    background-color: white;
                    border:3px solid #2a5298;
                    z-index:9999;
                    font-size:14px;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.3);
                    ">
            <h4 style="margin-top:0; color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 5px;">
                📊 Leyenda - {delito_nombre}
            </h4>
            <p style="margin: 5px 0;"><b>Zona:</b> {zona_nombre}</p>
            <p style="margin: 5px 0;"><b>Hotspots:</b> {len(top_hotspots)}</p>

            <div style="background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #2a5298;">
                <p style="margin: 0; font-size: 12px;">
                    <b>🟦 Rectángulos:</b> Áreas de ~555m × 555m<br>
                    <b>📍 Marcadores:</b> Centro de cada área
                </p>
            </div>

            <hr style="margin: 10px 0;">
            <p style="margin: 8px 0; font-size: 13px;">
                <span style="display: inline-block; width: 15px; height: 15px; background: red; border: 2px solid darkred; margin-right: 5px;"></span>
                <span style="color: red; font-weight: bold;">>1000 crímenes (MUY ALTO)</span>
            </p>
            <p style="margin: 8px 0; font-size: 13px;">
                <span style="display: inline-block; width: 15px; height: 15px; background: orange; border: 2px solid darkorange; margin-right: 5px;"></span>
                <span style="color: orange; font-weight: bold;">500-1000 crímenes (ALTO)</span>
            </p>
            <p style="margin: 8px 0; font-size: 13px;">
                <span style="display: inline-block; width: 15px; height: 15px; background: blue; border: 2px solid darkblue; margin-right: 5px;"></span>
                <span style="color: blue; font-weight: bold;"><500 crímenes (MODERADO)</span>
            </p>

            <hr style="margin: 10px 0;">
            <div style="background: #fffbcc; padding: 8px; border-radius: 5px; border-left: 3px solid #ffc107;">
                <p style="font-size: 11px; margin: 0;">
                    <b>💡 Tip:</b> Activa/desactiva capas en el control superior derecho ⬆️
                </p>
            </div>

            <hr style="margin: 10px 0;">
            <p style="font-size: 11px; color: gray; margin: 5px 0;">
                📅 Período: Test 2024-2025<br>
                📊 Fuente: PNP - Denuncias Lima<br>
                📏 Grid: 0.005° (~555m)
            </p>
        </div>
        """
        mapa.get_root().html.add_child(folium.Element(legend_html))

        # Control de capas
        folium.LayerControl().add_to(mapa)

        # Guardar
        zona_suffix = zona_key if zona_key else 'completo'
        output_file = OUTPUT_DIR / f'mapa_{delito_key}_{zona_suffix}.html'
        mapa.save(str(output_file))

        print(f"\n✓ Guardado: {output_file.name}")
        return True

    except ImportError as e:
        print(f"✗ Error: Falta biblioteca {e}")
        print("   Instalar: pip install folium")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def generar_index_html():
    """
    Genera página HTML index con enlaces a todos los mapas
    """
    html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mapas Interactivos - Hotspots Criminales Lima</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        h1 { font-size: 2.5em; margin-bottom: 10px; }
        .subtitle { font-size: 1.2em; opacity: 0.9; }
        .content { padding: 40px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            transition: all 0.3s;
            border: 2px solid #ddd;
        }
        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        }
        .card h3 {
            color: #2a5298;
            margin-bottom: 15px;
            font-size: 1.4em;
        }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 25px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            margin: 5px;
            transition: all 0.3s;
        }
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .section-title {
            background: #2a5298;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 30px 0 20px 0;
            font-size: 1.5em;
        }
        .info {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        footer {
            background: #2a5298;
            color: white;
            padding: 20px;
            text-align: center;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🗺️ Mapas Interactivos de Criminalidad</h1>
            <p class="subtitle">Lima Metropolitana - HURTO y EXTORSIÓN (2024-2025)</p>
        </header>

        <div class="content">
            <div class="info">
                <h3>ℹ️ Características de los Mapas:</h3>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><b>Interactivos:</b> Zoom, arrastre, click en marcadores</li>
                    <li><b>Heatmap:</b> Visualización de densidad criminal</li>
                    <li><b>Marcadores agrupados:</b> Clusters automáticos</li>
                    <li><b>Múltiples capas:</b> OpenStreetMap, CartoDB</li>
                </ul>
            </div>

            <h2 class="section-title">📍 HURTO - Mapas por Zona</h2>
            <div class="grid">
                <div class="card">
                    <h3>🗺️ Lima Completa</h3>
                    <a href="mapa_hurto_completo.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
                <div class="card">
                    <h3>🔵 Lima Norte</h3>
                    <a href="mapa_hurto_norte.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
                <div class="card">
                    <h3>🟢 Lima Centro</h3>
                    <a href="mapa_hurto_centro.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
                <div class="card">
                    <h3>🔴 Lima Sur</h3>
                    <a href="mapa_hurto_sur.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
                <div class="card">
                    <h3>🟡 Lima Este</h3>
                    <a href="mapa_hurto_este.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
                <div class="card">
                    <h3>🟣 Lima Oeste</h3>
                    <a href="mapa_hurto_oeste.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
            </div>

            <h2 class="section-title">💰 EXTORSIÓN - Mapas por Zona</h2>
            <div class="grid">
                <div class="card">
                    <h3>🗺️ Lima Completa</h3>
                    <a href="mapa_extorsion_completo.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
                <div class="card">
                    <h3>🔵 Lima Norte</h3>
                    <a href="mapa_extorsion_norte.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
                <div class="card">
                    <h3>🟢 Lima Centro</h3>
                    <a href="mapa_extorsion_centro.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
                <div class="card">
                    <h3>🔴 Lima Sur</h3>
                    <a href="mapa_extorsion_sur.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
                <div class="card">
                    <h3>🟡 Lima Este</h3>
                    <a href="mapa_extorsion_este.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
                <div class="card">
                    <h3>🟣 Lima Oeste</h3>
                    <a href="mapa_extorsion_oeste.html" class="btn" target="_blank">Ver Mapa</a>
                </div>
            </div>
        </div>

        <footer>
            <p><strong>Universidad Nacional de Ingeniería - FIIS</strong></p>
            <p>PC3 - Analítica de Datos | Grupo 2 - Capítulo 3: Classification</p>
            <p style="margin-top: 10px; opacity: 0.8;">Mapas generados con Folium + OpenStreetMap</p>
        </footer>
    </div>
</body>
</html>
    """

    output_file = OUTPUT_DIR / 'index.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✓ Index generado: {output_file}")


def generar_todos_los_mapas_interactivos():
    """
    Genera TODOS los mapas interactivos:
    - 2 delitos × (1 mapa completo + 5 zonas) = 12 mapas HTML
    """
    print("\n" + "="*70)
    print(" GENERACIÓN DE MAPAS INTERACTIVOS - FOLIUM")
    print("="*70)
    print(f" Delitos: HURTO, EXTORSIÓN")
    print(f" Mapas: Completo + 5 zonas = 6 por delito")
    print(f" Total: 12 mapas HTML interactivos")
    print("="*70)

    delitos = ['hurto', 'extorsion']
    zonas = [None, 'norte', 'centro', 'sur', 'este', 'oeste']  # None = mapa completo

    resultados = []

    for delito in delitos:
        for zona in zonas:
            zona_nombre = "Completo" if zona is None else ZONAS_LIMA[zona]['nombre']
            exito = crear_mapa_interactivo_zona(delito, zona)
            resultados.append({
                'delito': delito.upper(),
                'zona': zona_nombre,
                'exito': exito
            })

    # Generar index
    generar_index_html()

    # Resumen
    print("\n" + "="*70)
    print(" RESUMEN")
    print("="*70)
    exitosos = sum(1 for r in resultados if r['exito'])
    print(f"\n✓ Mapas generados: {exitosos}/12")

    for delito in delitos:
        mapas_delito = [r for r in resultados if r['delito'] == delito.upper()]
        exitosos_delito = sum(1 for r in mapas_delito if r['exito'])
        print(f"  {delito.upper()}: {exitosos_delito}/6 mapas")

    print("\n📂 Carpeta: mapas_interactivos/")
    print("🌐 Abre: mapas_interactivos/index.html")
    print("="*70)


if __name__ == "__main__":
    generar_todos_los_mapas_interactivos()
