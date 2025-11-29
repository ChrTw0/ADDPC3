# 🚨 ADDPC3 - Predicción de Hotspots Criminales en Lima

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Clasificación de Hotspots Criminales usando Machine Learning - Capítulo 3**

Sistema integral de clasificación para predicción de zonas de alto riesgo criminal en Lima Metropolitana, utilizando 7 algoritmos de clasificación aplicados a 3 problemas operacionales diferentes.

---

## 📊 Resumen del Proyecto

- **42 modelos de clasificación** (7 algoritmos × 3 tipos × 2 delitos)
- **Delitos analizados:** HURTO (213K registros) y EXTORSIÓN (32K registros)
- **3 problemas de clasificación:**
  - 🟢🟡🟠🔴 **Nivel de Riesgo** (4 clases) - Asignación de recursos
  - ⚠️ **Hotspot Crítico** (binario) - Intervención inmediata
  - 📈📊📉 **Tendencia** (3 clases) - Sistema de alerta temprana
- **Métricas:** F1-Score promedio de **94.10%** (HURTO) y **93.87%** (EXTORSIÓN)

---

## 🗂️ Estructura del Repositorio

```
ADDPC3/
│
├── 📁 scripts/                         # Scripts ejecutables
│   ├── ejecutar_todos_modelos.py      # ⭐ SCRIPT PRINCIPAL
│   ├── analisis/                      # Análisis exploratorios
│   │   ├── analisis_critico_problema.py
│   │   ├── analisis_tendencias_contexto.py
│   │   └── generar_analisis_avanzado.py
│   └── visualizacion/                 # Generación de gráficos y mapas
│       ├── generar_graficos_paper.py
│       ├── generar_mapa_mejorado.py
│       ├── generar_mapas_interactivos.py
│       ├── generar_mapas_zonificados.py
│       └── validacion_metodologia_mysql.py
│
├── 📁 config/                          # Configuración centralizada
│   ├── __init__.py
│   └── config.py                      # Hiperparámetros, constantes, delitos
│
├── 📁 models/                          # Implementación de modelos ML
│   ├── __init__.py
│   ├── common.py                      # Conexión a base de datos
│   ├── classification_models.py       # 7 algoritmos de clasificación
│   └── best_models/                   # Modelos entrenados (.joblib)
│
├── 📁 utils/                           # Utilidades compartidas
│   ├── __init__.py
│   ├── data_preparation.py            # Extracción y preparación de datos
│   ├── feature_engineering.py         # Ingeniería de características
│   ├── target_engineering.py          # Creación de targets de clasificación
│   └── model_evaluation.py            # Evaluación y persistencia
│
├── 📁 docs/                            # Documentación completa
│   ├── paper/                         # Paper académico
│   │   ├── PaperPc3_COMPLETO.md      # Paper completo
│   │   ├── REFERENCIAS.txt            # Estado del arte
│   │   └── PaperPc3_Secciones_Faltantes.txt
│   ├── metodologia/                   # Metodología detallada
│   │   ├── PaperPc3_Parte1_Metodologia.md
│   │   └── marco_teorico.txt
│   ├── resultados/                    # Resultados y análisis
│   │   └── PaperPc3_Parte2_Resultados_Completo.md
│   └── guias/                         # Guías de uso
│       ├── INICIO_RAPIDO.md
│       ├── GUIA_OPTIMIZACION.md
│       ├── GUIA_MULTIDELITO.md
│       └── LibroCap3.md
│
├── 📁 results/                         # Resultados generados (CSV)
├── 📁 figures/                         # Gráficos y visualizaciones
├── 📁 mapas_interactivos/              # Mapas HTML interactivos
│
├── .env                                # ⚠️ Credenciales MySQL (no versionado)
├── .gitignore                          # Archivos excluidos de git
├── requirements.txt                    # Dependencias Python
└── README.md                           # Este archivo
```

---

## 🚀 Inicio Rápido

### 1️⃣ **Clonar el Repositorio**

```bash
git clone https://github.com/ChrTw0/ADDPC3.git
cd ADDPC3
```

### 2️⃣ **Instalar Dependencias**

```bash
pip install -r requirements.txt
```

### 3️⃣ **Configurar Base de Datos**

Crear archivo `.env` en la raíz con tus credenciales MySQL:

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=tu_password
DB_NAME=crimenes_lima
DB_PORT=3306
```

### 4️⃣ **Ejecutar el Pipeline Principal**

```bash
python scripts/ejecutar_todos_modelos.py
```

**Opciones del menú:**
1. Solo HURTO (213K registros)
2. Solo EXTORSIÓN (32K registros)
3. AMBOS (42 modelos completos) ⭐ _Recomendado_

**Optimización:**
- **NO optimizar** → Entrenamiento rápido (~10-15 min)
- **SÍ optimizar** → Búsqueda de hiperparámetros (~30-45 min)

---

## 📈 Modelos Implementados

### 7 Algoritmos de Clasificación (Capítulo 3)

| Algoritmo | Tipo | Características |
|-----------|------|-----------------|
| **SGD Classifier** | Lineal | Rápido, escalable |
| **Logistic Regression** | Lineal | Interpretable, baseline |
| **Random Forest** | Ensemble | Robusto, no lineal |
| **Gradient Boosting** | Ensemble | Alto rendimiento |
| **KNN** | Basado en instancias | Simple, efectivo |
| **Decision Tree** | Árbol | Interpretable, rápido |
| **AdaBoost** | Ensemble | Boosting adaptativo |

### 3 Problemas de Clasificación

#### 1. **Nivel de Riesgo** (Multiclase - 4 niveles)
- 🟢 **Bajo** (0-2 crímenes) → Patrullaje rutinario
- 🟡 **Medio** (3-5 crímenes) → Patrullaje reforzado
- 🟠 **Alto** (6-10 crímenes) → Operativo focalizado
- 🔴 **Muy Alto** (>10 crímenes) → Intervención especial

**Pregunta:** _"¿Qué nivel de recursos necesita esta zona?"_

#### 2. **Hotspot Crítico** (Binario)
- ✅ **Normal** → Sin intervención
- ⚠️ **Crítico** (>5 crímenes) → Requiere intervención

**Pregunta:** _"¿Debo intervenir en esta zona esta semana?"_

#### 3. **Tendencia de Riesgo** (Multiclase - 3 niveles)
- 📉 **Descenso** → Zona mejorando
- 📊 **Estable** → Sin cambios significativos
- 📈 **Escalada** → Zona empeorando

**Pregunta:** _"¿Esta zona está mejorando o empeorando?"_

---

## 📊 Resultados Destacados

### Mejores Modelos por Delito

| Delito | Mejor Modelo | F1-Score | Tipo |
|--------|--------------|----------|------|
| **HURTO** | Gradient Boosting | **99.56%** | Hotspot Crítico |
| **EXTORSIÓN** | Random Forest | **99.23%** | Hotspot Crítico |

### Estadísticas Generales

- ✅ **100% de modelos** superan el umbral de producción (F1 > 85%)
- 📈 **F1 promedio:** 94.10% (HURTO), 93.87% (EXTORSIÓN)
- ⚡ **Mejor algoritmo:** Gradient Boosting
- 🎯 **Problema más predecible:** Hotspot Crítico (binario)

---

## 🛠️ Uso Avanzado

### Generar Mapas Interactivos

```bash
python scripts/visualizacion/generar_mapas_interactivos.py
```

### Análisis Exploratorio

```bash
python scripts/analisis/analisis_critico_problema.py
python scripts/analisis/analisis_tendencias_contexto.py
```

### Validación Metodológica

```bash
python scripts/visualizacion/validacion_metodologia_mysql.py
```

---

## 📦 Dependencias Principales

```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
sqlalchemy >= 1.4.0
mysql-connector-python >= 8.0.0
python-dotenv >= 0.19.0
joblib >= 1.1.0
xgboost >= 1.5.0
```

---

## 📚 Documentación

- **Paper completo:** [`docs/paper/PaperPc3_COMPLETO.md`](docs/paper/PaperPc3_COMPLETO.md)
- **Metodología:** [`docs/metodologia/PaperPc3_Parte1_Metodologia.md`](docs/metodologia/PaperPc3_Parte1_Metodologia.md)
- **Resultados:** [`docs/resultados/PaperPc3_Parte2_Resultados_Completo.md`](docs/resultados/PaperPc3_Parte2_Resultados_Completo.md)
- **Guía de inicio:** [`docs/guias/INICIO_RAPIDO.md`](docs/guias/INICIO_RAPIDO.md)

---

## 🏗️ Arquitectura del Sistema

El proyecto sigue una **arquitectura modular** que separa responsabilidades:

```
┌─────────────────────────────────────────────────────────┐
│  scripts/ejecutar_todos_modelos.py (Orquestador)       │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼─────┐
│ config │      │  utils   │
│        │      │          │
│ • Hiper│      │ • Data   │
│   params│     │ • Features│
│ • Const│      │ • Targets│
└────────┘      └────┬─────┘
                     │
              ┌──────▼──────┐
              │   models    │
              │             │
              │ • Common    │
              │ • Classifier│
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   results   │
              │  & figures  │
              └─────────────┘
```

---

## 👥 Contribuidores

**Grupo 2 - PC3**
- Universidad Nacional de Ingeniería
- Facultad de Ingeniería Industrial y de Sistemas
- Curso: Analitica de Datos

---

## 📄 Licencia

Este proyecto es de código abierto bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 🔗 Enlaces Útiles

- 📖 [Hands-On Machine Learning (Capítulo 3)](https://github.com/ageron/handson-ml3)
- 🔍 [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- 🗺️ [Mapas Interactivos](mapas_interactivos/index.html)

---

## ⚠️ Notas Importantes

1. **Base de datos requerida:** El proyecto necesita acceso a una base de datos MySQL con los datos de denuncias criminales de Lima.
2. **Tiempo de ejecución:** El pipeline completo con 42 modelos tarda entre 10-45 minutos dependiendo de la optimización.
3. **Memoria RAM:** Se recomienda al menos 8GB de RAM para procesar ambos delitos simultáneamente.
4. **Python 3.8+:** Asegúrate de tener Python 3.8 o superior instalado.

---

**Última actualización:** 28 de Noviembre de 2025
