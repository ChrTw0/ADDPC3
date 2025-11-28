PROYECTO PC3 - PREDICCION DE CRIMENES
======================================

Estructura del Proyecto (MODULAR):
-----------------------------------
Pc3Final/
├── ejecutar_todos_modelos.py    # Script principal (simplificado)
│
├── config/
│   ├── __init__.py
│   └── config.py                 # Configuración central (constantes, hiperparámetros)
│
├── models/
│   ├── __init__.py
│   ├── common.py                 # Conexión a base de datos
│   ├── regression_models.py     # Modelos de regresión
│   ├── classification_models.py # Modelos de clasificación
│   └── best_models/              # (Se genera) Mejores modelos entrenados
│
├── utils/
│   ├── __init__.py
│   ├── data_preparation.py      # Extracción y preparación de datos
│   ├── feature_engineering.py   # Creación de features
│   └── model_evaluation.py      # Evaluación y persistencia
│
├── results/                      # (Se genera) Resultados CSV
├── .env                          # Credenciales de base de datos
└── requirements.txt              # Dependencias

Archivos Principales:
---------------------
1. ejecutar_todos_modelos.py - Script principal orquestador
2. config/config.py - Configuración centralizada
3. models/ - Implementación de modelos
4. utils/ - Utilidades para datos y evaluación
5. .env - Credenciales MySQL

Dependencias:
-------------
- pandas
- numpy
- scikit-learn
- sqlalchemy
- mysql-connector-python
- python-dotenv
- joblib
- xgboost (opcional)

Cómo Ejecutar:
--------------
1. Verificar que .env tenga las credenciales correctas de MySQL
2. Ejecutar: python ejecutar_todos_modelos.py
3. Seleccionar delito a procesar (HURTO/EXTORSIÓN/AMBOS)
4. Elegir si optimizar hiperparámetros

Modelos Implementados:
----------------------
REGRESIÓN (6): Random Forest, Gradient Boosting, Extra Trees, KNN, AdaBoost, XGBoost
CLASIFICACIÓN (7): SGD, Logistic, Random Forest, Gradient Boosting, KNN, Decision Tree, AdaBoost

Total: 24 modelos (12 para HURTO + 12 para EXTORSIÓN)
