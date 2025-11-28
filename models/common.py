import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine
from dotenv import load_dotenv

def get_db_connection():
    """
    Crea y retorna una conexión de SQLAlchemy a la base de datos.
    Lee las credenciales desde el archivo .env.
    """
    load_dotenv()
    db_user = os.getenv("MYSQL_USER")
    db_password = os.getenv("MYSQL_PASSWORD")
    db_host = os.getenv("MYSQL_HOST")
    db_name = os.getenv("MYSQL_DB")

    if not all([db_user, db_password, db_host, db_name]):
        print("Error: Faltan variables de entorno para la base de datos en el archivo .env")
        return None

    try:
        connection_string = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
        engine = create_engine(connection_string)
        # Probar la conexión
        connection = engine.connect()
        connection.close()
        print("Conexión a la base de datos exitosa.")
        return engine
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

def get_preprocessed_data(test_size=0.2, random_state=42):
    """
    Carga los datos preprocesados desde el archivo Parquet, realiza la codificación
    y la división en conjuntos de entrenamiento y prueba.
    """
    # --- 1. Cargar Datos Preprocesados ---
    try:
        # Actualizado para reflejar la nueva estructura de carpetas
        df = pd.read_parquet('data/raw/preprocessed_crimes.parquet')
        print("DataFrame preprocesado cargado exitosamente.")
    except FileNotFoundError:
        print("Error: El archivo 'data/raw/preprocessed_crimes.parquet' no fue encontrado.")
        print("Asegúrate de que el archivo de datos original exista.")
        return None

    # --- 2. Separar Features (X) y Target (y) ---
    X = df.drop('modalidad_hecho', axis=1)
    y = df['modalidad_hecho']

    # --- 3. Codificar el Target ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Etiquetas del target codificadas.")

    # --- 4. Definir Preprocesamiento para Features ---
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64', 'int32']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 5. Dividir los datos en Entrenamiento y Prueba ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    print(f"Datos divididos en {len(X_train)} para entrenamiento y {len(X_test)} para prueba.")

    return X_train, X_test, y_train, y_test, preprocessor, le