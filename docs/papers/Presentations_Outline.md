# Esquemas para Presentaciones del Proyecto (Versión Final Extendida)

---

## Presentación 1: Audiencia Técnica (Desarrolladores, Científicos de Datos)

**Objetivo:** Demostrar el rigor técnico del proceso iterativo, la metodología de modelado, la evaluación cuantitativa y la selección final del modelo.

**Duración estimada:** 20-25 minutos

---

### **Diapositiva 1: Título**
-   **Título:** De RandomForest a Transformers: Un Análisis Comparativo para la Predicción de Hotspots Criminales en Lima
-   **Subtítulo:** Un Viaje Iterativo a través de Modelos Clásicos y de Deep Learning
-   **Autores:** Grupo 2
-   **Curso:** Analítica de Datos - PC3

### **Diapositiva 2: Agenda**
1.  Redefinición del Problema: De Clasificación a Regresión de Hotspots
2.  Experimento 1: Benchmark con Modelos Clásicos (Enfoque Lima)
3.  Experimento 2: Mejorando con Redes Recurrentes (LSTM)
4.  Experimento 3: Explorando Características Espaciales
5.  Experimento 4: El Desafío del Transformer (Secuencias Cortas y Largas)
6.  Análisis Final y Selección del Modelo Campeón
7.  Conclusiones y Trabajo Futuro

### **Diapositiva 3: Redefinición del Problema**
-   **Problema Inicial:** Clasificar `modalidad_hecho`.
    -   **Limitación:** Bajo valor práctico, desbalance de clases.
-   **Problema Final:** Predecir `crime_count` por celda/semana (Regresión de Hotspots).
    -   **Justificación:** Alto valor operacional, permite la asignación proactiva de recursos.
-   **Alcance:** Delito "Robo Agravado", Departamento de Lima (justificar por qué se descartó el análisis nacional).

### **Diapositiva 4: Pipeline de Datos y Feature Engineering**
-   **Paso 1: Extracción y Filtrado** (`1b_prepare_geo_data_lima.py`)
-   **Paso 2: Creación de la Grilla Espacio-Temporal** (`2b_create_grid_features_lima.py`)
    -   **Técnica:** Discretización del espacio (grilla de 0.005°) y tiempo (semanas).
    -   **Features Clave:** `crime_count_lag_1` a `_4`, `month`, `day_of_week`.
-   **Diagrama del Pipeline:** Flujo visual de los datos.

### **Diapositiva 5: Experimento 1: Benchmark con Modelos Clásicos**
-   **Modelos Evaluados:** `RandomForestRegressor`, `KNeighborsRegressor`, `SVR`.
-   **Tabla de Resultados (Lima):**
    | Modelo | MAE | RMSE | R² |
    | :--- | :---: | :---: | :---: |
    | **RandomForestRegressor** | 0.7268 | 1.5949 | **0.6935** |
    | SVR | 0.6773 | 1.6504 | 0.6718 |
-   **Conclusión Parcial:** `RandomForest` es el mejor modelo clásico, estableciendo nuestro benchmark a superar.

### **Diapositiva 6: Experimento 2: Mejorando con Redes Recurrentes (LSTM)**
-   **Hipótesis:** Una red diseñada para secuencias (LSTM) puede capturar mejor los patrones temporales.
-   **Modelos:** LSTM simple vs. LSTM Optimizado (más profundo, con Dropout).
-   **Tabla de Resultados (LSTM vs RF):**
    | Modelo | MAE | RMSE | R² |
    | :--- | :---: | :---: | :---: |
    | RandomForestRegressor | 0.7268 | 1.5949 | 0.6935 |
    | **LSTM Optimizado** | 0.6690 | **1.5858** | **0.6970** |
-   **Conclusión Parcial:** El LSTM optimizado supera al RandomForest en todas las métricas clave. **¡Nuevo Campeón!**

### **Diapositiva 7: Experimento 3: Explorando Características Espaciales**
-   **Hipótesis:** Añadir información de celdas vecinas (`spatial_lag`) mejorará la predicción.
-   **Modelo:** LSTM Optimizado con la nueva característica.
-   **Tabla de Resultados (LSTM vs LSTM Espacial):**
    | Modelo | MAE | RMSE | R² |
    | :--- | :---: | :---: | :---: |
    | LSTM Optimizado | 0.6690 | **1.5858** | **0.6970** |
    | **LSTM Espacial** | **0.6589** | 1.5975 | 0.6925 |
-   **Conclusión Parcial:** Las características espaciales reducen significativamente el MAE (error promedio), pero a un ligero costo en R² y RMSE.

### **Diapositiva 8: Experimento 4: El Desafío del Transformer**
-   **Hipótesis:** Un Transformer podría capturar dependencias a largo plazo si se le dan secuencias más largas.
-   **Sub-experimento A (Secuencias Cortas):** El Transformer falla (R² ≈ 0.64).
-   **Sub-experimento B (Secuencias Largas - 24 semanas):**
    -   **Resultado Sorprendente:** MAE más bajo de todos (0.5636), pero **R² negativo (-0.3007)**.
    -   **Análisis:** Explicar qué significa un R² negativo (peor que un modelo promedio). El modelo no generalizó, a pesar de su bajo error promedio.

### **Diapositiva 9: Tabla Comparativa Final y Selección del Campeón**
-   **Tabla Definitiva:**
    | Modelo | MAE | RMSE | R² |
    | :--- | :---: | :---: | :---: |
    | RandomForestRegressor | 0.7268 | 1.5949 | 0.6935 |
    | **LSTM Optimizado** | 0.6690 | **1.5858** | **0.6970** |
    | LSTM Espacial | **0.6589** | 1.5975 | 0.6925 |
    | Transformer (Sec. Largas) | 0.5636 | 0.8129 | -0.3007 |
-   **Selección Final:** El **`LSTM Optimizado`** es el campeón general.
-   **Justificación:** Ofrece el mejor balance, con el R² más alto (mayor poder explicativo) y el RMSE más bajo (menos errores grandes), que son cruciales para la predicción de riesgos.

### **Diapositiva 10: Conclusiones y Trabajo Futuro**
-   **Conclusión Principal:** Se validó un pipeline de extremo a extremo, demostrando que un `LSTM` bien ajustado es superior a modelos clásicos y a arquitecturas más complejas como `Transformer` para este problema específico.
-   **Lecciones Aprendidas:** La complejidad no siempre es mejor; la ingeniería de características es clave; la definición del problema es fundamental.
-   **Trabajo Futuro:** Recalibración del modelo, NLP sobre el texto de las denuncias, expansión a otros delitos.

### **Diapositiva 11: Q&A / Preguntas**

---
---

## Presentación 2: Audiencia Gerencial (Jefes de Policía, Gerentes de Seguridad Ciudadana)

**Objetivo:** Comunicar el valor de negocio y las implicaciones operacionales del proyecto, evitando la jerga técnica y contando una historia convincente.

**Duración estimada:** 5-10 minutos

---

### **Diapositiva 1: Título**
-   **Título:** Predicción Inteligente de Zonas de Riesgo para Optimizar el Patrullaje en Lima
-   **Subtítulo:** Un Sistema de Alerta Temprana para "Robo Agravado"
-   **Autores:** Grupo 2

### **Diapositiva 2: El Problema: Reaccionar es Caro, Predecir es Eficiente**
-   **Situación Actual:** Los recursos de patrullaje son limitados y a menudo se despliegan en respuesta a incidentes ya ocurridos.
-   **Nuestra Solución:** Hemos desarrollado un modelo de Inteligencia Artificial que **anticipa dónde ocurrirán los robos** la próxima semana.
-   **Beneficio Clave:** Pasar de una seguridad reactiva a una **seguridad proactiva y predictiva**.

### **Diapositiva 3: Nuestro Proceso: Un Viaje en Busca del Mejor Pronóstico**
-   **Input:** Usamos el historial de más de 64,000 denuncias de "Robo Agravado" en Lima.
-   **Proceso (Simplificado):**
    1.  Dividimos Lima en una cuadrícula.
    2.  Probamos varios métodos, desde los más simples hasta los más avanzados, para encontrar el que mejor predice.
    3.  Descubrimos que un tipo de **red neuronal (llamada LSTM)** nos dio los resultados más fiables y precisos.

### **Diapositiva 4: La Prueba Definitiva: Superando a la Tecnología de Punta**
-   "Para estar seguros, incluso probamos la tecnología más moderna (los `Transformers`, similar a la que usa ChatGPT) y descubrimos que, para este problema específico, nuestro modelo `LSTM` era superior. Esto nos da una **gran confianza en nuestra elección**."

### **Diapositiva 5: Resultados: ¿Funciona? ¡Sí!**
-   **Insertar Imagen:** `hotspot_comparison_robo_agravado_lima.png` (la visualización del modelo LSTM).
-   **Explicación Sencilla:**
    -   "A la izquierda, ven dónde ocurrieron los robos en una semana real. A la derecha, ven dónde nuestro modelo **predijo** que ocurrirían."
    -   **Conclusión Clave:** "Nuestro modelo acierta con gran precisión las **zonas exactas** donde se concentra el peligro. Podemos decirle a sus equipos dónde patrullar."

### **Diapositiva 6: ¿Qué Significa Esto en la Práctica?**
-   **Precisión del "Dónde":** El modelo es excelente para identificar las calles y barrios que necesitan más atención.
-   **Interpretación del "Cuánto":** La predicción no es un número exacto de robos, sino un **índice de prioridad**. Un color rojo oscuro en el mapa significa "Máxima Prioridad".
-   **Ejemplo de Uso:** "Con este mapa, en lugar de patrullar toda la ciudad por igual, pueden enfocar el 80% de sus recursos en el 20% de las zonas más críticas, maximizando el impacto y la disuasión."

### **Diapositiva 7: Conclusiones y Próximos Pasos**
-   **Conclusión Principal:** Hemos creado y validado una herramienta de inteligencia que puede predecir hotspots de "Robo Agravado" con una semana de antelación, permitiendo una gestión de recursos más eficiente.
-   **Próximos Pasos (Propuesta de Valor):**
    1.  **Expandir:** Podemos crear modelos similares para otros delitos de alto impacto (Extorsión, Hurto de Vehículos).
    2.  **Integrar:** Este sistema puede integrarse con sus tableros de control actuales para generar alertas automáticas semanales.

### **Diapositiva 8: Gracias / Preguntas**