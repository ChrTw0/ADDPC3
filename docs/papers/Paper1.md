### Página 1

2024 International Conference on Machine Learning and Applications (ICMLA)

## Multi-Label Behavioral Health Classification from Police Narrative Report

**Abm Adnan Azmee\***, **Francis Nweke\***, **Md Abdullah Al Hafiz Khan†**, **Yong Pei†**, **Dominic Thomas†**, **Monica Nandan†**
Kennesaw State University, Georgia, USA
\*{aazmee, fnweke}@students.kennesaw.edu
†{mkhan74, ypei, dthom310, mnandan}@kennesaw.edu

***Abstract***—**Comprender el comportamiento de una comunidad es crucial para construir una sociedad fuerte. Para lograrlo, debemos ofrecer ayuda a las personas que necesitan apoyo en salud conductual. En los últimos años, el término "salud conductual" ha surgido para definir una amplia gama de problemas sociales, incluyendo la salud mental, el abuso de sustancias, etc. Durante las emergencias, los socorristas garantizan la seguridad y el bienestar de la comunidad. En la actualidad, los socorristas dependen de la iteración manual a través de informes narrativos públicos al final de cada mes para identificar casos vinculados a individuos que luchan con problemas de salud conductual. Este método resulta arduo y requiere mucho tiempo. Además, hay una disponibilidad limitada de recursos de expertos, lo que provoca retrasos en la identificación y el tratamiento de casos potenciales. El procesamiento del lenguaje natural y las técnicas de aprendizaje profundo han mostrado resultados prometedores en el análisis de datos textuales. Sin embargo, es un desafío para los modelos tradicionales identificar los casos, ya que los informes son de naturaleza no estructurada y contienen ruido. Para superar este desafío, en esta investigación, desarrollamos un novedoso marco de detección de salud conductual multi-etiqueta habilitado por atención para identificar diferentes casos de salud conductual. Nuestro modelo propuesto utiliza mecanismos de atención y considera información de diferentes niveles (palabra, oración y documento) para obtener una comprensión integral del informe narrativo policial e identificar eficazmente diferentes tipos de casos de salud conductual. Además, nuestro modelo propuesto superó a los modelos de última generación con una precisión del 71%.**

***Index Terms***—**Procesamiento del Lenguaje Natural, Redes Neuronales, Salud Conductual, Informe Policial, Llamadas al 911**

### I. INTRODUCCIÓN

El término salud conductual a menudo abarca condiciones relacionadas con la salud mental, el abuso de sustancias, las crisis y los síntomas físicos causados por el estrés. La salud conductual es esencial para nuestro bienestar general. En los EE. UU., alrededor del 20.78% de los adultos sufren diferentes tipos de problemas de salud mental, lo que supone más de 50 millones de personas. Mientras que 14.1 millones de adultos sufren de enfermedades mentales graves. Además, más del 10% de los jóvenes en los EE. UU. han afirmado que sus dificultades de salud conductual están afectando negativamente su capacidad para desempeñarse eficazmente en el lugar de trabajo.

Los socorristas, incluyendo la policía, el servicio de bomberos y el personal de EMT, son el punto de contacto inicial para aquellos que experimentan una crisis. Después de responder a una crisis, los socorristas escriben informes sobre los incidentes. La información contenida en estos informes narrativos públicos puede servir como un recurso crucial para obtener información sobre el bienestar de la comunidad. Los socorristas, en particular los agentes del orden, revisan manualmente estos informes al final de cada mes para identificar posibles casos de salud conductual. Las revisiones manuales de estos informes consumen mucho tiempo y son propensas a errores. Además, hay una escasez de agentes del orden debido a la dificultad en el reclutamiento y un aumento en la tasa de renuncias. Además, determinar los casos de salud conductual requiere la experiencia de expertos en la materia (SMEs). En este estudio, empleamos expertos en la materia para anotar el informe narrativo público proporcionado por los socorristas (policía).

Los desarrollos recientes en el procesamiento del lenguaje natural (NLP) han demostrado ser un activo en el análisis y la derivación de conocimientos a partir de datos textuales. El NLP se utiliza actualmente para diversas tareas como resumir, parafrasear, análisis de sentimientos, etc. Impulsado por el avance del aprendizaje profundo, el progreso realizado en el NLP ha creado oportunidades para comprender e interpretar el significado del texto. Sin embargo, los datos en la narrativa pública de los socorristas carecen de estructura y contienen una cantidad significativa de ruido, lo que dificulta la comprensión por parte de las técnicas tradicionales de NLP y de aprendizaje profundo. En, los autores de trabajos anteriores solo consideraron la información a nivel de documento para analizar los contenidos. Sin embargo, depender únicamente de la información a nivel de documento no proporciona una comprensión completa del texto. La información a nivel de palabra y oración también es muy importante para obtener una comprensión contextual del texto. Para superar las limitaciones de los trabajos anteriores, en este estudio, desarrollamos un novedoso clasificador de salud conductual multinivel habilitado por atención, que considera tres niveles diferentes de datos para obtener conocimientos completos de los informes.

Además, la investigación anterior generalmente agrupaba cada incidente en una sola categoría. Sin embargo, en el mundo real, un incidente a menudo puede involucrar múltiples categorías. Para abordar este problema en este trabajo, consideramos la clasificación multi-etiqueta, donde cada informe puede contener múltiples problemas relacionados con la salud conductual. En nuestro marco de detección de salud conductual multinivel habilitado por atención, consideramos palabras relevantes, representaciones de oraciones y características del documento para obtener una comprensión contextual crucial para determinar los casos de salud conductual. Adicionalmente, nuestro enfoque propuesto implicó una clasificación multi-etiqueta para reflejar mejor los escenarios del mundo real donde un caso puede abarcar múltiples categorías simultáneamente. Nuestras contribuciones a la investigación se resumen de la siguiente manera:

---

### Página 2

*   Propusimos un novedoso marco de detección de salud conductual multi-nivel habilitado por atención que primero utilizó mecanismos de atención para obtener una representación vectorial contextualizada habilitada por consulta a partir de palabras ponderadas y una representación automática de oraciones contextuales; segundo, aprendió una representación integral del documento y acumuló las características para identificar casos de salud conductual.
*   Implementamos un método de clasificación multi-etiqueta que aborda la complejidad de los escenarios del mundo real donde un solo caso puede pertenecer a múltiples categorías.
*   Realizamos evaluaciones exhaustivas en informes narrativos policiales reales para demostrar la efectividad de nuestro enfoque propuesto en escenarios del mundo real.

### II. TRABAJO RELACIONADO

En esta sección, vamos a discutir los trabajos relacionados con nuestro estudio. Antes de discutir la investigación existente, analicemos el concepto de salud conductual. Según los Centros de Servicios de Medicare y Medicaid de EE. UU., la salud conductual abarca los aspectos psicológicos y conductuales que impactan el estado de bienestar completo de los individuos. La salud conductual, según la define la Administración de Servicios de Abuso de Sustancias y Salud Mental (SAMHSA) de EE. UU., incluye una gama más amplia de áreas como el bienestar mental, el tratamiento de trastornos mentales y por uso de sustancias, y el apoyo a las personas afectadas por la situación. Al revisar la literatura existente, encontramos que una parte significativa de la investigación actual se centra principalmente en la detección de la salud mental. Los investigadores se basaron en datos de diversas plataformas de redes sociales, textos clínicos y registros de salud electrónicos (EHR) para detectar diferentes problemas de salud mental, como depresión, anorexia, suicidio, etc. En, los autores utilizaron datos de la plataforma de redes sociales Twitter para la predicción de la depresión ansiosa. En esta investigación, los investigadores entrenaron tres clasificadores de aprendizaje automático utilizando un método de votación por conjunto para clasificar el texto. Los investigadores en utilizaron datos de los subreddit r/depression y r/SuicideWatch para detectar la depresión y la ideación suicida. Los autores emplearon algoritmos de aprendizaje automático como random forest y SVM y lograron una precisión del 77%. Además, mencionaron que para mejorar su rendimiento, utilizarían técnicas de aprendizaje profundo en su trabajo futuro. En, los autores utilizaron datos de las redes sociales Twitter y Reddit para detectar dos trastornos de salud mental comunes, la depresión y la anorexia. Además, utilizaron TF-IDF y el algoritmo de aprendizaje profundo CNN para detectar publicaciones que contenían estos problemas de salud mental. Además, utilizaron precisión, recall y F-score para evaluar el rendimiento de su modelo. En un estudio similar, Kim et al. detectaron trastornos mentales como depresión, ansiedad, trastorno bipolar, esquizofrenia y autismo a partir de publicaciones de Reddit. Para detectar las publicaciones, los autores utilizaron varios clasificadores de aprendizaje automático y de aprendizaje profundo. Además, compararon el rendimiento de los clasificadores de aprendizaje automático y de aprendizaje profundo y concluyeron que los clasificadores de aprendizaje profundo ofrecen un rendimiento superior.

Las notas clínicas son utilizadas por los autores en para detectar estados mentales alterados. Los autores utilizaron la clasificación internacional de códigos de enfermedades para etiquetar las notas. Además, utilizaron el clasificador de aprendizaje profundo CNN para identificar los casos a partir de informes de EHR. De manera similar, en, los investigadores utilizaron técnicas de procesamiento del lenguaje natural para detectar el comportamiento suicida de mujeres embarazadas a partir de EHR. En este estudio, los investigadores utilizaron términos relacionados con el comportamiento suicida para detectar los casos. En nuestro estudio, consideramos palabras relevantes para detectar eficazmente los casos de salud conductual.

El trabajo mencionado anteriormente utiliza datos textuales, aunque relacionados con los datos de la narrativa pública de los socorristas, que son significativamente diferentes de los de las redes sociales o los informes de EHR. El patrón lingüístico en el informe narrativo público no estructurado de los socorristas es un desafío para los marcos tradicionales de aprendizaje profundo. El trabajo revolucionario de Vaswani et al. en "Attention is All You Need" introdujo la arquitectura Transformer, que se basa en gran medida en el mecanismo de atención. El mecanismo de atención puede manejar secuencias más largas en comparación con CNN o RNN. Además, al ponderar adecuadamente el contexto del embedding, ayuda a centrarse en la parte importante del texto. Además, los investigadores en demostraron que la atención de producto escalar escalado muestra un rendimiento prometedor en la clasificación de datos textuales. Inspirados por el rendimiento probado de los mecanismos de atención, lo utilizamos en nuestro marco de atención multinivel para detectar la salud conductual.

### III. METODOLOGÍA

En esta sección, discutiremos nuestro marco propuesto para detectar diferentes casos de salud conductual. La Figura 1 muestra la descripción general de alto nivel de nuestro marco de detección de salud conductual propuesto. Nuestro informe narrativo público preprocesado pasa por el clasificador de salud conductual multinivel habilitado por atención, donde se analizan los casos para determinar diferentes casos de salud conductual. En nuestro marco propuesto, utilizamos diferentes niveles (palabra, oración y documento) de datos para identificar casos de salud conductual.

#### A. Preprocesamiento y Limpieza de Datos

El preprocesamiento de datos es una de las partes más cruciales de cualquier experimento de aprendizaje profundo. Eliminamos los espacios en blanco redundantes, los signos y símbolos innecesarios del conjunto de datos. Además, ponemos las palabras en minúsculas para mantener la coherencia. Además, transformamos las etiquetas a un formato binario utilizando MultiLabelBinarizer, asegurando que las etiquetas estén correctamente formateadas para la tarea de clasificación multi-etiqueta.

#### B. Clasificador de Salud Conductual Multinivel Habilitado por Atención

La capa del clasificador habilitado por atención recibe el informe narrativo público como entrada y lo analiza para determinar diferentes tipos de casos de salud conductual.
**Embedding:** La computadora entiende valores numéricos, específicamente 0 y 1. Inicialmente, el conjunto de datos pasa por un embedding para convertir el texto de entrada en vectores numéricos.

---
### Página 3



**Descripción de la Figura 1:** La figura muestra un diagrama de flujo titulado "Marco de Detección de Salud Conductual". El proceso comienza con "Informe Narrativo Público" que pasa por "Preprocesamiento de Datos". El siguiente paso es "Embedding". A partir de ahí, el flujo se divide en dos rutas paralelas. La ruta superior consta de "Extracción de Características Contextuales", "Representación Automática de Oraciones Contextuales" y "Representación Comprensiva del Documento". La ruta inferior consiste en "Representación Ponderada de Palabras" y "Representación Vectorial Habilitada por Consulta". Ambas rutas convergen en "Acumulación de Características", que luego alimenta al "Clasificador Multi-etiqueta" para producir la "Salida".

En la ecuación 1, la función de incrustación `Ek` genera la representación numérica para el informe narrativo público.

`Ek(K) = {ek,1, ek,2, ..., ek,n}` (1)

**Extracción de características contextuales:** LSTM bidireccional (Bi-LSTM) es una versión modificada del modelo LSTM. El modelo LSTM tenía las limitaciones de analizar secuencias desde una sola dirección. Mientras que el modelo Bi-LSTM analiza las secuencias desde ambas direcciones para capturar un contexto más completo. Los autores en utilizaron Bi-LSTM para la extracción de características y demostraron su efectividad. En nuestro marco propuesto, utilizamos Bi-LSTM para extraer características contextuales por su probada efectividad.

`hk,1 = L(ek,t, hk-1,1; θ→)`
`hk,1 = L(ek,t, hk+1,1; θ←)`
`hk,1 = [hk,1; hk,1]` (2)

Para capturar el contexto secuencial y las dependencias, aplicamos Bi-LSTM a la representación incrustada `Ek` que se muestra en la ecuación 2. De la ecuación, podemos ver que la capa Bi-LSTM produce el estado oculto utilizando dos LSTM desde ambas direcciones.

**Representación Automática de Oraciones Contextuales:** La representación de oraciones nos permite encapsular el significado semántico de una oración. La representación automática de oraciones contextuales se refiere al método automático que utilizamos para obtener la información contextual de la oración. Los investigadores establecieron la efectividad de las operaciones de agrupación (pooling) para obtener la representación de la oración. En nuestro marco propuesto, obtenemos la representación de la oración `Srep` aplicando una agrupación promedio sobre las características contextuales extraídas `hk,1`, lo que resulta en un único vector que encapsula la esencia de la oración.

**Representación Comprensiva del Documento:** La representación del documento es esencial para una comprensión integral del contexto general. En nuestro marco, para obtener la representación del documento `drep`, empleamos LSTM en la representación de la oración `Srep` seguido de una capa de agrupación promedio. Este método convierte la representación de la oración en una representación vectorial que captura la importancia de todo el documento.

**Representación Ponderada de Palabras:** Comprender las palabras importantes es crucial para entender el contexto general. Los investigadores en, demostraron la efectividad de utilizar la auto-atención para encontrar la porción importante de una secuencia. Para mejorar el enfoque de nuestro modelo en palabras importantes y relevantes, utilizamos un mecanismo de auto-atención modificado. En el mecanismo de auto-atención, se asigna un peso a diferentes palabras según su importancia; utilizamos un umbral determinado empíricamente alfa (α) en el mecanismo de auto-atención para obtener la representación de palabras basada en la atención `Watt` a partir de `hk,1`. En la ecuación 3, los vectores `WQ`, `WK` y `WV` representan los vectores de clave, consulta y valor, respectivamente.

`Watt = α * ((hk,1 * WQ) * (hk,1 * WK)T / sqrt(dk)) * hk,1 * WV` (3)

Estas palabras (`Watt`) son cruciales para comprender el contexto de los casos de salud conductual.

**Representación Vectorial Habilitada por Consulta:** El mecanismo de atención cruzada (cross-attention) nos ayuda a centrarnos en porciones importantes de una secuencia integrando información de diferentes fuentes. Los autores en mostraron la efectividad de la atención cruzada con extensos experimentos. Inspirados en trabajos previos, en nuestro marco, obtuvimos una representación vectorial habilitada por consulta empleando atención cruzada.

`Aij = e^(srep * wattT / sqrt(dk)) / Σe^(srep * wattT / sqrt(dk))` (4)
`vecqtt = A · Watt`

La representación ponderada de palabras `Watt` proporciona palabras importantes que contienen la información local. Además, la representación de la oración `Srep` captura la representación contextual desde una perspectiva más amplia. En la ecuación 4, utilizamos `Watt` como consulta en la representación de la oración `Srep` para obtener un vector habilitado por consulta que es contextualmente robusto.

---
### Página 4

**Acumulación de Características:** En esta capa, obtenemos la representación comprensiva del documento `drep` y la representación vectorial habilitada por consulta `vecqtt`. Posteriormente, las combinamos para obtener un vector de representación global `concatrep`.

`concatrep = [vecqtt; drep]` (5)

Este vector global contiene la representación que encapsula el contexto del informe basado en el vector de características atendido por consulta `vecatt` y la representación del documento `drep`. Después, el vector global se pasa a la siguiente capa del clasificador multi-etiqueta para clasificar los casos de salud conductual.

**Clasificador Multi-etiqueta:** En esta capa, obtenemos el vector de características acumulado `concatrep` de la capa de acumulación de características. La distribución inter e intra-clase se aprende del vector `vecqtt` aplicando la capa de avance (feed-forward). Además, la salida de la red de avance pasa a través de la capa sigmoide para clasificar los casos de salud conductual multi-etiqueta.

### IV. EVALUACIÓN EXPERIMENTAL

En esta sección, discutiremos los detalles de nuestra evaluación experimental.

#### A. Descripción del Conjunto de Datos

En esta sección, discutiremos el conjunto de datos que se utiliza para llevar a cabo nuestro estudio. Utilizamos datos de las narrativas públicas de los socorristas (policía) para realizar nuestro experimento. Antes de utilizar los datos para el estudio, realizamos un proceso completo de desidentificación para eliminar toda la información de identificación personal. La eliminación de dicha información garantiza la privacidad y ayuda a que nuestro modelo permanezca imparcial. Después de realizar la desidentificación, entregamos los datos a nuestros anotadores expertos para su anotación. Nuestros anotadores expertos anotaron un total de 1050 informes relacionados con la salud conductual. Además, muestreamos 99 casos para dos anotadores y encontramos que su acuerdo general fue de alrededor del 90.9%. Según las estadísticas kappa, un acuerdo del 90.9% indica que nuestro anotador tiene un acuerdo casi perfecto. Las estadísticas de nuestro conjunto de datos anotado se muestran en la Tabla I.

**TABLA I: Estadísticas del Informe Narrativo Público de los Primeros Respondedores**
| Métrica | Cantidad |
| :--- | :--- |
| Recuento Total de Oraciones | 5,693 |
| Promedio de Oraciones por Informe | 19.10 |
| Longitud Promedio de la Oración | 22.07 |
| Recuento Total de Palabras | 123,591 |
| Promedio de Palabras por Informe | 414.73 |

Los diferentes tipos de clases de salud conductual en nuestro conjunto de datos se discuten en la Tabla II. La distribución de las diferentes clases de casos de salud conductual en nuestro conjunto de datos se muestra en la Figura 2. Al analizar la figura, podemos ver que la categoría más grande es la **Doméstica/Social**; esta distribución indica la importancia de abordar los factores domésticos/sociales en la investigación de la salud conductual. Se observa un número moderado de casos en las categorías de **Abuso de Sustancias** y **Salud Mental**. Observamos menos casos en la categoría **Crisis de Salud Mental y Abuso de Sustancias**; sin embargo, es importante señalar que el conjunto de datos incluye instancias multi-etiqueta.

**TABLA II: Tipos de Clases de Salud Conductual**
| Categoría | Descripción |
| :--- | :--- |
| **Salud Mental** | Esta categoría incluye casos que involucran a individuos con trastornos de salud mental diagnosticables como depresión, trastorno bipolar o trastorno de ansiedad. |
| **Abuso de Sustancias** | Esta categoría incluye casos con individuos que tienen problemas de abuso de alcohol/drogas. |
| **Doméstico/Social** | Esta categoría incluye casos donde se exhiben anormalidades de comportamiento entre dos o más personas en un entorno doméstico. |
| **No Doméstico/Social** | En esta categoría, se incluyen incidentes o actividades criminales fuera del hogar y principalmente entre extraños. |
| **Crisis de Salud Mental y Abuso de Sustancias** | Esta categoría incluye casos donde individuos experimentan una crisis de salud mental severa, junto con problemas de abuso de sustancias, que requieren intervención inmediata. Estos individuos pueden estar en alto riesgo de sobredosis, autolesión o daño a otros. |



**Descripción de la Figura 2:** La figura es un gráfico de barras que muestra la distribución de casos en diferentes categorías de salud conductual. El eje X representa las "Clases" y el eje Y el "Número de Instancias". La clase "Doméstico/Social" tiene el mayor número de instancias (más de 600). "Abuso de Sustancias" y "Salud Mental" tienen un número moderado de instancias (alrededor de 300-400). "No Doméstico/Social" tiene menos instancias, y "Crisis de Salud Mental y Abuso de Sustancias" tiene el menor número de instancias (alrededor de 100).

En el conjunto de datos, un solo caso puede pertenecer a múltiples categorías. Este etiquetado múltiple nos proporciona una comprensión más realista de los desafíos que enfrentan las personas con problemas de salud conductual.

#### B. Métricas de Evaluación

Para evaluar la robustez de nuestro estudio, utilizamos varias métricas de evaluación, incluyendo Precisión, Recall, F1-score y Exactitud. La Exactitud es la relación entre las predicciones correctas (verdaderos positivos y verdaderos negativos) y el número total de predicciones. La Precisión es la relación entre los verdaderos positivos y el número total de predicciones positivas. El Recall es la relación entre los verdaderos positivos y el número total de instancias positivas reales. El F1-score se conoce como la media armónica entre precisión y recall. Proporciona un equilibrio entre la puntuación de recall y la de precisión.

#### C. Detalles de Implementación

Utilizamos el lenguaje de programación Python para desarrollar nuestro marco avanzado de NLP. Además, utilizamos la biblioteca Keras con el backend de TensorFlow para crear nuestro modelo novedoso. Nuestro conjunto de datos se divide en una proporción de 70:30 para fines de entrenamiento y prueba. En la capa de red feed-forward de nuestro modelo propuesto, utilizamos ReLu como función de activación. Usamos el optimizador Adam con 0.0001 como tasa de aprendizaje para construir nuestro modelo. La Entropía Cruzada Binaria es la función de pérdida empleada con 32 como tamaño de lote. Además, para evaluar...

---
### Página 5

**TABLA III: Modelos y su Exactitud Correspondiente**

| Modelos | Exactitud |
| :--- | :--- |
| LSTM | 33% |
| GRU | 31% |
| Bi-LSTM | 35% |
| CNN | 53% |
| BERT | 51% |
| RoBERTa | 46% |
| **Modelo Propuesto** | **71%** |

la robustez y fiabilidad de nuestro modelo propuesto, realizamos una validación cruzada de 5 pliegues. Además, utilizamos GPUs de NVIDIA (A4500) para llevar a cabo el experimento.

### V. RESULTADOS EXPERIMENTALES

En esta sección, vamos a discutir nuestros resultados experimentales. Primero evaluamos y comparamos el rendimiento de nuestro modelo propuesto con otros modelos de referencia. Además, evaluamos el impacto de diferentes parámetros en el rendimiento de nuestro modelo.

#### A. Rendimiento del Modelo de Última Generación

Evaluamos el rendimiento de nuestro marco de detección de salud conductual multinivel habilitado para la atención. Además, comparamos el rendimiento de nuestro modelo propuesto con modelos de aprendizaje profundo como Bi-LSTM, LSTM, GRU y CNN. Para nuestro modelo propuesto, realizamos una validación cruzada de 5 pliegues para evaluar su fiabilidad.

La Tabla III demuestra la precisión de todos los modelos que utilizamos en nuestro experimento. En la tabla, podemos ver que los modelos de aprendizaje profundo de referencia LSTM, GRU, Bi-LSTM y CNN alcanzan una precisión que va del 31% al 53%. Además, utilizamos BERT y RoBERTa, dos modelos basados en transformadores en nuestro experimento, para analizar y comparar su efectividad con nuestro modelo propuesto. La tabla muestra que BERT y RoBERTa alcanzaron una precisión del 51% y 46%, respectivamente. En contraste, nuestro modelo propuesto supera a otros modelos al lograr una precisión del 71%. En nuestro experimento, el bajo rendimiento de los modelos basados en transformadores podría deberse a su complejidad. El resultado indica que incluso el modelo avanzado RoBERTa tuvo un peor rendimiento que BERT, lo que sugiere que una mayor complejidad no siempre produce resultados si no se alinea bien con los datos. Sin embargo, nuestro modelo propuesto utilizó datos de múltiples niveles para obtener una visión más completa y lograr un mejor rendimiento.

#### B. Rendimiento del Modelo por Clase

En esta sección, vamos a analizar el rendimiento de nuestro modelo propuesto para detectar diferentes clases de casos de salud conductual. La Figura 3 muestra el rendimiento de nuestro modelo de salud conductual multinivel habilitado para la atención para detectar diferentes clases de salud conductual. Al analizar la figura, podemos ver que nuestro modelo mostró una excelente puntuación F1 (88%) para detectar la clase Doméstico/Social, lo que demuestra la capacidad superior de nuestro modelo para identificar las características distintivas de esta clase. Sin embargo, el modelo tuvo dificultades para detectar las instancias de las clases Crisis de Salud Mental y Abuso de Sustancias. El rendimiento subóptimo de esta clase podría atribuirse al hecho de que el número de muestras para esta clase era muy limitado en los datos de entrenamiento. Además, las características superpuestas de múltiples clases dificultaron que el modelo diferenciara las instancias.



**Descripción de la Figura 3:** Este es un gráfico de barras que compara las métricas de Precisión, Recall y F1-Score para cinco clases diferentes de salud conductual. Para la clase "Doméstico/Social", todas las métricas son altas, especialmente el F1-Score (cerca de 0.9). Para "Abuso de Sustancias", la precisión es alta (0.8) pero el recall y el F1-Score son más bajos. Para "Salud Mental", las métricas son moderadas. "No Doméstico/Social" y "Crisis de Salud Mental y Abuso de Sustancias" tienen las métricas más bajas en general.

No obstante, nuestro modelo mostró un gran potencial para detectar las instancias de la clase Abuso de Sustancias con alta precisión (80%). La investigación futura puede centrarse en mejorar el rendimiento subóptimo de las clases complejas. En general, nuestro modelo demostró un rendimiento prometedor en la identificación de diferentes clases de salud conductual.

#### C. Impacto del Ruido de Datos en el Rendimiento

Evaluamos la robustez de nuestro modelo midiendo el rendimiento en niveles variados de datos ruidosos. Para realizar este experimento, introdujimos datos mal etiquetados en el conjunto de entrenamiento de forma incremental en un rango del 10% al 80% y observamos el rendimiento.



**Descripción de la Figura 4:** Este es un gráfico de líneas que muestra el impacto del ruido en la precisión del modelo. El eje X representa el "Porcentaje de Etiquetas Erróneas" (de 10% a 80%), y el eje Y representa la "Precisión". La precisión del modelo comienza por encima de 0.6 para un 10% de ruido y disminuye constantemente a medida que aumenta el porcentaje de etiquetas erróneas, cayendo por debajo de 0.2 con un 80% de ruido.

La Figura 4 muestra el rendimiento de nuestro modelo propuesto en diferentes niveles de ruido. Al observar el gráfico, podemos ver que nuestro modelo de salud conductual multinivel habilitado para la atención funcionó relativamente bien en niveles moderados (10% a 30%) de ruido. Sin embargo, con niveles de ruido más altos (40% a 80%), el modelo no pudo funcionar de manera óptima; esto podría atribuirse al hecho de que, debido al alto nivel de ruido, el modelo no tenía una cantidad suficiente de datos de entrenamiento para obtener información. Esto demuestra que nuestro modelo multinivel habilitado para la atención fue resistente para comprender y obtener...

---
### Página 6

...conocimientos del contexto de los informes a pesar de tener un nivel moderado de ruido.

#### D. Impacto de Alfa en el Rendimiento

En nuestro marco de detección de salud conductual multinivel habilitado por atención, alfa (α) es utilizado por el mecanismo de auto-atención para encontrar palabras importantes relevantes para determinar el contexto de la salud conductual.



**Descripción de la Figura 5:** Este es un gráfico de líneas que muestra la relación entre el "Valor Alfa" (eje X) y la "Precisión Media" (eje Y). La precisión media fluctúa a medida que cambia el valor de alfa, alcanzando un pico claro cuando alfa es 0.5, con una precisión media de aproximadamente 0.71. Para valores de alfa inferiores o superiores a 0.5, la precisión disminuye.

Para un rendimiento eficiente del modelo, determinar un valor óptimo para alfa es muy importante. Determinamos el valor óptimo para alfa cambiando el valor y observando su efecto en la precisión media del modelo. La Figura 5 muestra la relación entre el valor de alfa y la precisión media del modelo. En la figura, podemos ver que el valor óptimo para alfa es 0.5; cualquier valor inferior o superior al valor óptimo causa una degradación en el rendimiento. Una posible razón para esto podría ser que para un valor de alfa más bajo, el marco no tiene suficientes palabras relevantes, y para un valor de alfa más alto, el modelo podría tender a sobreajustarse.

### VI. CONCLUSIÓN

En este estudio, propusimos un novedoso marco multi-etiqueta habilitado por atención para detectar casos de salud conductual a partir de las narrativas públicas de los socorristas (policía). En nuestro modelo propuesto, utilizamos mecanismos de auto-atención y atención cruzada y utilizamos características de tres niveles diferentes (palabra, oración y documento) del informe para obtener conocimientos contextuales para detectar casos de salud conductual. Nuestro modelo propuesto superó a los modelos de aprendizaje profundo y basados en transformadores de última generación con una precisión del 71%. Aunque nuestro modelo propuesto mostró un rendimiento prometedor, todavía hay margen para mejorar; en nuestro trabajo futuro, nuestro objetivo es mejorar el rendimiento de nuestro modelo. Además, planeamos incorporar a los humanos en el enfoque "human-in-the-loop" para integrar el conocimiento de los expertos en el desarrollo de un marco mejorado. Creemos que nuestra investigación contribuiría al desarrollo de sistemas automatizados que puedan identificar casos de salud conductual de manera eficiente, permitiendo intervenciones más oportunas y apoyo por parte de los socorristas.

### REFERENCIAS

 American Medical Association. What is behavioral health?, Aug 2022.
 Mental Health America. The state of mental health in america.
 National Institute of Mental Health. Mental health information.
 Drew. 3 reasons for the nationwide police shortage article, Feb 2024.
 Marcel Trotzek, Sven Koitka, and Christoph M. Friedrich. Utilizing neural networks and linguistic metadata for early detection of depression indications in text sequences. IEEE Transactions on Knowledge and Data Engineering, 32(3):588–601, 2020.
 Pratyaksh Jain1, Karthik Ram Srinivas, and Abhishek Vichare. Depression and suicide analysis using machine learning and nlp.
 Akshi Kumar, Aditi Sharma, and Anshika Arora. Anxious depression prediction in real-time social data, May 2019.
 Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, and Antoine Bordes. Supervised learning of universal sentence representations from natural language inference data, 2018.
 Tuhin UniversityNew Chakrabarty and Kilol Gupta. Context-aware attention for understanding twitter abuse, 2018.
 Jina Kim, Jieon Lee, Eunil Park, and Jinyoung Han. A deep learning model for detecting mental illness from user content on social media, Jul 2020.
 Substance Abuse and Mental Health Services Administration (SAMHSA). Samhsa behavioral health integration.
 Chen H.-H.; Huang H.-H.; Wang Y.-T. A neural network approach to early risk detection of depression and anorexia on social media text, Jan 2018.
 Jihad S. Obeid, Erin R. Weeda, Andrew J. Matuskowitz, Kevin Gagnon, Tami Crawford, Christine M. Carr, and Lewis J. Frey. Automated detection of altered mental status in emergency department clinical notes: A deep learning approach - bmc medical informatics and decision making, Aug 2019.
 Qiu-Yue Zhong, Elizabeth W. Karlson, Bizu Gelaye, Sean Finan, Paul Avillach, Jordan W. Smoller, Tianxi Cai, and Michelle A. Williams. Screening pregnant women for suicidal behavior in electronic medical records: Diagnostic codes vs. clinical notes processed by natural language processing - bmc medical informatics and decision making, May 2018.
 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.
 Xiaobing Sun and Wei Lu. Understanding attention for text classification. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3418–3428, Online, July 2020. Association for Computational Linguistics.
 Scikit Learn Team. Multilabelbinarizer.
 Zhiheng Huang, Wei Xu, and Kai Yu. Bidirectional lstm-crf models for sequence tagging, 2015.
 Yao Chen, Changjiang Zhou, Tianxin Li, Hong Wu, Xia Zhao, Kai Ye, and Jun Liao. Named entity recognition from chinese adverse drug event reports with lexical feature based bilstm-crf and tri-training. Journal of Biomedical Informatics, 96:103252, 2019.
 Mehak Khan, Hongzhi Wang, Adnan Riaz, Aya Elfatyany, and Sajida Karim. Bidirectional lstm-rnn-based hybrid deep learning frameworks for univariate time series classification - the journal of supercomputing, Jan 2021.
 Ning Ding, Liangrui Peng, Changsong Liu, Yuqi Zhang, Ruixue Zhang, and Jie Li. Incorporating self-attention mechanism and multi-task learning into scene text detection. In Leonid Karlinsky, Tomer Michaeli, and Ko Nishino, editors, Computer Vision – ECCV 2022 Workshops, pages 314–328, Cham, 2023. Springer Nature Switzerland.
 Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations, 2018.
 Yuanhang Yang, Shiyi Qi, Chuanyi Liu, Qifan Wang, Cuiyun Gao, and Zenglin Xu. Once is enough: A light-weight cross-attention for fast sentence pair modeling. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2800–2806, Singapore, December 2023. Association for Computational Linguistics.
 Keras Team. Keras. https://https://keras.io/.
 Tensorflow Team. Tensorflow. https://www.tensorflow.org/.