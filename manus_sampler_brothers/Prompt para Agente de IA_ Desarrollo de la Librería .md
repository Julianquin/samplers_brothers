### **Prompt para Agente de IA: Desarrollo de la Librería `samplers_brothers`**

**Rol:** Eres un desarrollador Python senior especializado en librerías de data science y machine learning. Tu tarea es escribir código limpio, modular, bien documentado y eficiente.

**Objetivo General:** Desarrollar una librería en Python llamada `samplers_brothers` que implemente un conjunto de estrategias de muestreo para series de tiempo, basadas en resultados de clustering preexistentes. La librería debe permitir a un usuario seleccionar y aplicar fácilmente cualquiera de las estrategias definidas.

---

### **1. Contexto del Problema a Resolver**

La librería operará sobre un conjunto de datos de aproximadamente 35,000 series de tiempo. Estas series ya han sido procesadas y agrupadas en 300 clústeres mediante técnicas de embedding (GASF) y clustering (Spectral Clustering). Cada serie temporal tiene asociado un "patrón" (definido por su clúster) y un "volumen" (su magnitud o importancia estratégica). El objetivo es seleccionar un subconjunto representativo para entrenar modelos de forecasting.

---

### **2. Requisitos Funcionales y Estructura del Código**

1.  **Librería/Clase Principal:** Crea una clase principal llamada `TimeSampler`.
    * El constructor `__init__` debe recibir los datos necesarios para operar:
        * `series_data`: Un DataFrame de pandas donde las columnas son los IDs de las series y las filas son los valores en el tiempo.
        * `cluster_labels`: Un DataFrame o Serie de pandas que mapee cada ID de serie a su número de clúster (de 0 a 299).
        * `features`: Un array de NumPy o DataFrame con las características (embeddings GASF) usadas para el clustering, necesarias para calcular distancias y centroides.
    * La clase debe calcular y almacenar internamente métricas útiles, como el volumen de cada serie (p. ej., la suma o media de sus valores) y los centroides/medoides de cada clúster.

2.  **Método Principal de Muestreo:**
    * Implementa un método público principal: `sample(strategy: str, n_samples: int, **kwargs) -> list`.
    * `strategy`: Un string que identifica la estrategia a usar (p. ej., 'K', 'L', 'M', etc., como se define a continuación).
    * `n_samples`: El número total de series a seleccionar en la muestra.
    * `**kwargs`: Argumentos adicionales específicos para ciertas estrategias (p. ej., un modelo para las estrategias activas).
    * El método debe devolver una lista con los IDs de las series seleccionadas.

3.  **Modularidad:**
    * Cada estrategia de muestreo debe ser implementada como un método privado o protegido dentro de la clase (p. ej., `_sample_k(...)`, `_sample_l(...)`).
    * Utiliza funciones de ayuda (helpers) para tareas repetitivas como calcular medoides, distancias al centroide, outliers de volumen, etc.

---

### **3. Descripción Detallada de las Estrategias a Implementar**

Debes implementar las siguientes estrategias, identificadas por su letra clave.

* **'A' - Cluster Prototype Sampling:** Para cada clúster, selecciona su medoid (la serie real más cercana al centroide del clúster en el espacio de características). Ajusta para que el total sea `n_samples`.

* **'L' - Stratified Proportional Sampling + Rare Oversampling:**
    1.  Asigna cuotas de muestreo a cada clúster proporcionalmente a su tamaño.
    2.  Dentro de cada clúster, estratifica las series por volumen (percentiles: bajo 0-30, medio 30-70, alto 70-100).
    3.  Muestrea de cada estrato de volumen para cumplir la cuota del clúster.
    4.  Aplica un sobremuestreo a estratos minoritarios (aquellos con menos del 5% de las series del clúster).
    5.  Asegura la inclusión de outliers de volumen y forma (series en los bordes del clúster).

* **'K' - Diversity-Weighted Cluster Sampling:** Intenta seleccionar un número fijo de series por clúster (p. ej., 7) y luego sub-muestrea aleatoriamente hasta alcanzar `n_samples`. Para cada clúster, selecciona:
    1.  Las 2 series más cercanas al medoid.
    2.  Las 2 series más lejanas al medoid.
    3.  Las 2 series con mayor volumen total.
    4.  La serie con el volumen más extremo (outlier), si existe.

* **'M' - Active-Iterative Hard-Case Sampling:**
    1.  Empieza con una muestra inicial pequeña usando la estrategia 'A'.
    2.  Requiere un modelo proxy (p. ej., un LSTM simple pasado como argumento en `**kwargs`).
    3.  Entrena el modelo en la muestra actual.
    4.  Evalúa el modelo sobre las series no incluidas en la muestra.
    5.  Añade a la muestra las `k` series con el error de predicción más alto ("casos difíciles").
    6.  Repite hasta alcanzar `n_samples`.

* **'N' - Hybrid Full Coverage Sampling:** Combina los resultados de varias estrategias.
    1.  Selecciona el 50% de `n_samples` usando la estrategia 'K'.
    2.  Selecciona el 25% de `n_samples` usando la estrategia 'L'.
    3.  Selecciona el 25% de `n_samples` usando la estrategia 'M'.
    4.  Elimina duplicados y asegura la cobertura de un umbral de volumen global.

* **'O' - Hybrid Stratified Volume Sampling:** Para cada clúster:
    1.  Divide sus series en 3 estratos de volumen (bajo, medio, alto).
    2.  Muestrea proporcionalmente del tamaño de cada estrato dentro del clúster.
    3.  Asegura la inclusión obligatoria de series cuyo volumen supere 10 veces la media de volumen global.

* **'P' - Diversified Sampling with Volume Adjustment:** Para cada clúster, selecciona:
    1.  La serie más cercana al medoid.
    2.  Las 2 series más alejadas.
    3.  La serie de mayor volumen.
    4.  Después de recorrer todos los clústeres, si no se cubre el 70% del volumen total de todas las series, sigue añadiendo las series de mayor volumen no seleccionadas hasta alcanzarlo.

* **'Q' - Active Hybrid Sampling:** Similar a la 'M', pero al seleccionar los "casos difíciles", prioriza aquellos que tienen un alto error de predicción Y un alto volumen. Incluye obligatoriamente los outliers de volumen extremo.

* **'R' - Stratified Proportional Sampling with Volume Reinforcement:**
    1.  Realiza un muestreo proporcional al tamaño del clúster.
    2.  Sobremuestrea clústeres pequeños (<1% del total).
    3.  Reemplaza el 20% de las series seleccionadas (las de menor volumen) por las series de mayor volumen global que no fueron seleccionadas.

---

### **4. Ejemplo de Uso Esperado**

El usuario debería poder interactuar con la librería de la siguiente manera:

```python
import pandas as pd
from samplers_brothers import TimeSampler

# Cargar datos (ejemplos)
# series_data: DataFrame con series temporales (columnas=series_id)
# cluster_labels: DataFrame con mapeo (columnas='series_id', 'cluster_id')
# features: NumPy array con los embeddings de cada serie
series_data = pd.read_csv("series_data.csv", index_col=0)
cluster_labels = pd.read_csv("cluster_labels.csv")
features = pd.read_csv("series_features.csv").values

# 1. Inicializar el sampler
sampler = TimeSampler(
    series_data=series_data,
    cluster_labels=cluster_labels,
    features=features
)

# 2. Aplicar una estrategia de muestreo
# Estrategia K para obtener 500 muestras
sample_k = sampler.sample(strategy='K', n_samples=500)
print(f"Muestra con estrategia K: {len(sample_k)} series")
print(sample_k[:10])

# Estrategia L para obtener 500 muestras
sample_l = sampler.sample(strategy='L', n_samples=500)
print(f"Muestra con estrategia L: {len(sample_l)} series")
print(sample_l[:10])

# Estrategia activa Q (requeriría un modelo)
# from my_proxy_model import SimpleLSTM
# model = SimpleLSTM()
# sample_q = sampler.sample(strategy='Q', n_samples=500, model_proxy=model)
# print(f"Muestra con estrategia Q: {len(sample_q)} series")
# print(sample_q[:10])

```

### **5. Consideraciones Adicionales**

* **Documentación:** Incluye docstrings claras y completas para la clase y todos los métodos públicos, explicando qué hace cada uno, sus parámetros y lo que devuelve.
* **Dependencias:** La librería dependerá principalmente de `pandas`, `numpy` y `scikit-learn` (para cálculos de distancia, etc.).
* **Manejo de Errores:** Implementa validaciones para asegurar que los datos de entrada tienen el formato correcto y que los nombres de las estrategias son válidos.
* **Rendimiento:** El código debe ser eficiente, especialmente los cálculos de distancia y la búsqueda en DataFrames grandes. Utiliza operaciones vectorizadas de NumPy y Pandas siempre que sea posible.

