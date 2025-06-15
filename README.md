# samplers_brothers

This package implements several strategies for sampling time series based on precomputed clustering results. Use `TimeSampler` to build custom subsets of your data for forecasting models.

## Basic Usage

```python
import pandas as pd
from samplers_brothers import TimeSampler

# ``df`` must contain ``key``, ``dt_fecha``, ``target`` and ``cluster``.
df = pd.read_csv("series_long_format.csv")

sampler = TimeSampler(df)
sample_k = sampler.sample(strategy="K", n_samples=500)
print(sample_k[:10])
```



## **Contexto del Problema: Muestreo Representativo de Series de Tiempo para Forecasting con Deep Learning**

### **Antecedentes**

En numerosos entornos empresariales e industriales, la disponibilidad de datos de series de tiempo ha crecido de forma exponencial. Empresas de retail, logística, energía y finanzas generan y almacenan diariamente miles o incluso millones de series de tiempo que reflejan la evolución temporal de variables clave como ventas, demanda, consumos, precios, incidencias, entre otros.

El análisis predictivo de estas series, especialmente mediante técnicas modernas de aprendizaje profundo (deep learning), permite anticipar comportamientos futuros y mejorar la toma de decisiones. Sin embargo, el entrenamiento de modelos avanzados sobre el universo completo de series resulta prohibitivo en términos de coste computacional y tiempo de cómputo. Además, muchas de estas series pueden ser altamente redundantes (mostrar patrones de comportamiento muy similares) o poco representativas de los fenómenos más relevantes para el negocio.

### **Problema Específico**

En nuestro caso de estudio, **disponemos de aproximadamente 35,000 series de tiempo diarias, cada una con 5 años de historia**. Entrenar directamente modelos neuronales sobre todas las series implica un coste computacional que supera los recursos razonables del proyecto.

Por lo tanto, **es necesario diseñar una metodología de muestreo** que nos permita seleccionar un subconjunto de series de tiempo que sea:

* Representativo de la diversidad de patrones presentes en el universo de series.
* Suficientemente diverso para capturar tanto las formas típicas como atípicas de evolución temporal.
* Incluya los diferentes niveles de volumen (series de alta, media y baja magnitud), pues el nivel de la serie afecta la interpretación y el rendimiento del modelo en producción.

### **Desafíos Técnicos y Consideraciones**

* **Reducción de redundancia:** Evitar entrenar el modelo sobre cientos de series con patrones virtualmente idénticos, lo que añade coste pero no información adicional.
* **Captura de diversidad:** Incluir patrones de comportamiento minoritarios o anómalos, que podrían ser relevantes para la robustez del modelo.
* **Escalabilidad:** El pipeline de muestreo debe ser eficiente y aplicable a decenas de miles de series.
* **Representación de niveles de volumen:** Muchas técnicas de clustering (como GASF) normalizan la serie y sólo capturan la *forma*, no el *nivel absoluto*; esto puede dejar fuera del entrenamiento a las series de mayor importancia estratégica por volumen.
* **Validación del muestreo:** El subconjunto seleccionado debe permitir que el modelo generalice y rinda bien tanto en métricas globales como ponderadas por importancia (por ejemplo, error ponderado por volumen).

### **Metodología Inicial**

Hasta ahora, se ha realizado un **clustering avanzado** de las series de tiempo utilizando **GASF (Gramian Angular Summation Field)** como técnica de extracción de características y **Spectral Clustering** como método de agrupamiento. Esto ha permitido organizar las series en **300 clústeres** según la similitud de sus formas temporales.

El reto siguiente es establecer una o varias estrategias de muestreo que, a partir de esta estructura de clústeres, seleccionen las series óptimas para entrenar el modelo, considerando tanto la forma como el volumen absoluto de cada serie.

---

## **Pregunta de Investigación/Trabajo**

> **¿Cuál es la mejor estrategia de muestreo que permite seleccionar un subconjunto de series de tiempo representativo y diverso, en términos tanto de patrones temporales como de volumen, para entrenar modelos neuronales de forecasting de manera eficiente y robusta?**

---

## **Objetivo Específico**

* Definir, implementar y validar un pipeline de muestreo que combine información de forma (patrones temporales, clusters) y volumen (magnitud de la serie), de modo que el subconjunto seleccionado cubra adecuadamente la variabilidad y relevancia del universo de series original.

---

## **Implicaciones**

Un muestreo deficiente puede resultar en:

* Modelos que generalizan mal fuera de la muestra.
* Sobreajuste a patrones mayoritarios o de bajo volumen.
* Sub-representación de series de alta importancia estratégica.
* Uso ineficiente de recursos computacionales.

Por el contrario, una **metodología de muestreo óptima** incrementará la eficiencia y robustez del modelo, reducirá tiempos de entrenamiento y facilitará una generalización adecuada a todo el universo de series.
"""

quiero que analices todas las estrategias y me digas cuales son las estrategias que deberia implementar. Quiero que entiendas todas todas las ideas.

"""

### Estrategias Definitivas para Pruebas  
Basado en el análisis del problema, propongo **9 estrategias combinadas** que integran enfoques de forma (clusters), volumen y diversidad. Cada estrategia aborda los desafíos clave: redundancia, diversidad de patrones, representatividad de volumen y escalabilidad.

---

### Estrategias Definitivas para Muestreo Representativo de Series Temporales

---

### 1. **Cluster Prototype Sampling (A)**

**Descripción:**
Para cada uno de los 300 clústeres obtenidos por GASF + Spectral Clustering:

* Selecciona el centroide o medoid (la serie más representativa).
* Opcional: añade series adicionales (√n o log(n)) para mayor diversidad.

**Ventajas:**

* Simple y altamente interpretable.
* Cubre patrones principales.

**Limitaciones:**

* Puede perder variabilidad interna en clusters grandes.

---

### 2. **Stratified Proportional Sampling + Rare Oversampling (L)**

**Descripción:**

* Usa tamaño de cada clúster para asignar proporción de series.
* Estratificación secundaria por volumen (percentiles 0-30 bajo, 30-70 medio, 70-100 alto).
* Sobremuestreo (2x) de estratos minoritarios (menos del 5% del clúster).
* Inclusión obligatoria de outliers de volumen (> Q3 + 1.5·IQR) y formas atípicas (bordes).

**Ventajas:**

* Representa distribución real y refuerza patrones minoritarios.
* Asegura cobertura tanto de volumen como de rarezas.

---

### 3. **Diversity-Weighted Cluster Sampling (K)**

**Descripción:**
Para cada clúster:

* 2 series más cercanas al centroide (patrón típico).
* 2 series con máxima distancia al centroide (bordes del clúster).
* 2 series con mayor volumen anual.
* 1 serie que sea outlier extremo en volumen (si existe, o siguiente más alta).
* Ajuste proporcional adicional si un estrato de volumen es demasiado pequeño.

**Ventajas:**

* Equilibrio óptimo entre forma típica, atípica y volumen estratégico.
* Minimiza redundancia.

---

### 4. **Active–Iterative Hard-Case Sampling (M)**

**Descripción:**

* Comienza con una selección inicial representativa (e.g., estrategia Shape-Volume Prototype).
* Entrena un modelo ligero inicial (p.ej., LSTM pequeño).
* Evalúa sobre series no incluidas, identifica aquellas con errores altos ("casos difíciles").
* Añade iterativamente estas series difíciles hasta alcanzar una muestra óptima.

**Ventajas:**

* Enfoque adaptativo, centrado en series difíciles para el modelo.
* Maximiza cobertura efectiva del aprendizaje.

**Limitaciones:**

* Requiere capacidad computacional para iteraciones rápidas.

---

### 5. **Hybrid Full Coverage Sampling (N)**

**Descripción:**

* Combina:

  * 50% desde Diversity-Weighted Cluster Sampling (K).
  * 25% desde Stratified Proportional + Rare Oversampling (L).
  * 25% desde Active-Iterative Hard-Case Sampling (M).
* Ajuste final: asegura cubrir al menos el 30% del volumen global añadiendo series de alto volumen si es necesario.

**Ventajas:**

* Máxima seguridad en la cobertura de patrones temporales y estratégicos de volumen.

---

### 6. **Hybrid Stratified Volume Sampling (O)**

**Descripción:**

* Para cada clúster divide en 3 estratos por volumen global (bajo <30%, medio 30-70%, alto >70%).
* Selección proporcional según tamaño real del estrato en cada clúster (5 series por clúster como estándar).
* Inclusión obligatoria de series con volumen >10x la media global.

**Ventajas:**

* Excelente equilibrio forma-volumen.
* Priorización clara de series con alta importancia estratégica.

---

### 7. **Diversified Sampling with Volume Adjustment (P)**

**Descripción:**
Para cada clúster:

* Selecciona la serie más cercana al centroide (patrón típico).
* Selecciona las 2 series más alejadas del centroide (patrones atípicos).
* Selecciona la serie de mayor volumen absoluto.
* Ajusta hasta cubrir al menos el 70% del volumen global añadiendo series de alto volumen si es necesario.

**Ventajas:**

* Balance entre representatividad de patrones y cobertura estratégica de volumen.
* Minimiza redundancias.

---

### 8. **Active Hybrid Sampling (Q)**

**Descripción:**

* Inicial: toma una muestra representativa inicial basada en cluster prototypes.
* Iterativo: entrena un modelo proxy, identifica series difíciles (alto error/incertidumbre).
* Prioriza en la selección iterativa aquellas series difíciles con mayor volumen.
* Inclusión obligatoria de series outliers de volumen (>10x media global).

**Ventajas:**

* Proceso iterativo enfocado en mejorar continuamente el aprendizaje del modelo.
* Fuerte orientación estratégica hacia series de volumen importante y difíciles de predecir.

---

### 9. **Stratified Proportional Sampling with Volume Reinforcement (R)**

**Descripción:**

* Asignación de cuotas proporcionales al tamaño del clúster.
* Sobremuestreo de clústeres muy pequeños (<1% total).
* Refuerzo: reemplaza el 20% de series seleccionadas con las series de mayor volumen.
* Inclusión obligatoria de series outliers en volumen (>10x media global).

**Ventajas:**

* Estrategia estadísticamente robusta.
* Combina representatividad de distribución de patrones con atención prioritaria a series estratégicas por volumen.
