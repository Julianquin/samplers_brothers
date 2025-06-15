### Estrategias Definitivas para Muestreo Representativo de Series Temporales

---

### 1. **Cluster Prototype Sampling (A)**
**Descripción:**  
Para cada uno de los 300 clústeres obtenidos por GASF + Spectral Clustering:
- Selecciona el centroide o medoid (la serie más representativa).
- Opcional: añade series adicionales (√n o log(n)) para mayor diversidad.

**Ventajas:**  
- Simple y altamente interpretable.
- Cubre patrones principales.

**Limitaciones:**  
- Puede perder variabilidad interna en clusters grandes.

---

### 2. **Stratified Proportional Sampling + Rare Oversampling (L)**
**Descripción:**  
- Usa tamaño de cada clúster para asignar proporción de series.
- Estratificación secundaria por volumen (percentiles 0-30 bajo, 30-70 medio, 70-100 alto).
- Sobremuestreo (2x) de estratos minoritarios (menos del 5% del clúster).
- Inclusión obligatoria de outliers de volumen (> Q3 + 1.5·IQR) y formas atípicas (bordes).

**Ventajas:**  
- Representa distribución real y refuerza patrones minoritarios.
- Asegura cobertura tanto de volumen como de rarezas.

---

### 3. **Diversity-Weighted Cluster Sampling (K)**
**Descripción:**  
Para cada clúster:
- 2 series más cercanas al centroide (patrón típico).
- 2 series con máxima distancia al centroide (bordes del clúster).
- 2 series con mayor volumen anual.
- 1 serie que sea outlier extremo en volumen (si existe, o siguiente más alta).
- Ajuste proporcional adicional si un estrato de volumen es demasiado pequeño.

**Ventajas:**  
- Equilibrio óptimo entre forma típica, atípica y volumen estratégico.
- Minimiza redundancia.

---

### 4. **Active–Iterative Hard-Case Sampling (M)**
**Descripción:**  
- Comienza con una selección inicial representativa (e.g., estrategia Shape-Volume Prototype).
- Entrena un modelo ligero inicial (p.ej., LSTM pequeño).
- Evalúa sobre series no incluidas, identifica aquellas con errores altos ("casos difíciles").
- Añade iterativamente estas series difíciles hasta alcanzar una muestra óptima.

**Ventajas:**  
- Enfoque adaptativo, centrado en series difíciles para el modelo.
- Maximiza cobertura efectiva del aprendizaje.

**Limitaciones:**  
- Requiere capacidad computacional para iteraciones rápidas.

---

### 5. **Hybrid Full Coverage Sampling (N)**
**Descripción:**  
- Combina:
  - 50% desde Diversity-Weighted Cluster Sampling (K).
  - 25% desde Stratified Proportional + Rare Oversampling (L).
  - 25% desde Active-Iterative Hard-Case Sampling (M).
- Ajuste final: asegura cubrir al menos el 30% del volumen global añadiendo series de alto volumen si es necesario.

**Ventajas:**  
- Máxima seguridad en la cobertura de patrones temporales y estratégicos de volumen.

---

### 6. **Hybrid Stratified Volume Sampling (O)**
**Descripción:**  
- Para cada clúster divide en 3 estratos por volumen global (bajo <30%, medio 30-70%, alto >70%).
- Selección proporcional según tamaño real del estrato en cada clúster (5 series por clúster como estándar).
- Inclusión obligatoria de series con volumen >10x la media global.

**Ventajas:**  
- Excelente equilibrio forma-volumen.
- Priorización clara de series con alta importancia estratégica.

---

### 7. **Diversified Sampling with Volume Adjustment (P)**
**Descripción:**  
Para cada clúster:
- Selecciona la serie más cercana al centroide (patrón típico).
- Selecciona las 2 series más alejadas del centroide (patrones atípicos).
- Selecciona la serie de mayor volumen absoluto.
- Ajusta hasta cubrir al menos el 70% del volumen global añadiendo series de alto volumen si es necesario.

**Ventajas:**  
- Balance entre representatividad de patrones y cobertura estratégica de volumen.
- Minimiza redundancias.

---

### 8. **Active Hybrid Sampling (Q)**
**Descripción:**  
- Inicial: toma una muestra representativa inicial basada en cluster prototypes.
- Iterativo: entrena un modelo proxy, identifica series difíciles (alto error/incertidumbre).
- Prioriza en la selección iterativa aquellas series difíciles con mayor volumen.
- Inclusión obligatoria de series outliers de volumen (>10x media global).

**Ventajas:**  
- Proceso iterativo enfocado en mejorar continuamente el aprendizaje del modelo.
- Fuerte orientación estratégica hacia series de volumen importante y difíciles de predecir.

---

### 9. **Stratified Proportional Sampling with Volume Reinforcement (R)**
**Descripción:**  
- Asignación de cuotas proporcionales al tamaño del clúster.
- Sobremuestreo de clústeres muy pequeños (<1% total).
- Refuerzo: reemplaza el 20% de series seleccionadas con las series de mayor volumen.
- Inclusión obligatoria de series outliers en volumen (>10x media global).

**Ventajas:**  
- Estrategia estadísticamente robusta.
- Combina representatividad de distribución de patrones con atención prioritaria a series estratégicas por volumen.

