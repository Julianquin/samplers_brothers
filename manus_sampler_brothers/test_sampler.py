import pandas as pd
import numpy as np
from samplers_brothers.sampler import TimeSampler

# Generar datos de ejemplo
num_series = 100
num_timesteps = 50
num_clusters = 10

# series_data
series_data = pd.DataFrame(np.random.rand(num_timesteps, num_series) * 100, columns=[f'series_{i}' for i in range(num_series)])

# cluster_labels
cluster_labels_data = {'series_id': [f'series_{i}' for i in range(num_series)],
                       'cluster_id': np.random.randint(0, num_clusters, num_series)}
cluster_labels = pd.DataFrame(cluster_labels_data)

# features (embeddings GASF simulados)
features = np.random.rand(num_series, 128) # 128 es un tamaño de embedding arbitrario

print("Datos de ejemplo generados:")
print(f"series_data shape: {series_data.shape}")
print(f"cluster_labels shape: {cluster_labels.shape}")
print(f"features shape: {features.shape}")

# 1. Inicializar el sampler
sampler = TimeSampler(
    series_data=series_data,
    cluster_labels=cluster_labels,
    features=features
)
print("\nTimeSampler inicializado exitosamente.")

# 2. Aplicar una estrategia de muestreo
# Estrategia A para obtener 10 muestras
sample_a = sampler.sample(strategy='A', n_samples=10)
print(f"\nMuestra con estrategia A: {len(sample_a)} series")
print(sample_a)

# Estrategia K para obtener 20 muestras
sample_k = sampler.sample(strategy='K', n_samples=20)
print(f"\nMuestra con estrategia K: {len(sample_k)} series")
print(sample_k)

# Estrategia L para obtener 20 muestras
sample_l = sampler.sample(strategy='L', n_samples=20)
print(f"\nMuestra con estrategia L: {len(sample_l)} series")
print(sample_l)

# Estrategia O para obtener 20 muestras
sample_o = sampler.sample(strategy='O', n_samples=20)
print(f"\nMuestra con estrategia O: {len(sample_o)} series")
print(sample_o)

# Estrategia P para obtener 20 muestras
sample_p = sampler.sample(strategy='P', n_samples=20)
print(f"\nMuestra con estrategia P: {len(sample_p)} series")
print(sample_p)

# Estrategia R para obtener 20 muestras
sample_r = sampler.sample(strategy='R', n_samples=20)
print(f"\nMuestra con estrategia R: {len(sample_r)} series")
print(sample_r)

# Estrategia M y Q (requerirían un modelo proxy real, aquí solo simulamos)
# from my_proxy_model import SimpleLSTM # Esto fallaría sin el archivo
class SimpleLSTM:
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.random.rand(X.shape[0]) # Simular predicciones

model = SimpleLSTM()

sample_m = sampler.sample(strategy='M', n_samples=20, model_proxy=model)
print(f"\nMuestra con estrategia M: {len(sample_m)} series")
print(sample_m)

sample_q = sampler.sample(strategy='Q', n_samples=20, model_proxy=model)
print(f"\nMuestra con estrategia Q: {len(sample_q)} series")
print(sample_q)

# Estrategia N para obtener 30 muestras
sample_n = sampler.sample(strategy='N', n_samples=30, model_proxy=model)
print(f"\nMuestra con estrategia N: {len(sample_n)} series")
print(sample_n)


