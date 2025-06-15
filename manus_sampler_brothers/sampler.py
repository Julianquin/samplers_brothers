import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

class TimeSampler:
    """
    Librería para implementar un conjunto de estrategias de muestreo para series de tiempo,
    basadas en resultados de clustering preexistentes.
    """

    def __init__(self, series_data: pd.DataFrame, cluster_labels: pd.DataFrame, features: np.ndarray):
        """
        Inicializa el TimeSampler con los datos de las series de tiempo, las etiquetas de clúster
        y las características (embeddings) utilizadas para el clustering.

        Args:
            series_data (pd.DataFrame): DataFrame donde las columnas son los IDs de las series
                                        y las filas son los valores en el tiempo.
            cluster_labels (pd.DataFrame): DataFrame o Serie que mapea cada ID de serie a su número de clúster.
                                           Debe contener las columnas 'series_id' y 'cluster_id'.
            features (np.ndarray): Array de NumPy con las características (embeddings GASF) usadas
                                   para el clustering. Las filas deben corresponder a las series en series_data.
        """
        if not isinstance(series_data, pd.DataFrame):
            raise TypeError("series_data debe ser un DataFrame de pandas.")
        if not isinstance(cluster_labels, pd.DataFrame) and not isinstance(cluster_labels, pd.Series):
            raise TypeError("cluster_labels debe ser un DataFrame o Serie de pandas.")
        if not isinstance(features, np.ndarray):
            raise TypeError("features debe ser un array de NumPy.")

        self.series_data = series_data
        self.cluster_labels = cluster_labels.set_index('series_id') if isinstance(cluster_labels, pd.DataFrame) else cluster_labels
        self.features = features

        # Asegurar que los índices de series_data y cluster_labels coincidan
        if not self.series_data.columns.equals(self.cluster_labels.index):
            raise ValueError("Los IDs de las series en series_data y cluster_labels no coinciden.")

        # Calcular volumen de cada serie (suma de sus valores)
        self.series_volumes = self.series_data.sum(axis=0)

        # Calcular centroides de cada clúster
        self.cluster_centroids = self._calculate_cluster_centroids()

        # Calcular medoides de cada clúster
        self.cluster_medoids = self._calculate_cluster_medoids()

    def _calculate_cluster_centroids(self) -> pd.DataFrame:
        """
        Calcula los centroides de cada clúster basado en las características.
        """
        # Asegurarse de que las características estén alineadas con los IDs de las series
        features_df = pd.DataFrame(self.features, index=self.series_data.columns)
        merged_data = features_df.merge(self.cluster_labels, left_index=True, right_index=True)
        return merged_data.groupby('cluster_id').mean()

    def _calculate_cluster_medoids(self) -> pd.Series:
        """
        Calcula el medoide (la serie real más cercana al centroide) para cada clúster.
        """
        medoids = pd.Series(dtype=object)
        for cluster_id in self.cluster_labels['cluster_id'].unique():
            cluster_series_ids = self.cluster_labels[self.cluster_labels['cluster_id'] == cluster_id].index
            if cluster_series_ids.empty:
                continue

            cluster_features = self.features[self.series_data.columns.isin(cluster_series_ids)]
            cluster_centroid = self.cluster_centroids.loc[cluster_id].values.reshape(1, -1)

            # Calcular distancias de todas las series del clúster al centroide
            distances = pairwise_distances(cluster_features, cluster_centroid)
            closest_series_idx_in_cluster = np.argmin(distances)
            medoid_series_id = cluster_series_ids[closest_series_idx_in_cluster]
            medoids.loc[cluster_id] = medoid_series_id
        return medoids

    def sample(self, strategy: str, n_samples: int, **kwargs) -> list:
        """
        Método principal para aplicar una estrategia de muestreo.

        Args:
            strategy (str): Identificador de la estrategia a usar 
                ('medoid', 'stratified', 'diversity', 'activeiterative', 
                'stratified2', 'diversity2', 'activeiterative2','stratified3').
            n_samples (int): El número total de series a seleccionar en la muestra.
            **kwargs: Argumentos adicionales específicos para ciertas estrategias.

        Returns:
            list: Una lista con los IDs de las series seleccionadas.
        """
        if not isinstance(strategy, str):
            raise TypeError("La estrategia debe ser un string.")
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples debe ser un entero positivo.")

        sampling_method = getattr(self, f'_sample_{strategy.lower()}', None)
        if sampling_method is None:
            raise ValueError(f"Estrategia de muestreo '{strategy}' no reconocida.")

        return sampling_method(n_samples, **kwargs)

    def _sample_medoid(self, n_samples: int, **kwargs) -> list:
        """
        Estrategia 'medoid' - Cluster Prototype Sampling.
        Para cada clúster, selecciona su medoide (la serie real más cercana al centroide del clúster).
        Ajusta para que el total sea n_samples.
        """
        selected_series = self.cluster_medoids.tolist()
        # Opcional: añade series adicionales (√n o log(n)) para mayor diversidad.
        # Por simplicidad, usaremos sqrt(n_samples) como número de series adicionales a considerar.
        num_additional_series = int(np.sqrt(n_samples))
        if num_additional_series > 0:
            all_series_ids = self.series_data.columns.tolist()
            # Excluir los medoides ya seleccionados
            remaining_series = [s_id for s_id in all_series_ids if s_id not in selected_series]
            if remaining_series:
                # Seleccionar aleatoriamente series adicionales de las restantes
                additional_series = np.random.choice(remaining_series, min(num_additional_series, len(remaining_series)), replace=False).tolist()
                selected_series.extend(additional_series)
        selected_series = list(set(selected_series)) # Eliminar duplicados

        # Ajusta para que el total sea n_samples.
        if len(selected_series) > n_samples:
            selected_series = np.random.choice(selected_series, n_samples, replace=False).tolist()
        elif len(selected_series) < n_samples:
            # Si hay menos series seleccionadas que n_samples, añadir series aleatorias de las restantes
            remaining_series = [s_id for s_id in self.series_data.columns if s_id not in selected_series]
            if remaining_series:
                num_to_add = n_samples - len(selected_series)
                additional_series = np.random.choice(remaining_series, min(num_to_add, len(remaining_series)), replace=False).tolist()
                selected_series.extend(additional_series)

        return selected_series


    def _sample_diversity(self, n_samples: int, **kwargs) -> list:
        """
        Estrategia 'diversity' - Diversity-Weighted Cluster Sampling.
        Intenta seleccionar un número fijo de series por clúster (p. ej., 7) y luego sub-muestrea aleatoriamente
        hasta alcanzar n_samples. Para cada clúster, selecciona:
        1. Las 2 series más cercanas al medoid.
        2. Las 2 series más lejanas al medoid.
        3. Las 2 series con mayor volumen total.
        4. La serie con el volumen más extremo (outlier), si existe.
        """
        selected_series = []
        series_ids_by_cluster = self.cluster_labels.groupby('cluster_id').groups

        for cluster_id, series_ids_in_cluster in series_ids_by_cluster.items():
            series_ids_in_cluster = series_ids_in_cluster.tolist()
            if not series_ids_in_cluster:
                continue

            cluster_features = self.features[self.series_data.columns.isin(series_ids_in_cluster)]
            cluster_series_volumes = self.series_volumes[series_ids_in_cluster]

            # 1. Las 2 series más cercanas al medoid
            medoid_series_id = self.cluster_medoids.get(cluster_id)
            if medoid_series_id and medoid_series_id in series_ids_in_cluster:
                # Calculate distances to the medoid for all series in the cluster
                medoid_feature = self.features[self.series_data.columns.get_loc(medoid_series_id)].reshape(1, -1)
                distances_to_medoid = pairwise_distances(cluster_features, medoid_feature).flatten()
                sorted_indices = np.argsort(distances_to_medoid)
                closest_series = [series_ids_in_cluster[i] for i in sorted_indices[:2] if i < len(series_ids_in_cluster)]
                selected_series.extend(closest_series)

            # 2. Las 2 series más lejanas al medoid
            if medoid_series_id and medoid_series_id in series_ids_in_cluster:
                farthest_series = [series_ids_in_cluster[i] for i in sorted_indices[-2:] if i < len(series_ids_in_cluster)]
                selected_series.extend(farthest_series)

            # 3. Las 2 series con mayor volumen total
            top_volume_series = cluster_series_volumes.nlargest(2).index.tolist()
            selected_series.extend(top_volume_series)

            # 4. La serie con el volumen más extremo (outlier), si existe.
            # Definir outlier de volumen como el que está más allá de 1.5 * IQR del volumen del clúster
            Q1 = cluster_series_volumes.quantile(0.25)
            Q3 = cluster_series_volumes.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR

            outliers = cluster_series_volumes[(cluster_series_volumes > upper_bound) | (cluster_series_volumes < lower_bound)]
            if not outliers.empty:
                # Tomar el outlier con el volumen más extremo (mayor desviación de la media/mediana)
                extreme_outlier = outliers.abs().idxmax() # idxmax() returns index of max value
                selected_series.append(extreme_outlier)

        selected_series = list(set(selected_series)) # Eliminar duplicados

        # Ajustar al número total de muestras requerido
        if len(selected_series) > n_samples:
            selected_series = np.random.choice(selected_series, n_samples, replace=False).tolist()
        elif len(selected_series) < n_samples:
            # Si no se alcanzan n_samples, añadir series aleatorias de los clústeres restantes
            remaining_series = [s_id for s_id in self.series_data.columns if s_id not in selected_series]
            if remaining_series:
                num_to_add = n_samples - len(selected_series)
                additional_series = np.random.choice(remaining_series, num_to_add, replace=False).tolist()
                selected_series.extend(additional_series)

        return selected_series


    def _sample_stratified(self, n_samples: int, **kwargs) -> list:
        """
        Estrategia 'stratified' - Stratified Proportional Sampling + Rare Oversampling.
        1. Asigna cuotas de muestreo a cada clúster proporcionalmente a su tamaño.
        2. Dentro de cada clúster, estratifica las series por volumen (percentiles: bajo 0-30, medio 30-70, alto 70-100).
        3. Muestrea de cada estrato de volumen para cumplir la cuota del clúster.
        4. Aplica un sobremuestreo a estratos minoritarios (aquellos con menos del 5% de las series del clúster).
        5. Asegura la inclusión de outliers de volumen y forma (series en los bordes del clúster).
        """
        selected_series = []
        series_ids_by_cluster = self.cluster_labels.groupby('cluster_id').groups
        total_series = len(self.series_data.columns)

        # 1. Asignar cuotas de muestreo a cada clúster proporcionalmente a su tamaño.
        cluster_sizes = self.cluster_labels['cluster_id'].value_counts()
        cluster_quotas = (cluster_sizes / total_series * n_samples).round().astype(int)

        for cluster_id, series_ids_in_cluster in series_ids_by_cluster.items():
            series_ids_in_cluster = series_ids_in_cluster.tolist()
            if not series_ids_in_cluster:
                continue

            cluster_quota = cluster_quotas.get(cluster_id, 0)
            if cluster_quota == 0:
                continue

            cluster_series_volumes = self.series_volumes[series_ids_in_cluster]

            # 2. Dentro de cada clúster, estratifica las series por volumen
            low_volume_threshold = cluster_series_volumes.quantile(0.30)
            high_volume_threshold = cluster_series_volumes.quantile(0.70)

            low_volume_series = cluster_series_volumes[cluster_series_volumes <= low_volume_threshold].index.tolist()
            medium_volume_series = cluster_series_volumes[(cluster_series_volumes > low_volume_threshold) & (cluster_series_volumes <= high_volume_threshold)].index.tolist()
            high_volume_series = cluster_series_volumes[cluster_series_volumes > high_volume_threshold].index.tolist()

            strata = {
                'low': low_volume_series,
                'medium': medium_volume_series,
                'high': high_volume_series
            }

            current_cluster_sample = []
            # 3. Muestrea de cada estrato de volumen para cumplir la cuota del clúster.
            for stratum_name, stratum_series_ids in strata.items():
                if not stratum_series_ids:
                    continue
                # Proporción de series en este estrato dentro del clúster
                stratum_proportion = len(stratum_series_ids) / len(series_ids_in_cluster)
                num_to_sample_from_stratum = round(cluster_quota * stratum_proportion)

                if num_to_sample_from_stratum > len(stratum_series_ids):
                    num_to_sample_from_stratum = len(stratum_series_ids)

                if num_to_sample_from_stratum > 0:
                    current_cluster_sample.extend(np.random.choice(stratum_series_ids, num_to_sample_from_stratum, replace=False).tolist())

            # 4. Aplica un sobremuestreo a estratos minoritarios (2x)
            for stratum_name, stratum_series_ids in strata.items():
                if len(stratum_series_ids) / len(series_ids_in_cluster) < 0.05 and stratum_series_ids:
                    # Si este estrato fue muestreado, duplicar la cantidad muestreada de él, hasta su tamaño total
                    num_to_oversample = len(stratum_series_ids) # Inicialmente, considerar todo el estrato
                    if num_to_oversample > 0:
                        # Añadir series adicionales de este estrato, asegurando no duplicados y no excediendo el tamaño del estrato
                        additional_series_candidates = [s for s in stratum_series_ids if s not in current_cluster_sample]
                        num_to_add = min(num_to_oversample, len(additional_series_candidates))
                        if num_to_add > 0:
                            current_cluster_sample.extend(np.random.choice(additional_series_candidates, num_to_add, replace=False).tolist())

            # 5. Asegura la inclusión de outliers de volumen y forma
            # Outliers de volumen (usando IQR como en _sample_k)
            Q1 = cluster_series_volumes.quantile(0.25)
            Q3 = cluster_series_volumes.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            volume_outliers = cluster_series_volumes[(cluster_series_volumes > upper_bound) | (cluster_series_volumes < lower_bound)].index.tolist()
            current_cluster_sample.extend(volume_outliers)

            # Outliers de forma (series en los bordes del clúster - las más lejanas al centroide)
            cluster_features = self.features[self.series_data.columns.isin(series_ids_in_cluster)]
            cluster_centroid = self.cluster_centroids.loc[cluster_id].values.reshape(1, -1)
            distances_to_centroid = pairwise_distances(cluster_features, cluster_centroid).flatten()
            sorted_indices = np.argsort(distances_to_centroid)
            # Seleccionar las 2 series más lejanas al centroide como outliers de forma
            form_outliers = [series_ids_in_cluster[i] for i in sorted_indices[-2:] if i < len(series_ids_in_cluster)]
            current_cluster_sample.extend(form_outliers)

            selected_series.extend(list(set(current_cluster_sample))) # Eliminar duplicados dentro del clúster

        selected_series = list(set(selected_series)) # Eliminar duplicados globales

        # Ajustar al número total de muestras requerido
        if len(selected_series) > n_samples:
            selected_series = np.random.choice(selected_series, n_samples, replace=False).tolist()
        elif len(selected_series) < n_samples:
            # Si no se alcanzan n_samples, añadir series aleatorias de las restantes
            remaining_series = [s_id for s_id in self.series_data.columns if s_id not in selected_series]
            if remaining_series:
                num_to_add = n_samples - len(selected_series)
                additional_series = np.random.choice(remaining_series, num_to_add, replace=False).tolist()
                selected_series.extend(additional_series)

        return selected_series


    def _sample_activeiterative(self, n_samples: int, **kwargs) -> list:
        """
        Estrategia 'activeiterative' - Active-Iterative Hard-Case Sampling.
        1. Empieza con una muestra inicial pequeña usando la estrategia 'A'.
        2. Requiere un modelo proxy (p. ej., un LSTM simple pasado como argumento en **kwargs).
        3. Entrena el modelo en la muestra actual.
        4. Evalúa el modelo sobre las series no incluidas en la muestra.
        5. Añade a la muestra las k series con el error de predicción más alto ("casos difíciles").
        6. Repite hasta alcanzar n_samples.
        """
        model_proxy = kwargs.get("model_proxy")
        if model_proxy is None:
            raise ValueError("La estrategia 'activeiterative' requiere un 'model_proxy' en kwargs.")

        # 1. Empieza con una muestra inicial pequeña usando la estrategia 'A'.
        current_sample = self._sample_medoid(n_samples=min(50, n_samples)) # Muestra inicial de 50 o n_samples si es menor

        while len(current_sample) < n_samples:
            # Obtener datos para el entrenamiento (series_data de las series en current_sample)
            train_data = self.series_data[current_sample]
            # Obtener datos para la evaluación (series_data de las series no en current_sample)
            remaining_series_ids = [s_id for s_id in self.series_data.columns if s_id not in current_sample]
            if not remaining_series_ids:
                break # No hay más series para añadir
            eval_data = self.series_data[remaining_series_ids]

            # 3. Entrena el modelo en la muestra actual.
            # Asumimos que model_proxy tiene un método .fit(X, y) o similar
            # Aquí se necesitaría una implementación real del modelo y la preparación de los datos
            # Para este ejemplo, simularemos el entrenamiento y la evaluación.
            # model_proxy.fit(train_data, ...)

            # 4. Evalúa el modelo sobre las series no incluidas en la muestra.
            # predictions = model_proxy.predict(eval_data)
            # errors = np.abs(eval_data - predictions) # Simulación de errores
            # Para simular, generamos errores aleatorios para las series restantes
            errors = pd.Series(np.random.rand(len(remaining_series_ids)), index=remaining_series_ids)

            # 5. Añade a la muestra las k series con el error de predicción más alto ("casos difíciles").
            k_to_add = min(n_samples - len(current_sample), 10) # Añadir hasta 10 series o las necesarias para alcanzar n_samples
            hard_cases = errors.nlargest(k_to_add).index.tolist()
            current_sample.extend(hard_cases)
            current_sample = list(set(current_sample)) # Eliminar duplicados

        return current_sample


    def _sample_dsa(self, n_samples: int, **kwargs) -> list:
        """
        Estrategia 'N' - Hybrid Full Coverage Sampling.
        Combina los resultados de varias estrategias.
        1. Selecciona el 50% de n_samples usando la estrategia 'diversity'.
        2. Selecciona el 25% de n_samples usando la estrategia 'stratified'.
        3. Selecciona el 25% de n_samples usando la estrategia 'activeiterative'.
        4. Elimina duplicados y asegura la cobertura de un umbral de volumen global.
        """
        sample_k_count = int(n_samples * 0.5)
        sample_l_count = int(n_samples * 0.25)
        sample_m_count = n_samples - sample_k_count - sample_l_count # Asegurar que la suma sea n_samples

        sample_k = self._sample_diversity(sample_k_count)
        sample_l = self._sample_stratified(sample_l_count)
        sample_m = self._sample_activeiterative(sample_m_count, **kwargs) # Pasar kwargs para el modelo de la estrategia M

        combined_sample = list(set(sample_k + sample_l + sample_m))

        # Asegurar la cobertura de un umbral de volumen global
        # Calcular el volumen total de todas las series
        total_global_volume = self.series_volumes.sum()
        # Definir un umbral de cobertura, por ejemplo, 30% del volumen total
        volume_threshold = total_global_volume * 0.30

        current_volume_coverage = self.series_volumes[combined_sample].sum()

        # Si la cobertura actual es menor que el umbral, añadir series de mayor volumen no seleccionadas
        if current_volume_coverage < volume_threshold:
            remaining_series = self.series_volumes[~self.series_volumes.index.isin(combined_sample)].sort_values(ascending=False)
            for series_id, volume in remaining_series.items():
                if current_volume_coverage >= volume_threshold:
                    break
                combined_sample.append(series_id)
                current_volume_coverage += volume

        # Ajustar al número total de muestras requerido si se excede o no se alcanza
        if len(combined_sample) > n_samples:
            # Si se excede, priorizar series con mayor volumen
            combined_sample_volumes = self.series_volumes[combined_sample].sort_values(ascending=False)
            combined_sample = combined_sample_volumes.head(n_samples).index.tolist()
        elif len(combined_sample) < n_samples:
            # Si no se alcanza, añadir series aleatorias de las restantes
            remaining_series = [s_id for s_id in self.series_data.columns if s_id not in combined_sample]
            if remaining_series:
                num_to_add = n_samples - len(combined_sample)
                additional_series = np.random.choice(remaining_series, num_to_add, replace=False).tolist()
                combined_sample.extend(additional_series)

        return combined_sample


    def _sample_stratified2(self, n_samples: int, **kwargs) -> list:
        """
        Estrategia 'stratified2' - Hybrid Stratified Volume Sampling.
        Para cada clúster:
        1. Divide sus series en 3 estratos de volumen (bajo, medio, alto).
        2. Muestrea proporcionalmente del tamaño de cada estrato dentro del clúster.
        3. Asegura la inclusión obligatoria de series cuyo volumen supere 10 veces la media de volumen global.
        """
        selected_series = []
        series_ids_by_cluster = self.cluster_labels.groupby('cluster_id').groups
        global_mean_volume = self.series_volumes.mean()
        volume_threshold_for_inclusion = 10 * global_mean_volume

        for cluster_id, series_ids_in_cluster in series_ids_by_cluster.items():
            series_ids_in_cluster = series_ids_in_cluster.tolist()
            if not series_ids_in_cluster:
                continue

            cluster_series_volumes = self.series_volumes[series_ids_in_cluster]

            # 1. Divide sus series en 3 estratos de volumen (bajo, medio, alto).
            low_volume_threshold = cluster_series_volumes.quantile(0.30)
            high_volume_threshold = cluster_series_volumes.quantile(0.70)

            low_volume_series = cluster_series_volumes[cluster_series_volumes <= low_volume_threshold].index.tolist()
            medium_volume_series = cluster_series_volumes[(cluster_series_volumes > low_volume_threshold) & (cluster_series_volumes <= high_volume_threshold)].index.tolist()
            high_volume_series = cluster_series_volumes[cluster_series_volumes > high_volume_threshold].index.tolist()

            strata = {
                'low': low_volume_series,
                'medium': medium_volume_series,
                'high': high_volume_series
            }

            current_cluster_sample = []
            # 2. Muestrea proporcionalmente del tamaño de cada estrato dentro del clúster.
            total_series_in_cluster = len(series_ids_in_cluster)
            if total_series_in_cluster == 0: # Evitar división por cero
                continue

            # Calcular la cuota para este clúster basada en su tamaño relativo al total de series
            cluster_proportion = total_series_in_cluster / len(self.series_data.columns)
            cluster_quota = int(n_samples * cluster_proportion)
            if cluster_quota == 0 and n_samples > 0: # Asegurar que clústeres pequeños obtengan al menos 1 si n_samples > 0
                cluster_quota = 1

            for stratum_name, stratum_series_ids in strata.items():
                if not stratum_series_ids:
                    continue
                stratum_proportion_in_cluster = len(stratum_series_ids) / total_series_in_cluster
                num_to_sample_from_stratum = round(cluster_quota * stratum_proportion_in_cluster)

                if num_to_sample_from_stratum > len(stratum_series_ids):
                    num_to_sample_from_stratum = len(stratum_series_ids)

                if num_to_sample_from_stratum > 0:
                    current_cluster_sample.extend(np.random.choice(stratum_series_ids, num_to_sample_from_stratum, replace=False).tolist())

            # 3. Asegura la inclusión obligatoria de series cuyo volumen supere 10 veces la media de volumen global.
            high_volume_outliers = cluster_series_volumes[cluster_series_volumes > volume_threshold_for_inclusion].index.tolist()
            current_cluster_sample.extend(high_volume_outliers)

            selected_series.extend(list(set(current_cluster_sample))) # Eliminar duplicados dentro del clúster

        selected_series = list(set(selected_series)) # Eliminar duplicados globales

        # Ajustar al número total de muestras requerido
        if len(selected_series) > n_samples:
            # Si se excede, priorizar series con mayor volumen
            selected_series_volumes = self.series_volumes[selected_series].sort_values(ascending=False)
            selected_series = selected_series_volumes.head(n_samples).index.tolist()
        elif len(selected_series) < n_samples:
            # Si no se alcanza, añadir series aleatorias de las restantes
            remaining_series = [s_id for s_id in self.series_data.columns if s_id not in selected_series]
            if remaining_series:
                num_to_add = n_samples - len(selected_series)
                additional_series = np.random.choice(remaining_series, num_to_add, replace=False).tolist()
                selected_series.extend(additional_series)

        return selected_series


    def _sample_diversity2(self, n_samples: int, **kwargs) -> list:
        """
        Estrategia 'diversity2' - Diversified Sampling with Volume Adjustment.
        Para cada clúster, selecciona:
        1. La serie más cercana al medoid.
        2. Las 2 series más alejadas.
        3. La serie de mayor volumen.
        Después de recorrer todos los clústeres, si no se cubre el 70% del volumen total de todas las series,
        sigue añadiendo las series de mayor volumen no seleccionadas hasta alcanzarlo.
        """
        selected_series = []
        series_ids_by_cluster = self.cluster_labels.groupby("cluster_id").groups

        for cluster_id, series_ids_in_cluster in series_ids_by_cluster.items():
            series_ids_in_cluster = series_ids_in_cluster.tolist()
            if not series_ids_in_cluster:
                continue

            cluster_features = self.features[self.series_data.columns.isin(series_ids_in_cluster)]
            cluster_series_volumes = self.series_volumes[series_ids_in_cluster]

            # 1. La serie más cercana al medoid.
            medoid_series_id = self.cluster_medoids.get(cluster_id)
            if medoid_series_id and medoid_series_id in series_ids_in_cluster:
                selected_series.append(medoid_series_id)

            # 2. Las 2 series más alejadas.
            if medoid_series_id and medoid_series_id in series_ids_in_cluster:
                medoid_feature = self.features[self.series_data.columns.get_loc(medoid_series_id)].reshape(1, -1)
                distances_to_medoid = pairwise_distances(cluster_features, medoid_feature).flatten()
                sorted_indices = np.argsort(distances_to_medoid)
                farthest_series = [series_ids_in_cluster[i] for i in sorted_indices[-2:] if i < len(series_ids_in_cluster)]
                selected_series.extend(farthest_series)

            # 3. La serie de mayor volumen.
            if not cluster_series_volumes.empty:
                top_volume_series = cluster_series_volumes.idxmax()
                selected_series.append(top_volume_series)

        selected_series = list(set(selected_series)) # Eliminar duplicados

        # Ajuste por volumen
        total_global_volume = self.series_volumes.sum()
        target_volume_coverage = total_global_volume * 0.70

        current_volume_coverage = self.series_volumes[selected_series].sum()

        if current_volume_coverage < target_volume_coverage:
            remaining_series = self.series_volumes[~self.series_volumes.index.isin(selected_series)].sort_values(ascending=False)
            for series_id, volume in remaining_series.items():
                if current_volume_coverage >= target_volume_coverage:
                    break
                selected_series.append(series_id)
                current_volume_coverage += volume

        # Ajustar al número total de muestras requerido
        if len(selected_series) > n_samples:
            # Si se excede, priorizar series con mayor volumen
            selected_series_volumes = self.series_volumes[selected_series].sort_values(ascending=False)
            selected_series = selected_series_volumes.head(n_samples).index.tolist()
        elif len(selected_series) < n_samples:
            # Si no se alcanza, añadir series aleatorias de las restantes
            remaining_series = [s_id for s_id in self.series_data.columns if s_id not in selected_series]
            if remaining_series:
                num_to_add = n_samples - len(selected_series)
                additional_series = np.random.choice(remaining_series, num_to_add, replace=False).tolist()
                selected_series.extend(additional_series)

        return selected_series


    def _sample_activeiterative2(self, n_samples: int, **kwargs) -> list:
        """
        Estrategia 'activeiterative2' - Active Hybrid Sampling.
        Similar a la 'activeiterative', pero al seleccionar los "casos difíciles", prioriza aquellos que tienen un alto error de predicción
        Y un alto volumen. Incluye obligatoriamente los outliers de volumen extremo.
        """
        model_proxy = kwargs.get("model_proxy")
        if model_proxy is None:
            raise ValueError("La estrategia 'activeiterative2' requiere un 'model_proxy' en kwargs.")

        # 1. Empieza con una muestra inicial pequeña usando la estrategia 'A'.
        current_sample = self._sample_medoid(n_samples=min(50, n_samples)) # Muestra inicial de 50 o n_samples si es menor

        # Incluir obligatoriamente los outliers de volumen extremo desde el inicio
        global_mean_volume = self.series_volumes.mean()
        volume_threshold_for_extreme_outlier = 10 * global_mean_volume # Definición de outlier extremo
        extreme_volume_outliers = self.series_volumes[self.series_volumes > volume_threshold_for_extreme_outlier].index.tolist()
        current_sample.extend(extreme_volume_outliers)
        current_sample = list(set(current_sample))

        while len(current_sample) < n_samples:
            train_data = self.series_data[current_sample]
            remaining_series_ids = [s_id for s_id in self.series_data.columns if s_id not in current_sample]
            if not remaining_series_ids:
                break
            eval_data = self.series_data[remaining_series_ids]

            # Simulación de entrenamiento y evaluación del modelo
            # model_proxy.fit(train_data, ...)
            errors = pd.Series(np.random.rand(len(remaining_series_ids)), index=remaining_series_ids)

            # Priorizar casos difíciles con alto error de predicción Y alto volumen
            # Combinar errores con volúmenes de las series restantes
            remaining_volumes = self.series_volumes[remaining_series_ids]
            combined_score = errors * remaining_volumes # Simple multiplicación para priorizar ambos

            k_to_add = min(n_samples - len(current_sample), 10)
            hard_cases = combined_score.nlargest(k_to_add).index.tolist()
            current_sample.extend(hard_cases)
            current_sample = list(set(current_sample))

        return current_sample


    def _sample_stratified3(self, n_samples: int, **kwargs) -> list:
        """
        Estrategia 'stratified3' - Stratified Proportional Sampling with Volume Reinforcement.
        1. Realiza un muestreo proporcional al tamaño del clúster.
        2. Sobremuestrea clústeres pequeños (<1% del total).
        3. Reemplaza el 20% de las series seleccionadas (las de menor volumen) por las series de mayor volumen global que no fueron seleccionadas.
        """
        selected_series = []
        series_ids_by_cluster = self.cluster_labels.groupby('cluster_id').groups
        total_series = len(self.series_data.columns)

        # 1. Realiza un muestreo proporcional al tamaño del clúster.
        cluster_sizes = self.cluster_labels['cluster_id'].value_counts()
        cluster_quotas = (cluster_sizes / total_series * n_samples).round().astype(int)

        for cluster_id, series_ids_in_cluster in series_ids_by_cluster.items():
            series_ids_in_cluster = series_ids_in_cluster.tolist()
            if not series_ids_in_cluster:
                continue

            cluster_quota = cluster_quotas.get(cluster_id, 0)
            if cluster_quota == 0:
                continue

            # 2. Sobremuestrea clústeres pequeños (<1% del total).
            if len(series_ids_in_cluster) / total_series < 0.01:
                cluster_quota = max(cluster_quota, int(n_samples * 0.01)) # Asegurar al menos 1% de la muestra total

            if cluster_quota > len(series_ids_in_cluster):
                cluster_quota = len(series_ids_in_cluster)

            if cluster_quota > 0:
                selected_series.extend(np.random.choice(series_ids_in_cluster, cluster_quota, replace=False).tolist())

        selected_series = list(set(selected_series)) # Eliminar duplicados

        # 3. Reemplaza el 20% de las series seleccionadas (las de menor volumen) por las series de mayor volumen global que no fueron seleccionadas.
        num_to_replace = int(len(selected_series) * 0.20)
        if num_to_replace > 0:
            # Series seleccionadas ordenadas por volumen (ascendente)
            selected_series_volumes = self.series_volumes[selected_series].sort_values(ascending=True)
            series_to_remove = selected_series_volumes.head(num_to_replace).index.tolist()

            # Series no seleccionadas ordenadas por volumen (descendente)
            remaining_series = self.series_volumes[~self.series_volumes.index.isin(selected_series)].sort_values(ascending=False)
            series_to_add = remaining_series.head(num_to_replace).index.tolist()

            # Realizar el reemplazo
            selected_series = [s_id for s_id in selected_series if s_id not in series_to_remove]
            selected_series.extend(series_to_add)
            selected_series = list(set(selected_series)) # Eliminar duplicados que puedan surgir del reemplazo

        # Inclusión obligatoria de series outliers en volumen (>10x media global).
        global_mean_volume = self.series_volumes.mean()
        volume_threshold_for_inclusion = 10 * global_mean_volume
        high_volume_outliers = self.series_volumes[self.series_volumes > volume_threshold_for_inclusion].index.tolist()
        selected_series.extend(high_volume_outliers)
        selected_series = list(set(selected_series)) # Eliminar duplicados

        # Ajustar al número total de muestras requerido
        if len(selected_series) > n_samples:
            # Si se excede, priorizar series con mayor volumen
            selected_series_volumes = self.series_volumes[selected_series].sort_values(ascending=False)
            selected_series = selected_series_volumes.head(n_samples).index.tolist()
        elif len(selected_series) < n_samples:
            # Si no se alcanza, añadir series aleatorias de las restantes
            remaining_series = [s_id for s_id in self.series_data.columns if s_id not in selected_series]
            if remaining_series:
                num_to_add = n_samples - len(selected_series)
                additional_series = np.random.choice(remaining_series, num_to_add, replace=False).tolist()
                selected_series.extend(additional_series)

        return selected_series


