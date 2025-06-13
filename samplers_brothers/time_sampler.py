"""Time series sampling helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from typing import List


class TimeSampler:
    """Sampling helper based on clustering results.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format DataFrame containing the following columns:
        ``key`` to identify each series, ``dt_fecha`` for the timestamp,
        ``target`` with the observed values and ``cluster`` with the cluster id
        of the series. All rows for a given ``key`` must share the same
        ``cluster`` value.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the sampler with a long-format DataFrame."""
        required_cols = {"key", "dt_fecha", "target", "cluster"}
        if not required_cols <= set(data.columns):
            raise ValueError(
                f"Input DataFrame must contain columns: {required_cols}"
            )

        self.data_long = data.copy()

        pivot = (
            data.pivot(index="dt_fecha", columns="key", values="target")
            .sort_index()
            .fillna(0)
        )
        self.series_data = pivot

        self.cluster_labels = (
            data[["key", "cluster"]]
            .drop_duplicates()
            .set_index("key")
            ["cluster"]
        )

        self.features = self.series_data.T.values
        self._volumes = self.series_data.sum()
        self._clusters = self.cluster_labels.groupby(
            self.cluster_labels
        ).groups
        self._medoids = self._compute_medoids()
        self._centroids = self._compute_centroids()

    @property
    def series_ids(self) -> List[str]:
        """Return the list of all available series identifiers."""
        return list(self.series_data.columns)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _compute_centroids(self) -> dict:
        centroids: dict = {}
        for cid, idx in self._clusters.items():
            centroids[cid] = self.features[list(idx)].mean(axis=0)
        return centroids

    def _compute_medoids(self) -> dict:
        medoids: dict = {}
        for cid, idx in self._clusters.items():
            feats = self.features[list(idx)]
            centroid = feats.mean(axis=0, keepdims=True)
            dists = pairwise_distances(feats, centroid)
            medoid_idx = list(idx)[int(np.argmin(dists))]
            medoids[cid] = medoid_idx
        return medoids

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def sample(self, strategy: str, n_samples: int, **kwargs) -> List[str]:
        """Select a subset of series identifiers using a strategy.

        Parameters
        ----------
        strategy : str
            Identifier of the sampling strategy to use.
        n_samples : int
            Maximum number of series to return.
        **kwargs : dict
            Additional keyword arguments passed to the concrete strategy
            implementation.

        Returns
        -------
        list[str]
            The identifiers of the selected series.
        """
        method = getattr(self, f"_sample_{strategy.lower()}", None)
        if method is None:
            raise ValueError(f"Unknown strategy '{strategy}'")
        return method(n_samples, **kwargs)

    # ------------------------------------------------------------------
    # strategies
    # ------------------------------------------------------------------
    def _sample_a(self, n_samples: int, **_: dict) -> List[str]:
        """Return cluster medoids followed by remaining series.

        Parameters
        ----------
        n_samples : int
            Desired number of series to select.
        **_ : dict
            Unused extra keyword arguments.

        Returns
        -------
        list[str]
            Selected series identifiers.
        """
        medoids = list(self._medoids.values())
        if len(medoids) >= n_samples:
            return medoids[:n_samples]
        remaining = [s for s in self.series_ids if s not in medoids]
        return (medoids + remaining)[:n_samples]

    def _sample_k(self, n_samples: int, per_cluster: int = 7) -> List[str]:
        """Select series closest/farthest to each cluster medoid.

        Parameters
        ----------
        n_samples : int
            Maximum number of series to return.
        per_cluster : int, default 7
            Limit of candidates drawn from each cluster before trimming.

        Returns
        -------
        list[str]
            Identifiers of the chosen series.
        """
        selected: List[str] = []
        for cid, ids in self._clusters.items():
            ids = list(ids)
            feats = self.features[ids]
            centroid = self.features[self._medoids[cid]][None]
            dists = pairwise_distances(feats, centroid).ravel()
            sorted_idx = np.argsort(dists)
            closest = [ids[i] for i in sorted_idx[:2]]
            farthest = [ids[i] for i in sorted_idx[-2:]]
            volumes = self._volumes[ids]
            top_vol = volumes.sort_values(ascending=False).index.tolist()[:2]
            volume_z = ((volumes - volumes.mean()) / volumes.std()).abs()
            outlier_idx = volume_z.idxmax()
            cluster_sel = list(
                dict.fromkeys(closest + farthest + top_vol + [outlier_idx])
            )
            selected.extend(cluster_sel[:per_cluster])
        if len(selected) > n_samples:
            selected = list(
                np.random.choice(selected, size=n_samples, replace=False)
            )
        return selected

    def _sample_l(self, n_samples: int) -> List[str]:
        """Stratified sampling within each cluster by volume.

        Parameters
        ----------
        n_samples : int
            Number of series to draw overall.

        Returns
        -------
        list[str]
            Identifiers of the selected series.
        """
        quotas = {
            cid: int(len(ids) / len(self.series_ids) * n_samples)
            for cid, ids in self._clusters.items()
        }
        selected: List[str] = []
        for cid, ids in self._clusters.items():
            vol = self._volumes[list(ids)]
            q = max(1, quotas[cid])
            bins = np.percentile(vol, [30, 70])
            strata = {
                'low': vol[vol <= bins[0]].index,
                'mid': vol[(vol > bins[0]) & (vol <= bins[1])].index,
                'high': vol[vol > bins[1]].index,
            }
            samples: List[str] = []
            for s_ids in strata.values():
                if len(s_ids) == 0:
                    continue
                k = max(1, int(len(s_ids) / len(ids) * q))
                k = min(k, len(s_ids))
                samples.extend(
                    np.random.choice(list(s_ids), size=k, replace=False)
                )
            selected.extend(samples[:q])
        if len(selected) > n_samples:
            selected = list(
                np.random.choice(selected, n_samples, replace=False)
            )
        return selected

    def _sample_m(
        self,
        n_samples: int,
        model_proxy=None,
        step: int = 5,
    ) -> List[str]:
        """Select series iteratively using a forecasting model.

        Parameters
        ----------
        n_samples : int
            Number of series to select.
        model_proxy : object, optional
            Object implementing ``fit`` and ``predict`` used to score series.
        step : int, default 5
            Number of series added on each iteration.

        Returns
        -------
        list[str]
            Identifiers of the chosen series.
        """
        if model_proxy is None:
            raise ValueError('model_proxy is required for strategy M')
        selected = self._sample_a(min(len(self.series_ids), step))
        remaining = [sid for sid in self.series_ids if sid not in selected]
        while len(selected) < n_samples and remaining:
            X_train = self.series_data[selected].T.values
            model_proxy.fit(X_train)
            X_rem = self.series_data[remaining].T.values
            preds = model_proxy.predict(X_rem)
            errors = np.mean(np.abs(X_rem - preds), axis=1)
            idx_sorted = np.argsort(errors)[-step:]
            new_ids = [remaining[i] for i in idx_sorted]
            selected.extend(new_ids)
            remaining = [sid for sid in remaining if sid not in new_ids]
        return selected[:n_samples]

    def _sample_n(self, n_samples: int, **kwargs) -> List[str]:
        """Blend strategies ``K``, ``L`` and ``M``.

        Parameters
        ----------
        n_samples : int
            Total number of series to select.
        **kwargs : dict
            Extra arguments passed to ``_sample_m``.

        Returns
        -------
        list[str]
            Combined list of selected series.
        """
        half = max(1, int(n_samples * 0.5))
        qtr = max(1, int(n_samples * 0.25))
        sample_k = self._sample_k(half)
        sample_l = self._sample_l(qtr)
        sample_m = self._sample_m(qtr, **kwargs)
        combined = list(dict.fromkeys(sample_k + sample_l + sample_m))
        if len(combined) > n_samples:
            combined = list(
                np.random.choice(combined, n_samples, replace=False)
            )
        return combined

    def _sample_o(self, n_samples: int) -> List[str]:
        """Stratified sampling giving weight to extreme volumes.

        Parameters
        ----------
        n_samples : int
            Desired number of series to return.

        Returns
        -------
        list[str]
            The identifiers of the selected series.
        """
        selected: List[str] = []
        global_mean = self._volumes.mean()
        for cid, ids in self._clusters.items():
            ids = list(ids)
            vols = self._volumes[ids]
            bins = np.percentile(vols, [33, 66])
            strata = [
                vols[vols <= bins[0]].index,
                vols[(vols > bins[0]) & (vols <= bins[1])].index,
                vols[vols > bins[1]].index,
            ]
            q = int(len(ids) / len(self.series_ids) * n_samples)
            for s in strata:
                if len(s) == 0:
                    continue
                k = max(1, int(len(s) / len(ids) * q))
                selected.extend(
                    np.random.choice(list(s), size=k, replace=False)
                )
            extreme = vols[vols > 10 * global_mean].index.tolist()
            selected.extend(extreme)
        selected = list(dict.fromkeys(selected))
        if len(selected) > n_samples:
            selected = list(
                np.random.choice(selected, n_samples, replace=False)
            )
        return selected

    def _sample_p(self, n_samples: int) -> List[str]:
        """Select medoid, extremes and high-volume series from each cluster.

        Parameters
        ----------
        n_samples : int
            Maximum number of series to return.

        Returns
        -------
        list[str]
            Identifiers of the selected series.
        """
        selected: List[str] = []
        for cid, ids in self._clusters.items():
            ids = list(ids)
            feats = self.features[ids]
            centroid = feats.mean(axis=0, keepdims=True)
            dists = pairwise_distances(feats, centroid).ravel()
            medoid = ids[int(np.argmin(dists))]
            farthest = [ids[i] for i in np.argsort(dists)[-2:]]
            highest_volume = self._volumes[ids].idxmax()
            selected.extend([medoid] + farthest + [highest_volume])
        selected = list(dict.fromkeys(selected))
        total_vol_selected = self._volumes[selected].sum()
        total_vol = self._volumes.sum()
        remaining = [sid for sid in self.series_ids if sid not in selected]
        if total_vol_selected / total_vol < 0.7:
            remaining_sorted = self._volumes[remaining].sort_values(
                ascending=False
            )
            for sid in remaining_sorted.index:
                if sid not in selected:
                    selected.append(sid)
                if self._volumes[selected].sum() / total_vol >= 0.7:
                    break
        return selected[:n_samples]

    def _sample_q(
        self,
        n_samples: int,
        model_proxy=None,
        step: int = 5,
    ) -> List[str]:
        """Select series iteratively using model scores weighted by volume.

        Parameters
        ----------
        n_samples : int
            Desired number of series.
        model_proxy : object, optional
            Object providing ``fit`` and ``predict`` to estimate errors.
        step : int, default 5
            Number of series added per iteration.

        Returns
        -------
        list[str]
            Identifiers of the selected series.
        """
        if model_proxy is None:
            raise ValueError('model_proxy is required for strategy Q')
        selected = self._sample_a(min(len(self.series_ids), step))
        remaining = [sid for sid in self.series_ids if sid not in selected]
        while len(selected) < n_samples and remaining:
            X_train = self.series_data[selected].T.values
            model_proxy.fit(X_train)
            X_rem = self.series_data[remaining].T.values
            preds = model_proxy.predict(X_rem)
            errors = np.mean(np.abs(X_rem - preds), axis=1)
            volumes = self._volumes[remaining].values
            scores = errors * volumes
            idx_sorted = np.argsort(scores)[-step:]
            new_ids = [remaining[i] for i in idx_sorted]
            selected.extend(new_ids)
            remaining = [sid for sid in remaining if sid not in new_ids]
        extreme_outliers = self._volumes[
            self._volumes > 3 * self._volumes.std()
        ].index.tolist()
        selected.extend(extreme_outliers)
        selected = list(dict.fromkeys(selected))
        return selected[:n_samples]

    def _sample_r(self, n_samples: int) -> List[str]:
        """Sample proportionally to cluster size with volume adjustment.

        Parameters
        ----------
        n_samples : int
            Desired number of series.

        Returns
        -------
        list[str]
            Identifiers of the selected series.
        """
        quotas = {
            cid: int(len(ids) / len(self.series_ids) * n_samples)
            for cid, ids in self._clusters.items()
        }
        selected: List[str] = []
        small_clusters = [
            cid
            for cid, ids in self._clusters.items()
            if len(ids) / len(self.series_ids) < 0.01
        ]
        for cid, ids in self._clusters.items():
            q = max(1, quotas[cid])
            if cid in small_clusters:
                q += 1
            selected.extend(
                np.random.choice(
                    list(ids), size=min(q, len(ids)), replace=False
                )
            )
        selected = list(dict.fromkeys(selected))
        vol_selected = self._volumes[selected]
        lowest = vol_selected.sort_values().index[
            : max(1, int(len(selected) * 0.2))
        ]
        remaining = [sid for sid in self.series_ids if sid not in selected]
        high_volume = self._volumes[remaining].sort_values(ascending=False)
        replacement = high_volume.index[: len(lowest)]
        selected = [sid for sid in selected if sid not in lowest]
        selected.extend(list(replacement))
        return selected[:n_samples]
