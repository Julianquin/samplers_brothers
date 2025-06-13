import numpy as np
import pandas as pd
from samplers_brothers import TimeSampler

np.random.seed(0)


def create_fake_df(n_series=5, length=10):
    keys = np.repeat(np.arange(n_series), length)
    dates = np.tile(
        pd.date_range("2022-01-01", periods=length, freq="D"),
        n_series,
    )
    target = np.random.randn(n_series * length)
    clusters = np.repeat(np.arange(n_series) % 3, length)
    return pd.DataFrame({
        "key": keys,
        "dt_fecha": dates,
        "target": target,
        "cluster": clusters,
    })


def test_basic_sampling():
    df = create_fake_df()
    sampler = TimeSampler(df)
    sample = sampler.sample("k", n_samples=3)
    assert len(sample) == 3
    assert set(sample).issubset(set(df["key"].unique()))


def test_strategy_a_returns_medoids_and_extra():
    np.random.seed(0)
    df = create_fake_df(n_series=6, length=5)
    sampler = TimeSampler(df)
    medoids = list(sampler._medoids.values())

    sample = sampler.sample("A", n_samples=len(medoids))
    assert set(sample) == set(medoids)

    sample_more = sampler.sample("A", n_samples=len(medoids) + 2)
    assert set(medoids).issubset(set(sample_more))
    assert len(sample_more) == len(medoids) + 2


def test_strategy_l_represents_all_clusters():
    np.random.seed(0)
    df = create_fake_df(n_series=9, length=5)
    sampler = TimeSampler(df)
    sample = sampler.sample("L", n_samples=6)
    assert len(sample) == 6
    assert set(sample).issubset(set(df["key"].unique()))
    clusters = sampler.cluster_labels.loc[sample].unique()
    assert len(clusters) == len(sampler._clusters)


def test_strategy_p_covers_all_series_when_large_request():
    np.random.seed(0)
    df = create_fake_df(n_series=6, length=5)
    sampler = TimeSampler(df)
    sample = sampler.sample("P", n_samples=len(df["key"].unique()))
    assert set(sample) == set(df["key"].unique())
    clusters = sampler.cluster_labels.loc[sample].unique()
    assert len(clusters) == len(sampler._clusters)
