import numpy as np
import pandas as pd
from samplers_brothers import TimeSampler


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
