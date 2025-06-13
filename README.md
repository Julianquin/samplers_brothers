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
