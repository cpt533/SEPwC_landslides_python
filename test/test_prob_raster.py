import os
import sys

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray)
import xarray

sys.path.insert(0, os.pardir)
sys.path.insert(0, ".")

from terrain_analysis import make_classifier, make_prob_raster_data


class TestMakeProbRaster:
    def test_make_prob_raster(self):
        # 1. Train a small classifier on the same columns create_dataframe uses.
        train = pd.DataFrame(
            {
                "elev": np.arange(20),
                "fault": np.arange(20),
                "slope": np.arange(20),
                "LC": np.arange(20),
                "Geol": np.arange(20),
            }
        )
        y = pd.Series([0] * 10 + [1] * 10)
        clf = make_classifier(train, y)

        # 2. Build 5 tiny aligned rasters, with one NaN pixel.
        def raster():
            arr = np.arange(16, dtype=float).reshape(4, 4)
            arr[0, 0] = np.nan
            da = xarray.DataArray(arr, dims=("y", "x"))
            da.rio.write_crs("EPSG:32643", inplace=True)
            return da

        topo, geo, lc, dist_fault, slope = (raster() for _ in range(5))

        # 3. Run and check the contract.
        result = make_prob_raster_data(topo, geo, lc, dist_fault, slope, clf)

        assert result.shape == topo.shape
        assert np.isnan(result.values[0, 0])           # NaN in -> NaN out
        finite = result.values[np.isfinite(result.values)]
        assert finite.min() >= 0 and finite.max() <= 1  # probabilities
