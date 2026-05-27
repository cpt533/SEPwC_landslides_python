import os
import sys

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so the test works headless

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray)
import xarray

sys.path.insert(0, os.pardir)
sys.path.insert(0, ".")

from terrain_analysis import plot_probability_raster


class TestPlotProbabilityRaster:
    def test_png_is_written(self, tmp_path):
        # Build a tiny probability raster with one NaN pixel.
        arr = np.linspace(0, 1, 16, dtype=float).reshape(4, 4)
        arr[0, 0] = np.nan
        da = xarray.DataArray(arr, dims=("y", "x"))
        da.rio.write_crs("EPSG:32643", inplace=True)

        out = tmp_path / "prob.png"
        plot_probability_raster(da, out)

        assert out.exists()
        assert out.stat().st_size > 0
