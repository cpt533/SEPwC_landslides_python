import sys

sys.path.insert(0, ".")

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray)
import xarray
from affine import Affine

from terrain_analysis import calculate_slope


class TestSlope:
    def test_flat_surface_is_zero(self):
        topo = xarray.DataArray(
            np.zeros((5, 5), dtype="float32"),
            dims=("y", "x"),
            coords={"y": np.arange(4.5, -0.5, -1.0), "x": np.arange(0.5, 5.5, 1.0)},
        )
        topo.rio.write_crs("EPSG:32630", inplace=True)
        topo.rio.write_transform(Affine.translation(0, 5) * Affine.scale(1, -1), inplace=True)

        result = calculate_slope(topo)

        np.testing.assert_allclose(result.values, np.zeros((5, 5)))

    def test_constant_ramp_is_forty_five_degrees(self):
        elevation = np.tile(np.arange(5, dtype="float32"), (5, 1))
        topo = xarray.DataArray(
            elevation,
            dims=("y", "x"),
            coords={"y": np.arange(4.5, -0.5, -1.0), "x": np.arange(0.5, 5.5, 1.0)},
        )
        topo.rio.write_crs("EPSG:32630", inplace=True)
        topo.rio.write_transform(Affine.translation(0, 5) * Affine.scale(1, -1), inplace=True)

        result = calculate_slope(topo)

        np.testing.assert_allclose(result.values, np.full((5, 5), 45.0))
