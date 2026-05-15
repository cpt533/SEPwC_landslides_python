import sys

sys.path.insert(0, ".")

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray)
import xarray
from affine import Affine
from shapely.geometry import LineString

from terrain_analysis import calculate_distance_to_faults


class TestDistanceToFaults:
    def test_distance_to_vertical_fault(self, tmp_path):
        template = xarray.DataArray(
            np.zeros((5, 5), dtype="float32"),
            dims=("y", "x"),
            coords={"y": np.arange(4.5, -0.5, -1.0), "x": np.arange(0.5, 5.5, 1.0)},
        )
        template.rio.write_crs("EPSG:32630", inplace=True)
        template.rio.write_transform(Affine.translation(0, 5) * Affine.scale(1, -1), inplace=True)

        fault_path = tmp_path / "fault.shp"
        gpd.GeoDataFrame(
            geometry=[LineString([(2.5, 0), (2.5, 5)])], crs="EPSG:32630"
        ).to_file(fault_path)

        result = calculate_distance_to_faults(fault_path, template)

        expected = np.tile([2.0, 1.0, 0.0, 1.0, 2.0], (5, 1))
        np.testing.assert_allclose(result.values, expected)
