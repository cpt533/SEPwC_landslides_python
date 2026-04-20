"""
Calculate hazard risk of probability for landslides
"""

import argparse

import numpy as np
import xarray


def extract_values_from_raster(da: xarray.DataArray, shapes):
    """
    Extract raster x and y values at point locations.

    Parameters
    ----------
    da : xarray.DataArray
        Raster data with x and y coordinates.
    shapes : geopandas.GeoSeries
        Point locations to sample.

    Returns
    -------
    numpy.ndarray
        Raster values for each point.
    """
    xs = shapes.x.values
    ys = shapes.y.values

    vals = np.array([da.sel(x=x, y=y, method="nearest").item() for x, y in zip(xs, ys)])
    return vals


def make_classifier(x, y, verbose=False):
    return


def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    return


def create_dataframe(topo, geo, lc, dist_fault, slope, shapes, landslide_label):
    return


def reproject_to_match(in_raster, template_raster):
    return


def calculate_distance_to_faults(fault_shapefile, template_raster):
    return


def main(args_list=None):
    parser = argparse.ArgumentParser(
        prog="Landslide hazard using ML",
        description="Calculate landslide hazards using machine learning",
    )
    parser.add_argument("--topography", required=True, help="topographic raster file")
    parser.add_argument("--geology", required=True, help="geology raster file")
    parser.add_argument("--landcover", required=True, help="landcover raster file")
    parser.add_argument("--faults", required=True, help="fault location shapefile")
    parser.add_argument("landslides", help="landslide location shapefile")
    parser.add_argument("output", help="output probability raster file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress")

    args = parser.parse_args(args_list)


if __name__ == "__main__":
    main()
