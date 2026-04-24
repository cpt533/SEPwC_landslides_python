"""
Calculate hazard risk of probability for landslides
"""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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


def make_classifier(x: pd.DataFrame, y: pd.Series, verbose: bool = False):
    """
    Train a RandomForestClassifier and evaluate it on a test split.

    The data is split into training and test sets. When verbose is True,
    the function prints the accuracy, actual test labels, and predicted labels.

    Args:
        x: Feature matrix as a pandas DataFrame.
        y: Target labels as a pandas Series.
        verbose: Whether to print evaluation details.

    Returns:
        A trained RandomForestClassifier instance.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42, stratify=y
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    if verbose:
        print("Accuracy:", acc)
        print("Actual:")
        print(y_test)
        print("Predicted:")
        print(y_pred)
    return model


def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    return


def create_dataframe(
    topo: xarray.DataArray,
    geo: xarray.DataArray,
    lc: xarray.DataArray,
    dist_fault: xarray.DataArray,
    slope: xarray.DataArray,
    shapes: gpd.geoseries.GeoSeries,
    landslide_label: int,
):

    df = pd.DataFrame(
        {
            "elev": extract_values_from_raster(topo, shapes),
            "fault": extract_values_from_raster(dist_fault, shapes),
            "slope": extract_values_from_raster(slope, shapes),
            "LC": extract_values_from_raster(lc, shapes),
            "Geol": extract_values_from_raster(geo, shapes),
        }
    )
    df["ls"] = landslide_label
    return df


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
