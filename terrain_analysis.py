"""
Calculate hazard risk of probability for landslides
"""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import xarray
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# TODO: i think the data has polygons in it but this function can only do points
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
    """
    Predict landslide probability for every pixel and return a raster.

    Each pixel of the input rasters is treated as one row of features (in the
    same column order used by ``create_dataframe``). The classifier's
    probability for the landslide class is reshaped back to the raster grid.

    Pixels where any input layer is NaN (e.g. edge pixels introduced by
    ``reproject_to_match``) are filled with 0 before prediction so the
    classifier does not crash, then set back to NaN.
    Parameters
    ----------
    topo, geo, lc, dist_fault, slope : xarray.DataArray
        Aligned input rasters (same shape, transform and CRS).
    classifier : sklearn estimator
        A fitted classifier whose ``predict_proba`` returns landslide
        probability in column index 1.

    Returns
    -------
    xarray.DataArray
        Raster of landslide probabilities in [0, 1] aligned with ``topo``.
    """
    shape = topo.shape

    df = pd.DataFrame(
        {
            "elev": topo.values.ravel(),
            "fault": dist_fault.values.ravel(),
            "slope": slope.values.ravel(),
            "LC": lc.values.ravel(),
            "Geol": geo.values.ravel(),
        }
    )

    nan_mask = df.isna().any(axis=1).values
    df_filled = df.fillna(0)

    probs = classifier.predict_proba(df_filled)[:, 1]
    probs[nan_mask] = np.nan
    prob_grid = probs.reshape(shape)

    result = xarray.DataArray(
        prob_grid,
        coords=topo.coords,
        dims=topo.dims,
    )
    result.rio.write_crs(topo.rio.crs, inplace=True)
    result.rio.write_transform(topo.rio.transform(), inplace=True)
    return result


def create_dataframe(
    topo: xarray.DataArray,
    geo: xarray.DataArray,
    lc: xarray.DataArray,
    dist_fault: xarray.DataArray,
    slope: xarray.DataArray,
    shapes: gpd.geoseries.GeoSeries,
    landslide_label: int,
):
    """
    Build a feature-matrix DataFrame by sampling rasters at point locations.

    For each point in ``shapes`` the nearest pixel is sampled from each of the
    five raster layers and stored as a row of features. A target column ``ls``
    is filled with ``landslide_label`` for every row, producing a table ready
    to pass to a classifier.

    Parameters
    ----------
    topo : xarray.DataArray
        Topography/elevation raster. Populates the ``elev`` column.
    geo : xarray.DataArray
        Geology raster. Populates the ``Geol`` column.
    lc : xarray.DataArray
        Land-cover raster. Populates the ``LC`` column.
    dist_fault : xarray.DataArray
        Distance-to-fault raster. Populates the ``fault`` column.
    slope : xarray.DataArray
        Slope raster. Populates the ``slope`` column.
    shapes : geopandas.GeoSeries
        Point geometries at which to sample the rasters.
    landslide_label : int
        Value written to the ``ls`` column for every row (e.g. ``1`` for
        landslide points, ``0`` for non-landslide points).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``elev``, ``fault``, ``slope``, ``LC``,
        ``Geol`` and ``ls``, one row per input point.
    """
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


def reproject_to_match(in_raster: xarray.DataArray, template_raster: xarray.DataArray):
    return in_raster.rio.reproject_match(template_raster, nodata=np.nan)


def calculate_slope(topo: xarray.DataArray) -> xarray.DataArray:
    pixel_size = abs(topo.rio.transform().a)
    dz_dy, dz_dx = np.gradient(topo.values, pixel_size)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    result = xarray.DataArray(
        slope_deg,
        coords=topo.coords,
        dims=topo.dims,
    )
    result.rio.write_crs(topo.rio.crs, inplace=True)
    result.rio.write_transform(topo.rio.transform(), inplace=True)
    return result


def calculate_distance_to_faults(fault_shapefile, template_raster: xarray.DataArray):
    """
    Build a raster where each pixel stores the distance to the nearest fault.

    Parameters
    ----------
    fault_shapefile : str
        Path to a shapefile containing fault line geometries.
    template_raster : xarray.DataArray
        Raster whose grid (shape, transform, CRS) the output should match.

    Returns
    -------
    xarray.DataArray
        A raster aligned with ``template_raster`` whose values are the
        distance (in the template's CRS units) from each pixel
        to the nearest fault.
    """
    faults = gpd.read_file(fault_shapefile)
    faults = faults.to_crs(template_raster.rio.crs)

    transform = template_raster.rio.transform()
    out_shape = (template_raster.rio.height, template_raster.rio.width)

    fault_mask = rasterio.features.rasterize(
        faults.geometry,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    )

    distances_px = distance_transform_edt(fault_mask == 0)
    pixel_size = abs(transform.a)
    distances = distances_px * pixel_size

    result = xarray.DataArray(
        distances,
        coords=template_raster.coords,
        dims=template_raster.dims,
    )
    result.rio.write_crs(template_raster.rio.crs, inplace=True)
    result.rio.write_transform(transform, inplace=True)
    return result


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
    parser.add_argument(
        "-v",
        "--v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Print progress",
    )

    args = parser.parse_args(args_list)


if __name__ == "__main__":
    main()
