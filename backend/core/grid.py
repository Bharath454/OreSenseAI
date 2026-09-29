"""
OreSense AI – Grid Utilities
Creates the spatial grid cells over the AOI and returns GeoDataFrame.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point


def build_grid_gdf(
    minlat: float,
    maxlat: float,
    minlon: float,
    maxlon: float,
    rows: int,
    cols: int,
) -> gpd.GeoDataFrame:
    """
    Divide the AOI into (rows × cols) cells.
    Returns a GeoDataFrame with columns:
        row_idx, col_idx, geom (Polygon), centroid (Point), elevation_m
    CRS: EPSG:4326
    """
    lat_edges = np.linspace(maxlat, minlat, rows + 1)  # top-to-bottom
    lon_edges = np.linspace(minlon, maxlon, cols + 1)

    records = []
    for r in range(rows):
        for c in range(cols):
            minlon_c = lon_edges[c]
            maxlon_c = lon_edges[c + 1]
            maxlat_r = lat_edges[r]
            minlat_r = lat_edges[r + 1]
            poly = box(minlon_c, minlat_r, maxlon_c, maxlat_r)
            centroid = Point(
                (minlon_c + maxlon_c) / 2,
                (minlat_r + maxlat_r) / 2,
            )
            # Synthetic elevation: gentle hill in the centre (realistic for
            # Deccan plateau terrain)
            dr = (r - rows / 2) / rows
            dc = (c - cols / 2) / cols
            elevation = 600 + 80 * np.exp(-(dr**2 + dc**2) / 0.05)
            records.append(
                {
                    "row_idx": r,
                    "col_idx": c,
                    "geom": poly,
                    "centroid": centroid,
                    "elevation_m": float(elevation),
                }
            )

    gdf = gpd.GeoDataFrame(records, geometry="geom", crs="EPSG:4326")
    return gdf


def cell_id(row_idx: int, col_idx: int, cols: int) -> int:
    """Flat index for a cell (1-based to match DB SERIAL starting at 1)."""
    return row_idx * cols + col_idx + 1
