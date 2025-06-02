# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import transform, unary_union
import pyproj
from pathlib import Path

def compute_and_save_convex_hull(csv_path_str, source_crs_str, target_crs_str, buffer_m, output_hull_gpkg_str):
    """Computes a convex hull around buffered points from a CSV and saves it as GeoPackage."""
    print(f"--- Running: Convex Hull Computation ---")
    csv_path = Path(csv_path_str)
    output_hull_gpkg = Path(output_hull_gpkg_str)

    print(f"Reading CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {csv_path}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to read CSV {csv_path}: {e}")
        raise

    if not {'longitude', 'latitude'}.issubset(df.columns):
        raise ValueError("CSV must contain 'longitude' and 'latitude' columns.")

    print(f"Setting up projection transformation from {source_crs_str} to {target_crs_str}")
    try:
        project = pyproj.Transformer.from_crs(
            pyproj.CRS.from_user_input(source_crs_str),
            pyproj.CRS.from_user_input(target_crs_str),
            always_xy=True
        ).transform
    except Exception as e:
        print(f"ERROR setting up CRS transformation: {e}")
        raise

    buffers = []
    print(f"Buffering points with {buffer_m} meters...")
    skipped_count = 0
    for index, row in df.iterrows():
        try:
            lon, lat = float(row['longitude']), float(row['latitude'])
            p_t = transform(project, Point(lon, lat))
            buf = p_t.buffer(buffer_m)
            if buf.is_valid and not buf.is_empty: buffers.append(buf)
            else: skipped_count += 1
        except (ValueError, TypeError): skipped_count += 1
        except Exception: skipped_count +=1
    if skipped_count > 0: print(f"  Skipped {skipped_count} rows due to invalid coordinates or buffer errors.")

    if not buffers:
        raise RuntimeError("No valid buffers created from input points. Check coordinates and CRS.")

    print(f"Created {len(buffers)} valid buffers.")
    print("Merging buffers and computing convex hull...")
    try:
        merged = unary_union(buffers) # More direct than GeoSeries for this
        if not merged.is_valid: merged = merged.buffer(0)
        if not merged.is_valid: raise ValueError("Failed to create valid merged buffer.")
        hull = merged.convex_hull
        if not hull.is_valid: hull = hull.buffer(0)
        if not hull.is_valid: raise ValueError("Failed to create valid convex hull.")
    except Exception as e:
        print(f"ERROR during buffer merge or hull computation: {e}")
        raise

    print("Convex hull computation complete.")
    
    # Save the hull
    hull_gdf = gpd.GeoDataFrame([{'id': 1, 'geometry': hull}], crs=target_crs_str)
    try:
        hull_gdf.to_file(output_hull_gpkg, driver="GPKG")
        print(f"Convex hull saved to: {output_hull_gpkg}")
        return str(output_hull_gpkg)
    except Exception as e:
        print(f"ERROR saving convex hull to {output_hull_gpkg}: {e}")
        raise

if __name__ == '__main__':
    # Example Usage (for testing this module independently)
    csv_file = 'defect_coordinates.csv' # Create a dummy for testing
    if not Path(csv_file).exists():
        pd.DataFrame({'latitude': [51.0, 51.1], 'longitude': [7.0, 7.1]}).to_csv(csv_file, index=False)

    output_dir = Path("output_test_hull")
    output_dir.mkdir(exist_ok=True)
    hull_file_path = compute_and_save_convex_hull(
        csv_path_str=csv_file,
        source_crs_str='EPSG:4326',
        target_crs_str='EPSG:25832',
        buffer_m=15,
        output_hull_gpkg_str=str(output_dir / 'convex_hull.gpkg')
    )
    if hull_file_path:
        print(f"Test successful. Hull saved to {hull_file_path}")