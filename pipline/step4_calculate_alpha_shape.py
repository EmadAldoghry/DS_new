# step4_calculate_alpha_shape.py
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import alphashape
import traceback
from helpers import plot_alpha_shape_result
import pandas as pd
import numpy as np

def calculate_and_save_alpha_shape(sampled_points_input,
                                   target_crs_str,
                                   output_dir_str,
                                   output_filename_stem,
                                   alpha_parameter=None, # Manual alpha (float)
                                   default_alpha_if_optimize_fails=1000.0,
                                   show_plots=True, save_plots=True, plot_dpi=150):
    print("\n--- Running: Alpha Shape Calculation ---")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_gml_path = output_dir / f"{output_filename_stem}.gml"

    points_for_alphashape_tuples = []
    original_points_for_plotting = []

    if isinstance(sampled_points_input, str) or isinstance(sampled_points_input, Path):
        try:
            points_df = pd.read_csv(sampled_points_input)
            if not {'x', 'y'}.issubset(points_df.columns):
                raise ValueError("Points CSV must contain 'x' and 'y' columns.")
            seen_coords_for_alpha_tuples = set()
            for _, row in points_df.iterrows():
                p = Point(row['x'], row['y'])
                original_points_for_plotting.append(p)
                coord_tuple = (p.x, p.y)
                if coord_tuple not in seen_coords_for_alpha_tuples:
                    points_for_alphashape_tuples.append(coord_tuple)
                    seen_coords_for_alpha_tuples.add(coord_tuple)
            print(f"Loaded {len(original_points_for_plotting)} points from {sampled_points_input}, "
                  f"{len(points_for_alphashape_tuples)} unique coordinate tuples for alpha shape.")
        except Exception as e:
            print(f"Error loading points from CSV {sampled_points_input}: {e}")
            return None
    elif isinstance(sampled_points_input, list) and all(isinstance(p, Point) for p in sampled_points_input):
        original_points_for_plotting = sampled_points_input
        seen_coords_for_alpha_tuples = set()
        for p_obj in original_points_for_plotting:
            if p_obj.is_valid and not p_obj.is_empty:
                coord_tuple = (p_obj.x, p_obj.y)
                if coord_tuple not in seen_coords_for_alpha_tuples:
                    points_for_alphashape_tuples.append(coord_tuple)
                    seen_coords_for_alpha_tuples.add(coord_tuple)
        print(f"Using {len(points_for_alphashape_tuples)} unique coordinate tuples from provided list for alpha shape.")
    else:
        print("Error: sampled_points_input must be a list of Shapely Points or a path to a CSV file.")
        return None

    if not points_for_alphashape_tuples or len(points_for_alphashape_tuples) < 3:
        print("Not enough valid unique points/tuples for alphashape. Need at least 3.")
        return None

    alpha_shape_polygon = None
    actual_alpha_used_display = "N/A" # For display in plot title

    try:
        if alpha_parameter is not None: # Manual alpha provided
            print(f"  Using manually set alpha: {alpha_parameter}")
            actual_alpha_used_display = f"{alpha_parameter:.4g}"
            alpha_shape_polygon = alphashape.alphashape(points_for_alphashape_tuples, alpha_parameter)
        else: # Attempt to optimize
            print("Attempting to optimize alpha (directly getting shape)...")
            try:
                # Try to get the shape directly from optimizealpha
                alpha_shape_polygon = alphashape.optimizealpha(points_for_alphashape_tuples)
                
                if alpha_shape_polygon and not alpha_shape_polygon.is_empty and isinstance(alpha_shape_polygon, (Polygon, MultiPolygon)):
                    print("  optimizealpha successfully returned a shape.")
                    actual_alpha_used_display = "optimized" # We don't get the float alpha value back from this call
                else:
                    print("  WARNING: optimizealpha did not return a valid polygon or returned empty. Will fall back.")
                    # Force fallback by ensuring alpha_shape_polygon is None
                    alpha_shape_polygon = None 
                    raise ValueError("optimizealpha failed to produce a usable polygon.") # To trigger fallback

            except Exception as opt_err: # Catches errors from optimizealpha OR the ValueError above
                print(f"  WARNING: optimizealpha process failed: {opt_err}")
                print(f"  Falling back to default alpha: {default_alpha_if_optimize_fails}") # Uses the passed-in default
                actual_alpha_used_display = f"{default_alpha_if_optimize_fails:.4g}"
                # Calculate with the (hopefully better) default alpha
                alpha_shape_polygon = alphashape.alphashape(points_for_alphashape_tuples, default_alpha_if_optimize_fails)

        # Validate the obtained polygon
        if alpha_shape_polygon is not None:
            if alpha_shape_polygon.is_empty:
                print("  Alpha shape calculation resulted in an empty geometry.")
                alpha_shape_polygon = None
            elif not isinstance(alpha_shape_polygon, (Polygon, MultiPolygon)):
                print(f"  Alpha shape resulted in {type(alpha_shape_polygon).__name__}, not Polygon/MultiPolygon.")
                alpha_shape_polygon = None
            elif not alpha_shape_polygon.is_valid:
                print("  Alpha shape is invalid, attempting buffer(0) fix.")
                alpha_shape_polygon = alpha_shape_polygon.buffer(0)
                if not alpha_shape_polygon.is_valid or alpha_shape_polygon.is_empty: # Check again after buffer
                    print("  Alpha shape invalid/empty even after buffer(0).")
                    alpha_shape_polygon = None
        else:
            print("  No alpha shape polygon was formed by any method.")

    except Exception as e: # Catch-all for unexpected errors during the try block
        print(f"GENERAL ERROR during alpha shape calculation process: {e}"); traceback.print_exc()
        alpha_shape_polygon = None


    # Use actual_alpha_used_display for plotting
    plot_alpha_shape_result(original_points_for_plotting, alpha_shape_polygon, actual_alpha_used_display,
                            target_crs_str, output_dir, f"{output_filename_stem}_plot",
                            show_plots, save_plots, plot_dpi)

    if alpha_shape_polygon:
        print(f"Saving alpha shape to GML: {output_gml_path}")
        try:
            alpha_gdf = gpd.GeoDataFrame(geometry=[alpha_shape_polygon], crs=target_crs_str)
            # Validity check is already done above, but another one here doesn't hurt
            if not alpha_gdf.geometry.is_valid.all():
                alpha_gdf['geometry'] = alpha_gdf.geometry.buffer(0)
            if alpha_gdf.crs is None: alpha_gdf.crs = target_crs_str

            alpha_gdf = alpha_gdf[alpha_gdf.geometry.is_valid & ~alpha_gdf.geometry.is_empty]
            if not alpha_gdf.empty:
                alpha_gdf.to_file(output_gml_path, driver="GML", GML_FEATURE_COLLECTION=True, GML_ID='auto')
                print(f"Successfully saved alpha shape to {output_gml_path}")
                return str(output_gml_path)
            else:
                print("  Skipping save: No valid geometry after checks/fixes for alpha shape.")
                return None
        except Exception as e:
            print(f"ERROR saving alpha shape to GML: {e}"); traceback.print_exc()
            return None
    else:
        print("No valid alpha shape polygon was calculated or it was empty. Skipping GML save.")
        return None

if __name__ == '__main__':
    # Example Usage
    test_output_dir = Path("output_test_alpha_shape")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy sampled points CSV
    dummy_points_data = {
        'x': [356000, 356500, 357000, 356800, 356200, 356050, 356450, 356950],
        'y': [5638000, 5638500, 5638000, 5638200, 5638300, 5638050, 5638450, 5638050]
    }
    dummy_points_csv_path = test_output_dir / "dummy_sampled_points.csv"
    pd.DataFrame(dummy_points_data).to_csv(dummy_points_csv_path, index=False)
    
    # Or test with a list of Shapely Points
    # dummy_shapely_points = [Point(x,y) for x,y in zip(dummy_points_data['x'], dummy_points_data['y'])]

    alpha_gml_path = calculate_and_save_alpha_shape(
        sampled_points_input=str(dummy_points_csv_path), # or dummy_shapely_points
        target_crs_str="EPSG:25832",
        output_dir_str=str(test_output_dir),
        output_filename_stem="my_alpha_shape",
        alpha_parameter=None, # Let it optimize
        # alpha_parameter=0.00002, # Manual test
        show_plots=False, save_plots=True
    )
    if alpha_gml_path:
        print(f"Alpha Shape Test successful. Saved to: {alpha_gml_path}")
    else:
        print("Alpha Shape Test failed or no shape generated.")