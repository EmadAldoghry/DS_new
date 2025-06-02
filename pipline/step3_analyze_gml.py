# -*- coding: utf-8 -*-
import time
from pathlib import Path
import pandas as pd # For saving points if needed
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import numpy as np
from helpers import plot_analysis_step # Assuming helpers.py is in the same directory or PYTHONPATH

def analyze_gml_and_sample_points(gml_file_path_str, hull_polygon_gpkg_path_str,
                                  output_dir_str, gml_filename_stem_for_plots,
                                  analysis_offset=0.1, sample_interval=10.0,
                                  show_plots=True, save_plots=True, plot_dpi=150):
    """
    Parses GML, applies filters, samples points along connected lines,
    plots each step, and returns the sampled points as a list of Shapely Points.
    """
    TOTAL_PLOTTING_STEPS = 4
    gml_file_path = Path(gml_file_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Running: GML Analysis for {gml_file_path.name} ---")

    if not gml_file_path.exists():
        print(f"ERROR: GML file not found: {gml_file_path}")
        return None
    try:
        hull_gdf = gpd.read_file(hull_polygon_gpkg_path_str)
        if hull_gdf.empty: raise ValueError("Hull GeoPackage is empty.")
        hull_polygon = hull_gdf.geometry.iloc[0]
        if not hull_polygon.is_valid: hull_polygon = hull_polygon.buffer(0)
        if not hull_polygon.is_valid: raise ValueError("Hull polygon is invalid.")
        # Assuming hull is already in the correct target CRS for analysis
    except Exception as e:
        print(f"ERROR: Could not read or prepare hull polygon from {hull_polygon_gpkg_path_str}: {e}")
        return None
    
    hull_centroid = hull_polygon.centroid

    # --- Step 1: Parsing GML ---
    print(f"\nStep 1/{TOTAL_PLOTTING_STEPS}: Parsing GML file...")
    try:
        gdf = gpd.read_file(gml_file_path)
        if gdf.empty: print("GML parsed but contains no features."); return None
        initial_lines_gdf = gdf[gdf.geometry.apply(lambda g: isinstance(g, LineString))].copy()
        if initial_lines_gdf.empty: print("No LineString features found in GML."); return None
        initial_lines_gdf['is_ring'] = initial_lines_gdf.geometry.apply(lambda g: g.is_ring if g and g.is_valid else False)
        closed_lines = initial_lines_gdf[initial_lines_gdf['is_ring']].geometry.tolist()
        open_lines = initial_lines_gdf[~initial_lines_gdf['is_ring']].geometry.tolist()
    except Exception as e:
        print(f"ERROR: Could not parse GML '{gml_file_path}': {e}"); return None

    plot_analysis_step("Initial Parsed Lines", 1, TOTAL_PLOTTING_STEPS,
        [{'geoms': closed_lines, 'color': 'green', 'label': 'Closed Lines'},
         {'geoms': open_lines, 'color': 'blue', 'label': 'Open Lines'}],
        output_dir, gml_filename_stem_for_plots, hull_polygon, hull_centroid, show_plots, save_plots, plot_dpi)

    # --- Step 2: Filter 1 ---
    print(f"\nStep 2/{TOTAL_PLOTTING_STEPS}: Filter 1 (Intersection with Buffered Rings)...")
    filtered_open_lines_step1 = []
    buffered_polygons_for_plot = []
    if closed_lines:
        valid_buffers_for_check = [Polygon(r.coords).buffer(analysis_offset) for r in closed_lines if r.is_valid and Polygon(r.coords).is_valid]
        valid_buffers_for_check = [b for b in valid_buffers_for_check if b and b.is_valid and not b.is_empty]
        buffered_polygons_for_plot.extend(valid_buffers_for_check)
        if valid_buffers_for_check and open_lines:
            combined_buffers = unary_union(valid_buffers_for_check)
            if not combined_buffers.is_valid: combined_buffers = combined_buffers.buffer(0)
            for line in open_lines:
                if line.is_valid and not line.intersects(combined_buffers):
                    filtered_open_lines_step1.append(line)
        else: filtered_open_lines_step1 = list(open_lines) # Keep all if no buffers or no open lines
    else: filtered_open_lines_step1 = list(open_lines) # Keep all if no closed lines
    print(f"  After Filter 1: {len(filtered_open_lines_step1)} open lines remain.")
    plot_analysis_step("Lines After Filter 1", 2, TOTAL_PLOTTING_STEPS,
        [{'geoms': closed_lines, 'color': 'lightgreen', 'label': 'Closed (Context)', 'alpha':0.5},
         {'geoms': filtered_open_lines_step1, 'color': 'orange', 'label': 'Remaining Open'},
         {'geoms': buffered_polygons_for_plot, 'color': 'lightcoral', 'label': f'Buffers ({analysis_offset}m)', 'lw':0.5, 'ls':'--', 'alpha':0.4}],
        output_dir, gml_filename_stem_for_plots, hull_polygon, hull_centroid, show_plots, save_plots, plot_dpi)

    # --- Step 3: Filter 2 ---
    print(f"\nStep 3/{TOTAL_PLOTTING_STEPS}: Filter 2 (Keep Connected Open Lines)...")
    connected_open_lines = []
    if len(filtered_open_lines_step1) > 1:
        lines_gdf = gpd.GeoDataFrame(geometry=filtered_open_lines_step1)
        lines_gdf['touches_other'] = False
        tree = lines_gdf.sindex
        for i, line_a in lines_gdf.geometry.items():
            if not line_a or not line_a.is_valid: continue
            possible_matches_idx = list(tree.intersection(line_a.bounds))
            for j in [idx for idx in possible_matches_idx if i != idx]:
                line_b = lines_gdf.geometry[j]
                if line_b and line_b.is_valid and line_a.intersects(line_b):
                    lines_gdf.loc[i, 'touches_other'] = True
                    lines_gdf.loc[j, 'touches_other'] = True; break # Found one connection for line_a
        connected_open_lines = lines_gdf[lines_gdf['touches_other']].geometry.tolist()
    # If only one line, it's not "connected" by this filter's definition. If zero, empty list.
    print(f"  After Filter 2: {len(connected_open_lines)} connected open lines remain.")
    plot_analysis_step("Lines After Filter 2", 3, TOTAL_PLOTTING_STEPS,
        [{'geoms': closed_lines, 'color': 'lightgreen', 'label': 'Closed (Context)', 'alpha':0.5},
         {'geoms': connected_open_lines, 'color': 'red', 'label': 'Connected Open Lines'}],
        output_dir, gml_filename_stem_for_plots, hull_polygon, hull_centroid, show_plots, save_plots, plot_dpi)

    # --- Step 4: Sample Points ---
    print(f"\nStep 4/{TOTAL_PLOTTING_STEPS}: Sampling points...")
    sampled_points = []
    if connected_open_lines and sample_interval > 0:
        point_coords_set = set()
        for line in connected_open_lines:
            if not line or not line.is_valid or line.is_empty or line.length == 0: continue
            distances = np.arange(0, line.length, sample_interval)
            for p in [line.interpolate(d) for d in distances] + [Point(line.coords[-1])]: # Add end point
                if p.is_valid: point_coords_set.add((p.x, p.y))
        sampled_points = [Point(coord) for coord in point_coords_set]
    print(f"  Generated {len(sampled_points)} unique valid points.")
    plot_analysis_step(f"Sampled Points (Interval: {sample_interval}m)", 4, TOTAL_PLOTTING_STEPS,
        [{'geoms': connected_open_lines, 'color': 'lightgrey', 'label': 'Connected Lines (Context)', 'alpha':0.7},
         {'geoms': sampled_points, 'color': 'blue', 'label': 'Sampled Points', 'marker':'o', 'ms':4}],
        output_dir, gml_filename_stem_for_plots, hull_polygon, hull_centroid, show_plots, save_plots, plot_dpi)
    
    print(f"\n--- GML Analysis of {gml_file_path.name} complete ---")
    
    # Optionally save sampled points to a CSV for inspection or direct use by next step
    if sampled_points:
        points_df = pd.DataFrame([(p.x, p.y) for p in sampled_points], columns=['x', 'y'])
        csv_out_path = output_dir / f"{gml_filename_stem_for_plots}_sampled_points.csv"
        points_df.to_csv(csv_out_path, index=False)
        print(f"Sampled points saved to: {csv_out_path}")

    return sampled_points # Return list of Shapely Points

if __name__ == '__main__':
    # Example Usage - Requires a GML file and a hull GPKG
    test_output_dir = Path("output_test_gml_analysis")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Dummy GML (replace with a real one for thorough testing)
    dummy_gml_content = """
    <gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2" xmlns:adv="http://www.adv-online.de/namespaces/adv/gid/6.0">
      <gml:featureMember>
        <adv:AX_Strassenverkehr>
          <adv:geometrie>
            <gml:LineString gml:id="LS1">
              <gml:posList>356000 5638000 356500 5638500 357000 5638000</gml:posList>
            </gml:LineString>
          </adv:geometrie>
        </adv:AX_Strassenverkehr>
      </gml:featureMember>
      <gml:featureMember>
        <adv:AX_Strassenverkehr>
          <adv:geometrie>
            <gml:LineString gml:id="LS2_ring">
              <gml:posList>356200 5638200 356300 5638200 356300 5638300 356200 5638300 356200 5638200</gml:posList>
            </gml:LineString>
          </adv:geometrie>
        </adv:AX_Strassenverkehr>
      </gml:featureMember>
    </gml:FeatureCollection>
    """
    dummy_gml_path = test_output_dir / "dummy_data.gml"
    with open(dummy_gml_path, "w") as f:
        f.write(dummy_gml_content)

    # Dummy Hull GPKG
    dummy_hull_path = test_output_dir / "dummy_hull_for_gml_analysis.gpkg"
    from shapely.geometry import Polygon as ShapelyPoly
    poly = ShapelyPoly([(355000, 5637000), (358000, 5637000), (358000, 5640000), (355000, 5640000)])
    gdf_hull = gpd.GeoDataFrame([{'id':1, 'geometry': poly}], crs="EPSG:25832")
    gdf_hull.to_file(dummy_hull_path, driver="GPKG")


    sampled_points_list = analyze_gml_and_sample_points(
        gml_file_path_str=str(dummy_gml_path),
        hull_polygon_gpkg_path_str=str(dummy_hull_path),
        output_dir_str=str(test_output_dir),
        gml_filename_stem_for_plots="dummy_data_analyzed",
        analysis_offset=1, # meters for buffer
        sample_interval=50, # meters
        show_plots=False, save_plots=True # Adjust for testing
    )
    if sampled_points_list:
        print(f"GML Analysis Test: Generated {len(sampled_points_list)} points.")
    else:
        print("GML Analysis Test: No points generated or an error occurred.")