# -*- coding: utf-8 -*-
from pathlib import Path
import geopandas as gpd
import pyproj
import osmnx as ox
import traceback

def fetch_clip_and_save_osm_streets(hull_polygon_gpkg_path_str, target_crs_str, out_dir_str):
    """
    Fetches street data from OpenStreetMap within the hull, clips it, and saves as GPKG.
    """
    print("\n--- Running: Fetching and Clipping OpenStreetMap Streets ---")
    output_dir = Path(out_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "osm_streets_clipped.gpkg"
    target_crs_obj = pyproj.CRS.from_user_input(target_crs_str)

    try:
        hull_gdf = gpd.read_file(hull_polygon_gpkg_path_str)
        if hull_gdf.empty: raise ValueError("Hull GeoPackage is empty.")
        hull_polygon = hull_gdf.geometry.iloc[0]
        if not hull_polygon.is_valid: hull_polygon = hull_polygon.buffer(0)
        if not hull_polygon.is_valid: raise ValueError("Hull polygon is invalid.")
        
        # Ensure hull is in target_crs_obj before converting to 4326 for OSM
        if hull_gdf.crs and not hull_gdf.crs.equals(target_crs_obj):
            hull_gdf = hull_gdf.to_crs(target_crs_obj)
            hull_polygon = hull_gdf.geometry.iloc[0]
        elif not hull_gdf.crs: # If hull GPKG had no CRS, assume it's already target_crs
             hull_gdf.crs = target_crs_obj


    except Exception as e:
        print(f"ERROR: Could not read or prepare hull polygon from {hull_polygon_gpkg_path_str}: {e}")
        return None

    try:
        ox.settings.log_console = False # Quieter OSMnx
        ox.settings.use_cache = True
        ox.settings.requests_timeout = 180

        hull_gdf_4326 = hull_gdf.to_crs("EPSG:4326")
        polygon_4326 = hull_gdf_4326.geometry.iloc[0]
        
        # Buffer slightly for OSM query to ensure edges are captured
        buffered_poly_4326 = polygon_4326.buffer(0.0005) # Approx 50m in degrees

        graph = ox.graph_from_polygon(buffered_poly_4326, network_type='drive', simplify=True, truncate_by_edge=True)
        print(f"  Downloaded OSM graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        
        edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        if edges_gdf.empty: print("  No street edges found from OSM."); return None
        
        edges_gdf = edges_gdf[edges_gdf.geometry.type == 'LineString'].copy()
        if edges_gdf.empty: print("  No LineString street geometries found."); return None

        edges_gdf_proj = edges_gdf.to_crs(target_crs_obj)
        
        # Clip to the original, non-buffered hull in target_crs
        precise_hull_gdf = gpd.GeoDataFrame(geometry=[hull_polygon], crs=target_crs_obj)
        clipped_streets = gpd.clip(edges_gdf_proj, precise_hull_gdf, keep_geom_type=True)
        
        clipped_streets = clipped_streets[clipped_streets.geometry.is_valid & ~clipped_streets.geometry.is_empty].copy()
        if 'length' in clipped_streets.columns: clipped_streets['length_orig_osm'] = clipped_streets['length']
        clipped_streets['length'] = clipped_streets.geometry.length


        if clipped_streets.empty: print("  No valid street segments remain after clipping."); return None
        print(f"  {len(clipped_streets)} valid street segments remain after clipping and cleaning.")

        columns_to_keep = ['osmid', 'highway', 'name', 'geometry', 'oneway', 'maxspeed', 'lanes', 'length']
        columns_present = [col for col in columns_to_keep if col in clipped_streets.columns]
        
        clipped_streets[columns_present].to_file(out_path, driver="GPKG", layer="streets")
        print(f"  OSM streets saved successfully to {out_path}")
        return str(out_path)

    except ImportError: print("ERROR: 'osmnx' library is required."); return None
    except Exception as e:
        print(f"ERROR during OSM street processing: {e}"); traceback.print_exc()
        return None

if __name__ == '__main__':
    test_output_dir = Path("output_test_osm")
    test_output_dir.mkdir(exist_ok=True)
    
    dummy_hull_path = test_output_dir / "dummy_hull_osm.gpkg"
    from shapely.geometry import Polygon
    # Using smaller area for faster OSM test
    poly = Polygon([(356000, 5638000), (357000, 5638000), (357000, 5639000), (356000, 5639000)])
    gdf_hull = gpd.GeoDataFrame([{'id':1, 'geometry': poly}], crs="EPSG:25832")
    gdf_hull.to_file(dummy_hull_path, driver="GPKG")

    osm_gpkg_path = fetch_clip_and_save_osm_streets(
        hull_polygon_gpkg_path_str=str(dummy_hull_path),
        target_crs_str='EPSG:25832',
        out_dir_str=str(test_output_dir)
    )
    if osm_gpkg_path:
        print(f"OSM Fetch Test successful. Saved to: {osm_gpkg_path}")