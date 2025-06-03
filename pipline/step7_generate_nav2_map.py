# -*- coding: utf-8 -*-
# step7_generate_nav2_map.py

import geopandas as gpd
import cv2
import numpy as np
import yaml
import math
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path
import traceback
from typing import List, Tuple, Union # For type hinting

# Define standard PGM color values for Nav2 maps
COLOR_OCCUPIED = 0    
COLOR_UNKNOWN = 128   
COLOR_FREE = 254      

def extract_polygons_from_gml(gml_path: Path) -> Union[List[List[Tuple[float, float]]], None]:
    """
    Extracts polygon vertices from a GML file.
    Returns a list, where each element is a list of (x, y) vertex tuples for a polygon's exterior.
    Handles Polygon and MultiPolygon. Returns None on read error, empty list if no valid polygons.
    """
    all_polygons_vertices_world: List[List[Tuple[float, float]]] = []
    try:
        gdf = gpd.read_file(gml_path)
    except Exception as e:
        print(f"  Error reading GML file {gml_path}: {e}")
        traceback.print_exc()
        return None

    if gdf.empty:
        print(f"  Warning: No features found in GML file: {gml_path}")
        return []

    for geometry in gdf.geometry:
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            continue
        
        geoms_to_process = []
        if geometry.geom_type == 'Polygon':
            geoms_to_process.append(geometry)
        elif geometry.geom_type == 'MultiPolygon':
            geoms_to_process.extend(list(geometry.geoms))
        
        for poly in geoms_to_process:
            if poly is not None and poly.geom_type == 'Polygon' and \
               not poly.is_empty and poly.is_valid:
                exterior_coords = list(poly.exterior.coords)
                all_polygons_vertices_world.append(exterior_coords)
                
    if not all_polygons_vertices_world:
        print(f"  Warning: No valid Polygon vertices extracted from {gml_path}")
    return all_polygons_vertices_world


def shapely_geom_to_vertex_lists(shapely_geom: Union[Polygon, MultiPolygon]) -> List[List[Tuple[float, float]]]:
    """Converts a Shapely Polygon or MultiPolygon to a list of exterior vertex lists."""
    vertex_lists: List[List[Tuple[float, float]]] = []
    if shapely_geom is None or shapely_geom.is_empty or not shapely_geom.is_valid:
        return vertex_lists

    geoms_to_process = []
    if shapely_geom.geom_type == 'Polygon':
        geoms_to_process.append(shapely_geom)
    elif shapely_geom.geom_type == 'MultiPolygon':
        geoms_to_process.extend(list(shapely_geom.geoms))
    
    for poly in geoms_to_process:
        if poly is not None and poly.geom_type == 'Polygon' and \
           not poly.is_empty and poly.is_valid:
            vertex_lists.append(list(poly.exterior.coords))
    return vertex_lists


def create_map_from_polygons_list(
    list_of_bounding_box_polygons_world: List[List[Tuple[float, float]]],
    list_of_free_space_polygons_world: List[List[Tuple[float, float]]], # This will come from the Shapely difference
    map_resolution: float,
    output_pgm_path_str: str,
    output_yaml_path_str: str,
    padding_m: float = 10.0,
    yaml_origin_override: Union[List[float], None] = None,
    boundary_thickness_px: int = 1
) -> bool:
    """
    Creates PGM and YAML map files with trinary values.
    """
    if not list_of_bounding_box_polygons_world:
        print("  Error: No bounding box polygon vertices provided. Cannot determine map extent.")
        return False

    all_bbox_vertices = [v for poly_verts in list_of_bounding_box_polygons_world for v in poly_verts]
    if not all_bbox_vertices:
        print("  Error: No vertices found in bounding_box_polygons. Cannot create map.")
        return False

    min_x_bbox_world = min(v[0] for v in all_bbox_vertices)
    max_x_bbox_world = max(v[0] for v in all_bbox_vertices)
    min_y_bbox_world = min(v[1] for v in all_bbox_vertices)
    max_y_bbox_world = max(v[1] for v in all_bbox_vertices)

    pgm_physical_origin_x = min_x_bbox_world - padding_m
    pgm_physical_origin_y = min_y_bbox_world - padding_m

    map_width_m = (max_x_bbox_world + padding_m) - pgm_physical_origin_x
    map_height_m = (max_y_bbox_world + padding_m) - pgm_physical_origin_y
    map_width_px = int(math.ceil(map_width_m / map_resolution))
    map_height_px = int(math.ceil(map_height_m / map_resolution))

    if map_width_px <= 0 or map_height_px <= 0:
        print(f"  Error: Invalid PGM map dimensions calculated ({map_width_px}x{map_height_px}).")
        return False

    print(f"    PGM Map Dimensions: {map_width_px}px width, {map_height_px}px height")
    print(f"    PGM Physical Origin (world coords of PGM bottom-left): [{pgm_physical_origin_x:.3f}, {pgm_physical_origin_y:.3f}]")

    image = np.full((map_height_px, map_width_px), COLOR_UNKNOWN, dtype=np.uint8)

    polygons_drawn = 0
    if list_of_free_space_polygons_world: # Check if there's anything to draw
        for polygon_vertices_world in list_of_free_space_polygons_world:
            polygon_vertices_px = []
            for wx, wy in polygon_vertices_world:
                px = int((wx - pgm_physical_origin_x) / map_resolution)
                py = map_height_px - 1 - int((wy - pgm_physical_origin_y) / map_resolution)
                polygon_vertices_px.append([px, py])

            if len(polygon_vertices_px) >= 3:
                try:
                    polygon_np_array = np.array([polygon_vertices_px], dtype=np.int32)
                    cv2.fillPoly(image, [polygon_np_array], COLOR_FREE)
                    cv2.polylines(image, [polygon_np_array], isClosed=True, color=COLOR_OCCUPIED, thickness=boundary_thickness_px)
                    polygons_drawn += 1
                except Exception as draw_err:
                    print(f"    Warning: Failed to draw a free-space polygon: {draw_err}")
    else:
        print("  Warning: No free-space polygons provided or calculable. Map will be entirely 'unknown'.")

    if polygons_drawn == 0 and list_of_free_space_polygons_world: # list_of_free_space_polygons_world might be non-empty but failed to draw
        print("  Warning: No free-space polygons were successfully drawn, though some were provided/calculated.")
    elif polygons_drawn > 0:
         print(f"  Successfully drew {polygons_drawn} free-space polygon(s).")


    final_yaml_origin: List[float] = []
    if yaml_origin_override:
        if len(yaml_origin_override) == 3 and all(isinstance(c, (int, float)) for c in yaml_origin_override):
            final_yaml_origin = yaml_origin_override
            print(f"    Using overridden YAML origin (Local Coords): {final_yaml_origin}")
        else:
            print(f"  Warning: Invalid yaml_origin_override: {yaml_origin_override}. Defaulting YAML origin to PGM physical origin (world).")
            final_yaml_origin = [pgm_physical_origin_x, pgm_physical_origin_y, 0.0]
    else:
        final_yaml_origin = [pgm_physical_origin_x, pgm_physical_origin_y, 0.0]
        print(f"    Using PGM physical origin as YAML origin (World Coords): {final_yaml_origin}")

    output_pgm_path = Path(output_pgm_path_str)
    try:
        cv2.imwrite(str(output_pgm_path), image)
        print(f"    Saved PGM map: {output_pgm_path.name}")
    except Exception as e_pgm:
        print(f"  Error saving PGM map to {output_pgm_path}: {e_pgm}")
        traceback.print_exc(); return False

    output_yaml_path = Path(output_yaml_path_str)
    map_yaml_data = {
        'image': output_pgm_path.name,
        'mode': 'trinary',  
        'resolution': map_resolution,
        'origin': final_yaml_origin,
        'negate': 0,  
        'occupied_thresh': 0.65, 
        'free_thresh': 0.25,     
    }
    try:
        with open(output_yaml_path, 'w') as f:
            yaml.dump(map_yaml_data, f, sort_keys=False, default_flow_style=None)
        print(f"    Saved YAML map metadata: {output_yaml_path.name}")
        return True
    except Exception as e_yaml:
        print(f"  Error saving YAML map metadata to {output_yaml_path}: {e_yaml}")
        traceback.print_exc(); return False


# --- Main function for pipeline integration ---
def generate_nav2_map(bounds_gml_input_path_str: str,
                      free_space_shapely_geom_world: Union[Polygon, MultiPolygon], # CHANGED
                      obj_local_frame_origin_world_xy: Tuple[float, float], 
                      obj_local_frame_base_z_val: float,      
                      output_dir_str: str,
                      output_map_basename: str,
                      map_resolution: float,
                      map_padding_m: float,
                      boundary_thickness_px: int = 1):
    """
    Main function for Step 7: Creates Nav2 map files.
    - Bounds GML defines map physical size.
    - free_space_shapely_geom_world (Shapely object) defines the traversable areas.
    - YAML origin is calculated relative to the local frame origin defined by step6b.
    Returns True on success, False otherwise.
    """
    print(f"\n--- Generating Nav2 Map Files (Aligned with OBJ Local Frame) ---")
    bounds_gml_path = Path(bounds_gml_input_path_str)
    # free_space_gml_path_str is removed
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pgm_path = output_dir / f"{output_map_basename}.pgm"
    output_yaml_path = output_dir / f"{output_map_basename}.yaml"

    print(f"  Bounds GML for PGM extents: {bounds_gml_path.name}")
    print(f"  Free Space from provided Shapely geometry (type: {free_space_shapely_geom_world.geom_type if free_space_shapely_geom_world else 'None'})")
    if obj_local_frame_origin_world_xy:
        print(f"  OBJ Local Frame Origin (World Coords from Step6b): X={obj_local_frame_origin_world_xy[0]:.3f}, Y={obj_local_frame_origin_world_xy[1]:.3f}")
    else:
        print(f"  Warning: OBJ Local Frame Origin (World Coords from Step6b) not provided. Map origin will not be aligned.")
    print(f"  OBJ Base Z in Local Frame (for map Z): {obj_local_frame_base_z_val:.3f}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Map Resolution: {map_resolution} m/px, Padding: {map_padding_m} m")
    print(f"  Boundary Thickness: {boundary_thickness_px} px")

    # --- Get Bounds Polygons (for map extents) ---
    list_of_bounds_polygons_vertices = extract_polygons_from_gml(bounds_gml_path)
    if list_of_bounds_polygons_vertices is None: # GML read error
        print("  Failed to extract polygons for PGM bounds due to GML read error.")
        return False
    if not list_of_bounds_polygons_vertices: # Empty GML or no polygons
        print("  No polygons extracted from Bounds GML. Cannot determine PGM map extents.")
        return False

    all_bounds_raw_vertices = [v for poly_verts in list_of_bounds_polygons_vertices for v in poly_verts]
    if not all_bounds_raw_vertices:
        print("  No vertices in bounds polygons for PGM extents calculation."); return False

    min_x_bounds_raw_world = min(v[0] for v in all_bounds_raw_vertices)
    min_y_bounds_raw_world = min(v[1] for v in all_bounds_raw_vertices)

    pgm_bl_world_x = min_x_bounds_raw_world - map_padding_m
    pgm_bl_world_y = min_y_bounds_raw_world - map_padding_m
    print(f"    PGM Bottom-Left (World Coords before padding adjustment): [{min_x_bounds_raw_world:.3f}, {min_y_bounds_raw_world:.3f}]")
    print(f"    PGM Bottom-Left (World Coords with padding): [{pgm_bl_world_x:.3f}, {pgm_bl_world_y:.3f}]")
    
    # --- Prepare YAML Origin ---
    yaml_origin_for_create_map: Union[List[float], None] = None
    if obj_local_frame_origin_world_xy:
        local_frame_origin_world_x = obj_local_frame_origin_world_xy[0]
        local_frame_origin_world_y = obj_local_frame_origin_world_xy[1]
        yaml_local_x = pgm_bl_world_x - local_frame_origin_world_x
        yaml_local_y = pgm_bl_world_y - local_frame_origin_world_y
        yaml_local_z = obj_local_frame_base_z_val # This is the Z of the OBJ's base in its local frame
        yaml_origin_for_create_map = [yaml_local_x, yaml_local_y, yaml_local_z]
        print(f"    Calculated YAML Origin (Local Coords): [{yaml_local_x:.3f}, {yaml_local_y:.3f}, {yaml_local_z:.3f}]")
    else:
        print("    OBJ local frame origin not provided. YAML origin will be PGM's physical world origin (world coordinates).")

    # --- Convert Free Space Shapely Geometry to Vertex Lists ---
    list_of_free_space_vertices = shapely_geom_to_vertex_lists(free_space_shapely_geom_world)
    if not list_of_free_space_vertices and free_space_shapely_geom_world and not free_space_shapely_geom_world.is_empty :
        print("  Warning: Provided free_space_shapely_geom_world did not yield any vertex lists for drawing.")
    elif not free_space_shapely_geom_world or free_space_shapely_geom_world.is_empty:
        print("  Info: free_space_shapely_geom_world is None or empty. Map will likely be all 'unknown'.")


    # --- Create the Map ---
    success = create_map_from_polygons_list(
        list_of_bounding_box_polygons_world=list_of_bounds_polygons_vertices,
        list_of_free_space_polygons_world=list_of_free_space_vertices, # Use converted list
        map_resolution=map_resolution,
        output_pgm_path_str=str(output_pgm_path),
        output_yaml_path_str=str(output_yaml_path),
        padding_m=map_padding_m,
        yaml_origin_override=yaml_origin_for_create_map,
        boundary_thickness_px=boundary_thickness_px
    )

    if success:
        print(f"  Nav2 map generation successful.")
    else:
        print(f"  Nav2 map generation failed.")
    return success

# --- Example Usage (for testing this module independently) ---
if __name__ == '__main__':
    print("--- Testing Step 7: Generate Nav2 Map (Aligned with OBJ Local Frame) ---")

    # Use a directory relative to this script for test outputs
    script_dir = Path(__file__).parent
    test_base_dir = script_dir.parent / "output_project_tests" / "test_step7_nav2_map_custom_free_space"
    test_base_dir.mkdir(parents=True, exist_ok=True)

    # Dummy Bounds GML (e.g., from edges.gml)
    bounds_gml_content = """<?xml version="1.0" encoding="utf-8" ?>
<FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2">
  <featureMember><TargetObject><geometry>
    <gml:Polygon srsName="urn:ogc:def:crs:EPSG::25832">
      <gml:exterior><gml:LinearRing>
          <gml:posList>1000 2000 1100 2000 1100 2050 1000 2050 1000 2000</gml:posList>
      </gml:LinearRing></gml:exterior></gml:Polygon>
  </geometry></TargetObject></featureMember></FeatureCollection>"""
    test_bounds_gml_path = test_base_dir / "test_bounds_world.gml"
    with open(test_bounds_gml_path, 'w') as f: f.write(bounds_gml_content)

    # Create dummy Shapely geometries for free space (simulating base.difference(tool))
    # Base polygon (matches bounds_gml_content for this simple test)
    base_poly_world = Polygon([(1000, 2000), (1100, 2000), (1100, 2050), (1000, 2050), (1000, 2000)])
    # Tool polygon (an area to be "cut out" from the base)
    tool_poly_world = Polygon([(1020, 2010), (1080, 2010), (1080, 2040), (1020, 2040), (1020, 2010)])
    
    # Calculate the actual free space
    actual_free_space_geom_world = base_poly_world.difference(tool_poly_world)
    if not actual_free_space_geom_world.is_valid:
        actual_free_space_geom_world = actual_free_space_geom_world.buffer(0)

    test_output_directory = test_base_dir / "nav2_map_output" # Store output in a subfolder
    test_map_base = "cut_area_map"
    test_resolution = 0.5  # meters per pixel
    test_padding = 5.0     # meters padding
    test_boundary_thickness = 1 

    # Simulate values that would come from Step 6b
    sim_obj_local_frame_origin_world_xy = (1000.0, 2000.0) # Example: OBJ origin is at min_x, min_y of bounds
    sim_obj_local_frame_base_z_val = 0.05 # Example: OBJ base is slightly above local XY plane

    if test_bounds_gml_path.exists() and actual_free_space_geom_world and not actual_free_space_geom_world.is_empty:
        print(f"\nGenerating map with boundary thickness: {test_boundary_thickness}px")
        print(f"  Input bounds GML: {test_bounds_gml_path}")
        print(f"  Calculated free space type: {actual_free_space_geom_world.geom_type}")

        generate_nav2_map(
            bounds_gml_input_path_str=str(test_bounds_gml_path),
            free_space_shapely_geom_world=actual_free_space_geom_world, # Pass Shapely object
            obj_local_frame_origin_world_xy=sim_obj_local_frame_origin_world_xy,
            obj_local_frame_base_z_val=sim_obj_local_frame_base_z_val,
            output_dir_str=str(test_output_directory),
            output_map_basename=test_map_base,
            map_resolution=test_resolution,
            map_padding_m=test_padding,
            boundary_thickness_px=test_boundary_thickness
        )
        print(f"Map generated: {test_output_directory / (test_map_base + '.pgm')}")
        print(f"YAML generated: {test_output_directory / (test_map_base + '.yaml')}")
        print("Please inspect the PGM. It should show the 'base_poly_world' area as white,")
        print("with the 'tool_poly_world' area within it appearing as gray (unknown/cut-out).")

    else:
        print(f"Test skipped: Bounds GML not found or calculated free space is invalid/empty.")

    print("--- Test Finished ---")