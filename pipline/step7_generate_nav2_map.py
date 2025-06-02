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

# Define standard PGM color values for Nav2 maps (when negate=0 and standard thresholds are used)
COLOR_OCCUPIED = 0    # Black, for boundaries (high occupancy probability)
COLOR_UNKNOWN = 128   # Mid-Gray, for unexplored areas (intermediate occupancy probability)
COLOR_FREE = 254      # Almost White, for traversable areas (low occupancy probability)
# Note: Using 254 instead of 255 to avoid potential edge cases with full white if any lib treats it specially.

# extract_polygons_from_gml function remains the same as in your provided code
def extract_polygons_from_gml(gml_path):
    """
    Extracts polygon vertices from a GML file.
    Returns a list, where each element is a list of (x, y) vertex tuples for a polygon.
    Handles both Polygon and MultiPolygon geometries.
    Returns None on read error, empty list if no polygons found or no valid ones.
    """
    all_polygons_vertices_world = []
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
        if isinstance(geometry, Polygon):
            if geometry.geom_type == 'Polygon':
                 exterior_coords = list(geometry.exterior.coords)
                 all_polygons_vertices_world.append(exterior_coords)
        elif isinstance(geometry, MultiPolygon):
            for poly in geometry.geoms: # Changed from geometry.geoms to poly.geoms, likely a typo in original. Corrected to geometry.geoms
                if poly is None or poly.is_empty or not poly.is_valid:
                    continue
                if poly.geom_type == 'Polygon':
                    exterior_coords = list(poly.exterior.coords)
                    all_polygons_vertices_world.append(exterior_coords)
    if not all_polygons_vertices_world:
        print(f"  Warning: No valid Polygon/MultiPolygon vertices extracted from {gml_path}")
    return all_polygons_vertices_world


def create_map_from_polygons_list(
    list_of_bounding_box_polygons_world,
    list_of_free_space_polygons_world,
    map_resolution,
    output_pgm_path_str,
    output_yaml_path_str,
    padding_m=10.0,
    yaml_origin_override=None,
    boundary_thickness_px=1): # New parameter for boundary thickness
    """
    Creates PGM and YAML map files with trinary values.
    - Map image extents are determined by list_of_bounding_box_polygons_world + padding.
    - Background is UNKNOWN (e.g., gray 128).
    - Free space areas (from list_of_free_space_polygons_world) are drawn as FREE (e.g., white 254).
    - Boundaries of free space areas are drawn as OCCUPIED (e.g., black 0).
    - YAML origin can be overridden.
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

    # Initialize image with COLOR_UNKNOWN (e.g., 128 Gray) for unexplored areas.
    image = np.full((map_height_px, map_width_px), COLOR_UNKNOWN, dtype=np.uint8)

    polygons_drawn = 0
    if list_of_free_space_polygons_world:
        for polygon_vertices_world in list_of_free_space_polygons_world:
            polygon_vertices_px = []
            for wx, wy in polygon_vertices_world:
                px = int((wx - pgm_physical_origin_x) / map_resolution)
                # PGM y is inverted from world y: (0,0) is top-left in image, bottom-left in map coordinates
                py = map_height_px - 1 - int((wy - pgm_physical_origin_y) / map_resolution)
                polygon_vertices_px.append([px, py])

            if len(polygon_vertices_px) >= 3:
                try:
                    polygon_np_array = np.array([polygon_vertices_px], dtype=np.int32)
                    # 1. Fill the polygon with COLOR_FREE (e.g., 254 White)
                    cv2.fillPoly(image, [polygon_np_array], COLOR_FREE)
                    # 2. Draw the polygon boundary with COLOR_OCCUPIED (e.g., 0 Black)
                    cv2.polylines(image, [polygon_np_array], isClosed=True, color=COLOR_OCCUPIED, thickness=boundary_thickness_px)
                    polygons_drawn += 1
                except Exception as draw_err:
                    print(f"    Warning: Failed to draw a free-space polygon: {draw_err}")
    else:
        print("  Warning: No free-space polygons provided. Map will be entirely 'unknown'.")

    if polygons_drawn == 0 and list_of_free_space_polygons_world:
        print("  Warning: No free-space polygons were successfully drawn, though some were provided.")


    final_yaml_origin = []
    if yaml_origin_override:
        if len(yaml_origin_override) == 3 and all(isinstance(c, (int, float)) for c in yaml_origin_override):
            final_yaml_origin = yaml_origin_override
            print(f"    Using overridden YAML origin (Local Coords): {final_yaml_origin}")
        else:
            print(f"  Warning: Invalid yaml_origin_override: {yaml_origin_override}. Defaulting YAML origin to PGM physical origin.")
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
    # Standard YAML parameters for Nav2 when PGM uses:
    # - Black (0) for occupied
    # - White (254/255) for free
    # - Gray (e.g., 128) for unknown
    # The map_server with negate=0 interprets:
    #   - low PGM values (darker) as higher probability of being occupied
    #   - high PGM values (lighter) as lower probability of being occupied
    map_yaml_data = {
        'image': output_pgm_path.name,
        'mode': 'trinary',  # Occupied, Free, Unknown
        'resolution': map_resolution,
        'origin': final_yaml_origin,
        'negate': 0,  # 0 means darker pixels are more occupied (PGM black=occupied, PGM white=free)
        'occupied_thresh': 0.65, # PGM values with occupancy prob > 0.65 are occupied
                                 # For negate=0, this effectively means PGM pixel val < (1-0.65)*255 = 89.25
                                 # Our COLOR_OCCUPIED=0 maps to occ_prob=1.0 (from (255-0)/255), which is >0.65.
        'free_thresh': 0.25,     # PGM values with occupancy prob < 0.25 are free
                                 # For negate=0, this effectively means PGM pixel val > (1-0.25)*255 = 191.25
                                 # Our COLOR_FREE=254 maps to occ_prob=(255-254)/255=0.0039, which is <0.25.
                                 # Our COLOR_UNKNOWN=128 maps to occ_prob=(255-128)/255=0.498.
                                 # Since 0.25 < 0.498 < 0.65, it's UNKNOWN.
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
def generate_nav2_map(bounds_gml_input_path_str,
                      free_space_gml_input_path_str,
                      obj_local_frame_origin_world_xy, # Tuple (world_x, world_y) from step6b
                      obj_local_frame_base_z_val,      # Z of object base in local frame (from step6b config)
                      output_dir_str,
                      output_map_basename,
                      map_resolution,
                      map_padding_m,
                      boundary_thickness_px=1): # Pass through boundary thickness
    """
    Main function for Step 7: Creates Nav2 map files.
    - Bounds GML defines map physical size.
    - Free Space GML defines the traversable (white) areas with black boundaries.
    - YAML origin is calculated relative to the local frame origin defined by step6b.
    Returns True on success, False otherwise.
    """
    print(f"\n--- Generating Nav2 Map Files (Aligned with OBJ Local Frame) ---")
    bounds_gml_path = Path(bounds_gml_input_path_str)
    free_space_gml_path = Path(free_space_gml_input_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pgm_path = output_dir / f"{output_map_basename}.pgm"
    output_yaml_path = output_dir / f"{output_map_basename}.yaml"

    print(f"  Bounds GML for PGM extents: {bounds_gml_path.name}")
    print(f"  Free Space GML for traversable area: {free_space_gml_path.name}")
    if obj_local_frame_origin_world_xy:
        print(f"  OBJ Local Frame Origin (World Coords from Step6b): X={obj_local_frame_origin_world_xy[0]:.3f}, Y={obj_local_frame_origin_world_xy[1]:.3f}")
    else:
        print(f"  Warning: OBJ Local Frame Origin (World Coords from Step6b) not provided. Map origin will not be aligned.")
    print(f"  OBJ Base Z in Local Frame (for map Z): {obj_local_frame_base_z_val:.3f}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Map Resolution: {map_resolution} m/px, Padding: {map_padding_m} m")
    print(f"  Boundary Thickness: {boundary_thickness_px} px")


    list_of_bounds_polygons = extract_polygons_from_gml(bounds_gml_path)
    if list_of_bounds_polygons is None:
        print("  Failed to extract polygons for PGM bounds due to GML read error.")
        return False
    if not list_of_bounds_polygons:
        print("  No polygons extracted from Bounds GML. Cannot determine PGM map extents.")
        return False

    all_bounds_raw_vertices = [v for poly_verts in list_of_bounds_polygons for v in poly_verts]
    if not all_bounds_raw_vertices:
        print("  No vertices in bounds polygons for PGM extents calculation."); return False

    min_x_bounds_raw_world = min(v[0] for v in all_bounds_raw_vertices)
    min_y_bounds_raw_world = min(v[1] for v in all_bounds_raw_vertices)

    pgm_bl_world_x = min_x_bounds_raw_world - map_padding_m
    pgm_bl_world_y = min_y_bounds_raw_world - map_padding_m
    print(f"    PGM Bottom-Left (World Coords before padding adjustment): [{min_x_bounds_raw_world:.3f}, {min_y_bounds_raw_world:.3f}]")
    print(f"    PGM Bottom-Left (World Coords with padding): [{pgm_bl_world_x:.3f}, {pgm_bl_world_y:.3f}]")

    yaml_origin_for_create_map = None
    if obj_local_frame_origin_world_xy:
        local_frame_origin_world_x = obj_local_frame_origin_world_xy[0]
        local_frame_origin_world_y = obj_local_frame_origin_world_xy[1]
        yaml_local_x = pgm_bl_world_x - local_frame_origin_world_x
        yaml_local_y = pgm_bl_world_y - local_frame_origin_world_y
        yaml_local_z = obj_local_frame_base_z_val
        yaml_origin_for_create_map = [yaml_local_x, yaml_local_y, yaml_local_z]
        print(f"    Calculated YAML Origin (Local Coords): [{yaml_local_x:.3f}, {yaml_local_y:.3f}, {yaml_local_z:.3f}]")
    else:
        print("    OBJ local frame origin not provided. YAML origin will be PGM's physical world origin.")

    list_of_free_space_polygons = extract_polygons_from_gml(free_space_gml_path)
    if list_of_free_space_polygons is None:
        print("  Failed to extract polygons for free space due to GML read error.")
        return False
    if not list_of_free_space_polygons:
        print("  Warning: No polygons extracted from Free Space GML. Map will be entirely unknown.")

    success = create_map_from_polygons_list(
        list_of_bounding_box_polygons_world=list_of_bounds_polygons,
        list_of_free_space_polygons_world=list_of_free_space_polygons,
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

    test_dir = Path("../output_project/test_step7_data_aligned_trinary")
    test_dir.mkdir(parents=True, exist_ok=True)

    bounds_gml_content = """<?xml version="1.0" encoding="utf-8" ?>
<FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2">
  <featureMember><TargetObject><geometry>
    <gml:Polygon srsName="urn:ogc:def:crs:EPSG::25832">
      <gml:exterior><gml:LinearRing>
          <gml:posList>1000 2000 1100 2000 1100 2050 1000 2050 1000 2000</gml:posList>
      </gml:LinearRing></gml:exterior></gml:Polygon>
  </geometry></TargetObject></featureMember></FeatureCollection>"""
    test_bounds_gml_path = test_dir / "test_bounds_world.gml"
    with open(test_bounds_gml_path, 'w') as f: f.write(bounds_gml_content)

    free_space_gml_content = """<?xml version="1.0" encoding="utf-8" ?>
<FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2">
  <featureMember><AlphaShapeObject><geometry>
    <gml:Polygon srsName="urn:ogc:def:crs:EPSG::25832">
      <gml:exterior><gml:LinearRing>
        <gml:posList>1010 2010 1040 2010 1040 2030 1010 2030 1010 2010</gml:posList>
      </gml:LinearRing></gml:exterior></gml:Polygon>
  </geometry></AlphaShapeObject></featureMember></FeatureCollection>"""
    test_free_space_gml_path = test_dir / "test_free_space_world.gml"
    with open(test_free_space_gml_path, 'w') as f: f.write(free_space_gml_content)

    test_output_directory = "../output_project/nav2_map_output_aligned_trinary"
    test_map_base = "aligned_local_trinary_map"
    test_resolution = 1.0  # 1 meter per pixel for easy visual inspection
    test_padding = 5.0     # 5 meters padding
    test_boundary_thickness = 2 # Make boundary 2 pixels thick for visibility

    sim_obj_local_frame_origin_world_xy = (1005.0, 2000.0)
    sim_obj_local_frame_base_z_val = 0.05

    if test_bounds_gml_path.exists() and test_free_space_gml_path.exists():
        print(f"\nGenerating map with boundary thickness: {test_boundary_thickness}px")
        generate_nav2_map(
            bounds_gml_input_path_str=str(test_bounds_gml_path),
            free_space_gml_input_path_str=str(test_free_space_gml_path),
            obj_local_frame_origin_world_xy=sim_obj_local_frame_origin_world_xy,
            obj_local_frame_base_z_val=sim_obj_local_frame_base_z_val,
            output_dir_str=test_output_directory,
            output_map_basename=test_map_base,
            map_resolution=test_resolution,
            map_padding_m=test_padding,
            boundary_thickness_px=test_boundary_thickness
        )
        print(f"Map generated: {Path(test_output_directory) / (test_map_base + '.pgm')}")
        print(f"YAML generated: {Path(test_output_directory) / (test_map_base + '.yaml')}")

        # Test with default boundary thickness (1px)
        test_map_base_default_thick = test_map_base + "_default_thickness"
        print(f"\nGenerating map with default boundary thickness (1px)")
        generate_nav2_map(
            bounds_gml_input_path_str=str(test_bounds_gml_path),
            free_space_gml_input_path_str=str(test_free_space_gml_path),
            obj_local_frame_origin_world_xy=sim_obj_local_frame_origin_world_xy,
            obj_local_frame_base_z_val=sim_obj_local_frame_base_z_val,
            output_dir_str=test_output_directory,
            output_map_basename=test_map_base_default_thick,
            map_resolution=test_resolution,
            map_padding_m=test_padding
            # boundary_thickness_px will use default from function signature (1px)
        )
        print(f"Map generated: {Path(test_output_directory) / (test_map_base_default_thick + '.pgm')}")
        print(f"YAML generated: {Path(test_output_directory) / (test_map_base_default_thick + '.yaml')}")

    else:
        print(f"Test skipped: One or both input GML files not found.")

    print("--- Test Finished ---")