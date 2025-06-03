# -*- coding: utf-8 -*-
import os
import time
from pathlib import Path
import math
import pandas as pd # For dummy CSV and GML creation
import geopandas as gpd # For dummy GML creation
from shapely.geometry import Polygon, MultiPolygon # For dummy GML creation and operations
from shapely.ops import unary_union
import traceback # For more detailed error printing in steps
import pyproj
from typing import Union # For type hinting

# Import functions from our step modules
from step5_generate_texture import generate_texture_from_polygon
from step5b_mark_defects_on_texture import mark_defects_on_texture
from step5c_mark_cracks_on_texture import mark_cracks_from_gml_on_texture
from step6_generate_cut_obj_model import generate_cut_obj_model
from step6b_transform_obj import transform_obj_file
from step7_generate_nav2_map import generate_nav2_map 
from step7b_generate_waypoints_yaml import generate_waypoints_yaml
from step8_generate_gazebo_world import create_gazebo_model_and_world


# --- Main Configuration ---
# Input Data
GML_INPUT_DIR = "gml_output" 
ROI_GML_FILENAME = "roi.gml" # This will now define the free space for Nav2 map
EDGES_GML_FILENAME = "edges.gml" # This will define the overall bounds for Nav2 map
CRACKS_GML_FILENAME = "cracks.gml" 
CSV_FILE = 'defect_coordinates.csv' 

# CRS Configuration
SOURCE_CRS = 'EPSG:4326'  
TARGET_CRS = 'EPSG:25832' 

# -- Step 5 Config (Texture) --
OUTPUT_TEXTURE_FILENAME = "road_texture.png"
WMS_TEXTURE_URL = "https://www.wms.nrw.de/geobasis/wms_nw_dop"
WMS_TEXTURE_LAYER = "nw_dop_rgb"
WMS_TEXTURE_VERSION = '1.3.0'
WMS_TEXTURE_FORMAT = 'image/tiff' 
WMS_TEXTURE_WIDTH = 5000
WMS_TEXTURE_HEIGHT = 5000
WMS_TEXTURE_TARGET_CRS = TARGET_CRS
WMS_BBOX_PADDING_METERS = 20.0
POLYGON_CRS_FALLBACK_FOR_TEXTURE = TARGET_CRS
TEXTURE_FILL_COLOR_RGB = [128, 128, 128]

# --- Config for Step 5b (Defect Marking) ---
MARK_DEFECTS_ON_TEXTURE = True

# --- Config for Step 5C (Crack Marking) ---
MARK_CRACKS_ON_TEXTURE = True 
CRACK_MARKING_COLOR_BGR = (0, 0, 0) 

# --- Step 6 Configuration (GML to OBJ Generation) ---
BASE_EXTRUSION_CUT_M = -0.3  
TOOL_EXTRUSION_CUT_M = -0.1  
CUT_SIMPLIFY_TOLERANCE_M = 0.01 # General simplification for 3D model parts
CUT_OBJ_OUTPUT_FILENAME = "final_cut_road_model.obj"
CUT_MTL_OUTPUT_FILENAME = "final_cut_road_model.mtl"
CUT_MODEL_OUTPUT_SUBDIR = "cut_model_output"
CONVERT_GENERATE_VT = True
CONVERT_Z_TOLERANCE = 0.01
CONVERT_MATERIAL_TOP = "RoadSurface"
CONVERT_MATERIAL_BOTTOM = "RoadBottom"
CONVERT_MATERIAL_SIDES = "RoadSides"

# --- Step 6b Configuration (OBJ Transformation) ---
TRANSFORM_Z_ADDITIONAL_OFFSET = 0.0
TRANSFORMED_OBJ_OUTPUT_FILENAME = "model.obj"

# --- Step 7 Configuration (Nav2 Map) ---
NAV2_MAP_OUTPUT_BASENAME = "interactive_area_nav2_map"
NAV2_MAP_RESOLUTION = 0.05
NAV2_MAP_PADDING_M = 5.0 # Padding around the EDGES_GML for map extent
NAV2_MAP_SIMPLIFY_TOLERANCE_M = 0.02 # Simplification for ROI GML before drawing on map
NAV2_MAP_OUTPUT_SUBDIR = "nav2_map_output"

# --- Step 7b Configuration (Nav2 Waypoints) ---
WAYPOINTS_OUTPUT_YAML_FILENAME = "nav_waypoints.yaml"
WAYPOINTS_DEFAULT_ORIENTATION_EULER_DEG = [0.0, 0.0, 0.0]
WAYPOINTS_MAP_FRAME_ID = "map"

# --- Step 8 Configuration (Gazebo World) ---
GAZEBO_OUTPUT_SUBDIR = "gazebo_output"
GAZEBO_MODEL_NAME = "pipeline_road_model"
GAZEBO_WORLD_FILENAME = "pipeline_generated.world"

# General Output & Plotting
OUTPUT_DIR_BASE = 'output_project_interactive' 
SHOW_PLOTS_ALL_STEPS = False
SAVE_PLOTS_ALL_STEPS = True
PLOT_DPI_ALL_STEPS = 150

# Helper function to load GML and return a single (Multi)Polygon in target CRS
def load_gml_to_shapely_geom(gml_path_str: str, target_crs_str: str, simplify_tolerance: float = 0.0) -> Union[Polygon, MultiPolygon, None]:
    gml_path = Path(gml_path_str)
    if not gml_path.exists():
        print(f"  ERROR: GML file not found for Shapely conversion: {gml_path}")
        return None
    try:
        gdf = gpd.read_file(gml_path)
        if gdf.empty:
            print(f"  WARNING: GML file {gml_path.name} is empty.")
            return None

        target_crs_obj = pyproj.CRS.from_user_input(target_crs_str)
        if gdf.crs and not gdf.crs.equals(target_crs_obj):
            gdf = gdf.to_crs(target_crs_obj)
        elif not gdf.crs: 
            print(f"  WARNING: GML {gml_path.name} has no CRS. Assuming it's {target_crs_str}.")
            gdf.crs = target_crs_obj

        valid_geoms = []
        for geom in gdf.geometry:
            if geom is not None and geom.is_valid and not geom.is_empty:
                if geom.geom_type == 'Polygon':
                    valid_geoms.append(geom)
                elif geom.geom_type == 'MultiPolygon':
                    valid_geoms.extend(list(g for g in geom.geoms if g.is_valid and not g.is_empty and g.geom_type == 'Polygon'))
        
        if not valid_geoms:
            print(f"  WARNING: No valid Polygon features found in {gml_path.name} after filtering.")
            return None
        
        unified_geom = unary_union(valid_geoms)
        if not unified_geom.is_valid:
            unified_geom = unified_geom.buffer(0) 
        
        if simplify_tolerance > 0:
            print(f"    Simplifying {gml_path.name} with tolerance {simplify_tolerance}m for Shapely object.")
            simplified_geom = unified_geom.simplify(simplify_tolerance, preserve_topology=True)
            if simplified_geom.is_valid and not simplified_geom.is_empty:
                unified_geom = simplified_geom.buffer(0) 
            else:
                print(f"  WARNING: Simplification of {gml_path.name} failed or resulted in empty geom. Using unsimplified.")

        if unified_geom.is_empty or not unified_geom.is_valid or unified_geom.geom_type not in ('Polygon', 'MultiPolygon'):
            print(f"  WARNING: Could not produce a single valid Polygon/MultiPolygon from {gml_path.name}. Type: {unified_geom.geom_type}")
            return None
        
        print(f"  Successfully loaded and processed {gml_path.name} into a {unified_geom.geom_type}.")
        return unified_geom
        
    except Exception as e:
        print(f"  ERROR loading GML {gml_path.name} to Shapely: {e}")
        traceback.print_exc()
        return None

def main():
    print("--- Orchestrator Script Start (Interactive GML Input Mode) ---")
    start_time_script = time.time()

    pipeline_output_dir = Path(OUTPUT_DIR_BASE)
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"All pipeline outputs will be saved in: {pipeline_output_dir.resolve()}")

    gml_input_base_path = Path(GML_INPUT_DIR)
    roi_gml_path = gml_input_base_path / ROI_GML_FILENAME
    edges_gml_path = gml_input_base_path / EDGES_GML_FILENAME
    cracks_gml_path = gml_input_base_path / CRACKS_GML_FILENAME

    # --- GML File Checks ---
    gml_paths_to_check = {
        "ROI": roi_gml_path, # Needed for Nav2 free space
        "Edges": edges_gml_path, # Needed for Nav2 bounds & texture
    }
    if MARK_CRACKS_ON_TEXTURE: 
         gml_paths_to_check["Cracks"] = cracks_gml_path

    all_gmls_found = True
    for name, gml_p in gml_paths_to_check.items():
        if not gml_p.exists():
            print(f"FATAL ERROR: {name} GML file not found at {gml_p}")
            all_gmls_found = False
    if not all_gmls_found:
        return

    print(f"Using ROI GML (for Nav2 free space & 3D cut tool): {roi_gml_path.resolve()}")
    print(f"Using Edges GML (for Nav2 bounds & 3D base/texture): {edges_gml_path.resolve()}")
    if cracks_gml_path.exists():
        print(f"Using Cracks GML (for texture marking): {cracks_gml_path.resolve()}")

    print("\n=== Steps 1-4 (Initial Data Processing) SKIPPED: Using direct GML inputs. ===")

    model_assets_output_dir = pipeline_output_dir / CUT_MODEL_OUTPUT_SUBDIR
    model_assets_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nModel assets (OBJ/MTL/PNG) will be in: {model_assets_output_dir.resolve()}")

    # --- Step 5: Texture Generation (based on EDGES_GML) ---
    base_texture_path_str = None
    cropped_texture_transform = None
    cropped_texture_crs_obj = None
    print(f"\n=== STEP 5: Generating Texture (based on {edges_gml_path.name}) ===")
    try:
        base_texture_path_str, cropped_texture_transform, cropped_texture_crs_obj = generate_texture_from_polygon(
            polygon_gml_path_str=str(edges_gml_path), # Texture covers the full road area
            output_dir_str=str(model_assets_output_dir), 
            output_texture_filename=OUTPUT_TEXTURE_FILENAME,
            wms_url=WMS_TEXTURE_URL, wms_layer=WMS_TEXTURE_LAYER,
            wms_version=WMS_TEXTURE_VERSION, wms_format=WMS_TEXTURE_FORMAT,
            wms_width=WMS_TEXTURE_WIDTH, wms_height=WMS_TEXTURE_HEIGHT,
            target_wms_crs_str=WMS_TEXTURE_TARGET_CRS,
            wms_bbox_padding=WMS_BBOX_PADDING_METERS,
            polygon_crs_fallback_str=TARGET_CRS, 
            fill_color_rgb=TEXTURE_FILL_COLOR_RGB,
            show_plots=SHOW_PLOTS_ALL_STEPS, save_plots=SAVE_PLOTS_ALL_STEPS, plot_dpi=PLOT_DPI_ALL_STEPS
        )
        if base_texture_path_str and cropped_texture_transform and cropped_texture_crs_obj:
            print(f"  Base Texture PNG Saved: {base_texture_path_str}")
            print(f"  Texture CRS: {cropped_texture_crs_obj.srs}")
        else:
            print("  ERROR: Texture generation in Step 5 failed or did not return all necessary info.")
    except Exception as e:
        print(f"ERROR in Step 5 (Texture Generation): {e}")
        traceback.print_exc()

    final_texture_path = Path(base_texture_path_str) if base_texture_path_str else None

    # --- Step 5b: Mark Defects on Texture (from CSV) ---
    if MARK_DEFECTS_ON_TEXTURE:
        if final_texture_path and final_texture_path.exists() and \
           cropped_texture_transform and cropped_texture_crs_obj and Path(CSV_FILE).exists():
            print("\n=== STEP 5b: Marking Defect Polygons (from CSV) on Texture ===")
            try:
                success_step5b = mark_defects_on_texture(
                    base_texture_path_str=str(final_texture_path),
                    texture_affine_transform=cropped_texture_transform,
                    texture_crs_pyproj_obj=cropped_texture_crs_obj,
                    csv_path_str=CSV_FILE,
                    defect_color_bgr=(0, 0, 0) 
                )
                if success_step5b:
                    print(f"  Defect polygons from CSV marked on texture: {final_texture_path}")
                else:
                    print("  Warning: Defect polygon marking (CSV) on texture failed.")
            except Exception as e_defect_mark:
                print(f"ERROR during Step 5b (Defect CSV Marking): {e_defect_mark}")
                traceback.print_exc()
        else:
            # ... (skip reasons as before) ...
            print("Skipping Step 5b (Defect CSV Marking):")
            if not (final_texture_path and final_texture_path.exists() and cropped_texture_transform and cropped_texture_crs_obj):
                 print("  - Base texture or its georeferencing info not available from Step 5.")
            if not Path(CSV_FILE).exists():
                 print(f"  - Input defect CSV file '{CSV_FILE}' not found.")
    else:
        print("Skipping Step 5b (Defect CSV Marking) as per configuration.")

    # --- Step 5c: Mark Cracks from GML on Texture ---
    if MARK_CRACKS_ON_TEXTURE:
        if final_texture_path and final_texture_path.exists() and \
           cropped_texture_transform and cropped_texture_crs_obj and \
           cracks_gml_path.exists(): 
            print("\n=== STEP 5c: Marking Crack Polygons (from GML) on Texture ===")
            try:
                success_step5c = mark_cracks_from_gml_on_texture(
                    base_texture_path_str=str(final_texture_path),
                    texture_affine_transform=cropped_texture_transform,
                    texture_crs_pyproj_obj=cropped_texture_crs_obj,
                    cracks_gml_path_str=str(cracks_gml_path),
                    texture_fallback_crs_str=POLYGON_CRS_FALLBACK_FOR_TEXTURE, 
                    crack_color_bgr=CRACK_MARKING_COLOR_BGR
                )
                if success_step5c:
                    print(f"  Crack polygons from GML marked on texture: {final_texture_path}")
                else:
                    print("  Warning: Crack polygon marking (GML) on texture failed.")
            except Exception as e_crack_gml_mark:
                print(f"ERROR during Step 5c (Crack GML Marking): {e_crack_gml_mark}")
                traceback.print_exc()
        else:
            # ... (skip reasons as before) ...
            print("Skipping Step 5c (Crack GML Marking):")
            if not (final_texture_path and final_texture_path.exists() and cropped_texture_transform and cropped_texture_crs_obj):
                print("  - Base texture or its georeferencing info not available from Step 5.")
            if not cracks_gml_path.exists():
                print(f"  - Input cracks GML file '{cracks_gml_path}' not found.")
    else:
        print("Skipping Step 5c (Crack GML Marking) as per configuration.")

    # --- Step 6: Generate Textured Cut OBJ Model ---
    cut_model_generated_step6 = False
    intermediate_obj_path_step6 = model_assets_output_dir / CUT_OBJ_OUTPUT_FILENAME
    intermediate_mtl_path_step6 = model_assets_output_dir / CUT_MTL_OUTPUT_FILENAME

    inputs_valid_for_cut_step6 = (
        edges_gml_path.exists() and
        roi_gml_path.exists() and
        final_texture_path and final_texture_path.exists()
    )

    if inputs_valid_for_cut_step6:
        print("\n=== STEP 6: Generating Textured Cut OBJ Model (Intermediate) ===")
        print(f"  Base for 3D model: {edges_gml_path.name}, Tool for 3D cut: {roi_gml_path.name}")
        try:
            texture_file_name_for_mtl = final_texture_path.name
            success_step6 = generate_cut_obj_model(
                base_gml_path_str=str(edges_gml_path),
                tool_gml_path_str=str(roi_gml_path),
                output_dir_str=str(model_assets_output_dir),
                target_crs=TARGET_CRS,
                base_extrusion_height=BASE_EXTRUSION_CUT_M,
                tool_extrusion_height=TOOL_EXTRUSION_CUT_M,
                simplify_tolerance=CUT_SIMPLIFY_TOLERANCE_M,
                output_obj_filename=CUT_OBJ_OUTPUT_FILENAME,
                output_mtl_filename=CUT_MTL_OUTPUT_FILENAME,
                texture_filename=texture_file_name_for_mtl,
                material_top=CONVERT_MATERIAL_TOP,
                material_bottom=CONVERT_MATERIAL_BOTTOM,
                material_sides=CONVERT_MATERIAL_SIDES,
                generate_vt=CONVERT_GENERATE_VT,
                z_tolerance=CONVERT_Z_TOLERANCE,
                show_plots=SHOW_PLOTS_ALL_STEPS, save_plots=SAVE_PLOTS_ALL_STEPS, plot_dpi=PLOT_DPI_ALL_STEPS
            )
            if success_step6:
                cut_model_generated_step6 = True
                print(f"Intermediate Textured Cut OBJ generated: {intermediate_obj_path_step6}")
            else:
                print("Intermediate Textured Cut OBJ generation failed.")
        except Exception as e_cut_obj:
            print(f"ERROR Step 6: {e_cut_obj}")
            traceback.print_exc()
    else:
        print("Skipping Step 6 (OBJ Generation): Missing GML inputs or texture from Step 5.")

    # --- Step 6b: Transform OBJ Model ---
    transformed_obj_path_step6b = model_assets_output_dir / TRANSFORMED_OBJ_OUTPUT_FILENAME
    transformed_obj_created = False
    obj_local_frame_origin_world_xy = None
    obj_original_min_z_world = None

    if cut_model_generated_step6 and intermediate_obj_path_step6.exists():
        print("\n=== STEP 6b: Transforming OBJ Model ===")
        try:
            success_step6b, lx_world, ly_world, lz_orig_world = transform_obj_file(
                input_obj_path_str=str(intermediate_obj_path_step6), 
                output_obj_path_str=str(transformed_obj_path_step6b), 
                z_additional_offset_val=TRANSFORM_Z_ADDITIONAL_OFFSET
            )
            if success_step6b and transformed_obj_path_step6b.exists():
                transformed_obj_created = True
                obj_local_frame_origin_world_xy = (lx_world, ly_world)
                obj_original_min_z_world = lz_orig_world
                print(f"OBJ transformation successful. Final Output: {transformed_obj_path_step6b}")
                print(f"  Local Frame Origin (World): X={lx_world:.3f}, Y={ly_world:.3f}, Original_Min_Z={lz_orig_world:.3f}")
            else:
                print(f"ERROR: OBJ Transformation in Step 6b failed or output file not created.")
        except Exception as e_transform_obj:
            print(f"ERROR during Step 6b (OBJ Transformation): {e_transform_obj}")
            traceback.print_exc()
    else:
        print("Skipping Step 6b (OBJ Transformation): Input OBJ from Step 6 not found or Step 6 failed.")

    # --- Step 7: Generate Nav2 Map ---
    nav2_map_generated = False
    nav2_map_output_dir = pipeline_output_dir / NAV2_MAP_OUTPUT_SUBDIR 
    
    # Prepare geometry for Nav2 map's free space (this is now the ROI_GML)
    nav_map_roi_free_space_geom = None
    if roi_gml_path.exists():
        print(f"\nLoading ROI geometry ({roi_gml_path.name}) for Nav2 map free space...")
        nav_map_roi_free_space_geom = load_gml_to_shapely_geom(
            str(roi_gml_path), 
            TARGET_CRS, 
            NAV2_MAP_SIMPLIFY_TOLERANCE_M # Use a specific simplify for map drawing if needed
        )
    else:
        print(f"ERROR: ROI GML file ({roi_gml_path}) not found, cannot define Nav2 free space.")


    inputs_valid_for_nav2 = (
        edges_gml_path.exists() and # Used for map bounds
        nav_map_roi_free_space_geom is not None and # Must have loaded ROI as free space
        not nav_map_roi_free_space_geom.is_empty and
        transformed_obj_created and 
        obj_local_frame_origin_world_xy is not None
    )

    if inputs_valid_for_nav2:
        print("\n=== STEP 7: Generating Nav2 Map Files (Aligned with Transformed OBJ) ===")
        nav2_map_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            map_z_in_local_frame = TRANSFORM_Z_ADDITIONAL_OFFSET 
            success_step7 = generate_nav2_map(
                bounds_gml_input_path_str=str(edges_gml_path), # Use EDGES_GML for overall map bounds
                free_space_shapely_geom_world=nav_map_roi_free_space_geom, # Pass ROI_GML geometry as free space
                obj_local_frame_origin_world_xy=obj_local_frame_origin_world_xy,
                obj_local_frame_base_z_val=map_z_in_local_frame,
                output_dir_str=str(nav2_map_output_dir),
                output_map_basename=NAV2_MAP_OUTPUT_BASENAME,
                map_resolution=NAV2_MAP_RESOLUTION,
                map_padding_m=NAV2_MAP_PADDING_M
            )
            if success_step7:
                nav2_map_generated = True
        except Exception as e_nav_map:
            print(f"ERROR during Step 7 (Nav2 Map Generation): {e_nav_map}")
            traceback.print_exc()
    else:
        print("Skipping Step 7 (Nav2 Map Generation):")
        if not edges_gml_path.exists(): print("  - Edges GML for bounds not found.")
        if nav_map_roi_free_space_geom is None or nav_map_roi_free_space_geom.is_empty:
            print("  - ROI geometry for Nav2 map free space is invalid or empty.")
        if not (transformed_obj_created and obj_local_frame_origin_world_xy is not None):
            print("  - Transformed OBJ or its local frame origin info not available from Step 6b.")

    # --- Step 7b: Generate Nav2 Waypoints ---
    waypoints_yaml_generated = False
    if transformed_obj_created and \
       obj_local_frame_origin_world_xy is not None and \
       obj_original_min_z_world is not None and \
       Path(CSV_FILE).exists():
        print("\n=== STEP 7b: Generating Nav2 Waypoints YAML ===")
        try:
            roll_rad = math.radians(WAYPOINTS_DEFAULT_ORIENTATION_EULER_DEG[0])
            pitch_rad = math.radians(WAYPOINTS_DEFAULT_ORIENTATION_EULER_DEG[1])
            yaw_rad = math.radians(WAYPOINTS_DEFAULT_ORIENTATION_EULER_DEG[2])
            default_orientation_rad = [roll_rad, pitch_rad, yaw_rad]
            waypoint_z_local = TRANSFORM_Z_ADDITIONAL_OFFSET
            waypoints_output_path = nav2_map_output_dir / WAYPOINTS_OUTPUT_YAML_FILENAME

            success_step7b = generate_waypoints_yaml(
                csv_path_str=CSV_FILE,
                source_crs_str=SOURCE_CRS,
                intermediate_crs_str=TARGET_CRS, 
                obj_local_frame_origin_world_xy=obj_local_frame_origin_world_xy,
                waypoint_z_in_local_frame=waypoint_z_local,
                output_yaml_path_str=str(waypoints_output_path),
                default_orientation_euler_rad=default_orientation_rad,
                map_frame_id=WAYPOINTS_MAP_FRAME_ID
            )
            if success_step7b:
                waypoints_yaml_generated = True
                print(f"Waypoints YAML saved to: {waypoints_output_path}")
            else:
                print("Waypoint YAML generation failed.")
        except Exception as e_waypoints:
            print(f"ERROR during Step 7b (Waypoints YAML Generation): {e_waypoints}")
            traceback.print_exc()
    else:
        print("Skipping Step 7b (Waypoints YAML Generation):")
        # ... (skip reasons as before) ...
        if not Path(CSV_FILE).exists(): print(f"  - Input CSV file '{CSV_FILE}' not found.")
        elif not (transformed_obj_created and obj_local_frame_origin_world_xy is not None and obj_original_min_z_world is not None):
            print("  - Required info from Step 6b (OBJ transformation) not available.")

    # --- Step 8: Generate Gazebo World ---
    gazebo_world_generated = False
    gazebo_files_output_dir = pipeline_output_dir / GAZEBO_OUTPUT_SUBDIR
    inputs_valid_for_gazebo = (
        transformed_obj_created and
        transformed_obj_path_step6b.exists() and
        intermediate_mtl_path_step6.exists() and 
        (final_texture_path and final_texture_path.exists()) and 
        intermediate_obj_path_step6.exists() 
    )
    if inputs_valid_for_gazebo:
        print("\n=== STEP 8: Generating Gazebo World Files ===")
        gazebo_files_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Gazebo files output directory: {gazebo_files_output_dir.resolve()}")
        try:
            success_step8 = create_gazebo_model_and_world(
                transformed_obj_file_path=str(transformed_obj_path_step6b), 
                mtl_file_path=str(intermediate_mtl_path_step6), 
                texture_file_path=str(final_texture_path),      
                original_obj_for_origin_calc_path=str(intermediate_obj_path_step6), 
                original_obj_crs_str=TARGET_CRS, 
                output_dir_str=str(gazebo_files_output_dir),
                gazebo_model_name=GAZEBO_MODEL_NAME,
                gazebo_world_filename=GAZEBO_WORLD_FILENAME
            )
            if success_step8:
                gazebo_world_generated = True
                print(f"Gazebo world and model files generated in: {gazebo_files_output_dir}")
            else:
                print("Gazebo world and model generation failed.")
        except Exception as e_gazebo:
            print(f"ERROR during Step 8 (Gazebo World Generation): {e_gazebo}")
            traceback.print_exc()
    else:
        print("Skipping Step 8 (Gazebo World Generation): Missing necessary input files from previous steps.")
        # ... (skip reasons as before) ...
        if not (transformed_obj_created and transformed_obj_path_step6b.exists()): print("  - Transformed OBJ missing.")
        if not intermediate_mtl_path_step6.exists(): print(f"  - MTL file missing: {intermediate_mtl_path_step6}")
        if not (final_texture_path and final_texture_path.exists()): print(f"  - Texture file missing: {final_texture_path}")
        if not intermediate_obj_path_step6.exists(): print(f"  - Original OBJ for origin calc missing: {intermediate_obj_path_step6}")

    end_time_script = time.time()
    print(f"\n--- Orchestrator Script Complete ---")
    print(f"Total execution time: {end_time_script - start_time_script:.2f} seconds")
    print(f"Main pipeline output directory: {pipeline_output_dir.resolve()}")

    # ... (final summary prints as before) ...
    if transformed_obj_created:
        print(f"Final transformed OBJ model assets directory: {model_assets_output_dir.resolve()}")
        print(f"  Transformed OBJ: {transformed_obj_path_step6b.name}")
        print(f"  Associated MTL: {intermediate_mtl_path_step6.name}")
        print(f"  Associated Texture: {final_texture_path.name if final_texture_path else 'N/A'}")
    elif cut_model_generated_step6:
        print(f"Intermediate (untransformed) OBJ model assets directory: {model_assets_output_dir.resolve()}")
        print(f"  Untransformed OBJ: {intermediate_obj_path_step6.name}")
        print(f"  Associated MTL: {intermediate_mtl_path_step6.name}")
        print(f"  Associated Texture: {final_texture_path.name if final_texture_path else 'N/A'}")

    if nav2_map_generated: print(f"Nav2 Map output directory: {nav2_map_output_dir.resolve()}")
    if waypoints_yaml_generated: print(f"Nav2 Waypoints YAML in: {nav2_map_output_dir.resolve()}")
    if gazebo_world_generated: print(f"Gazebo files output directory: {gazebo_files_output_dir.resolve()}")


def create_dummy_gml(file_path, polygon_coords_list, crs_string="urn:ogc:def:crs:EPSG::25832", object_name="DummyObject", feature_namespace="custom", feature_ns_uri="http://example.com/custom"):
    """Creates a simple GML file with a single Polygon feature, allowing namespace customization."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pos_list_str = " ".join([f"{coord[0]} {coord[1]}" for coord in polygon_coords_list])
    gml_content = f"""<gml:FeatureCollection 
    xmlns:gml="http://www.opengis.net/gml/3.2" 
    xmlns:{feature_namespace}="{feature_ns_uri}">
  <gml:featureMember>
    <{feature_namespace}:{object_name}>
      <{feature_namespace}:geometryProperty>
        <gml:Polygon srsName="{crs_string}">
          <gml:exterior><gml:LinearRing>
            <gml:posList>{pos_list_str}</gml:posList>
          </gml:LinearRing></gml:exterior>
        </gml:Polygon>
      </{feature_namespace}:geometryProperty>
    </{feature_namespace}:{object_name}>
  </gml:featureMember>
</gml:FeatureCollection>
"""
    with open(file_path, "w") as f:
        f.write(gml_content)
    print(f"Created dummy GML: {file_path}")


if __name__ == '__main__':
    gml_dir = Path(GML_INPUT_DIR) 
    gml_dir.mkdir(parents=True, exist_ok=True)
    
    # Dummy Edges (original road surface, defines Nav2 map bounds and texture coverage)
    dummy_edges_gml_path = gml_dir / EDGES_GML_FILENAME
    edges_coords = [(371500, 5758500), (372500, 5758500), (372500, 5759500), (371500, 5759500), (371500, 5758500)]
    if not dummy_edges_gml_path.exists():
        create_dummy_gml(dummy_edges_gml_path, edges_coords,
                         object_name="edges", 
                         feature_namespace="ogr",
                         feature_ns_uri="http://ogr.maptools.org/")

    # Dummy ROI (defines excavation area for 3D model AND Nav2 free space)
    dummy_roi_gml_path = gml_dir / ROI_GML_FILENAME
    # ROI should be smaller than edges and within it
    roi_coords = [(371700, 5758700), (372000, 5758700), (372000, 5759000), (371700, 5759000), (371700, 5758700)]
    if not dummy_roi_gml_path.exists():
        create_dummy_gml(dummy_roi_gml_path, roi_coords, 
                         object_name="roi_excavation_and_nav_area", 
                         feature_namespace="ogr", 
                         feature_ns_uri="http://ogr.maptools.org/")

    # Dummy Cracks (for texture marking only, ensure they are within EDGES_GML area)
    dummy_cracks_gml_path = gml_dir / CRACKS_GML_FILENAME
    # Example: A crack within the ROI area to see it on the "cut" part of the texture
    crack_polygon_for_dummy = [
        (371800, 5758800), (371802, 5758800), (371802, 5758850), (371800, 5758850), (371800, 5758800)
    ]
    if not dummy_cracks_gml_path.exists():
        create_dummy_gml(dummy_cracks_gml_path, crack_polygon_for_dummy,
                         object_name="crack_texture_mark", 
                         feature_namespace="ogr",
                         feature_ns_uri="http://ogr.maptools.org/")
    
    # Dummy CSV for defects (still used for waypoints and texture marking)
    csv_file_path = Path(CSV_FILE)
    if not csv_file_path.exists():
        print(f"Creating dummy '{csv_file_path}' for testing, relevant to dummy GML extents.")
        # Defect centers, ensure they are within the ROI_COORDS for realistic navigation waypoints
        defect_centers_epsg25832 = [
            (371750, 5758750), # Inside ROI
            (371950, 5758950)  # Inside ROI
        ]
        defect_data_list = []
        transformer_to_4326 = None
        try:
            transformer_to_4326 = pyproj.Transformer.from_crs(
                pyproj.CRS.from_string(TARGET_CRS), 
                pyproj.CRS.from_string(SOURCE_CRS), 
                always_xy=True 
            )
        except Exception as e:
            print(f"Error creating transformer for dummy CSV: {e}. Lat/Lon will be dummy.")

        for i, (cx, cy) in enumerate(defect_centers_epsg25832):
            defect_id = i + 1; label_name = f"DUMMY_DEFECT_{defect_id}"
            half_size = 0.2 # Smaller defects for clarity if marked on texture
            poly_coords = [
                (cx - half_size, cy - half_size), (cx + half_size, cy - half_size),
                (cx + half_size, cy + half_size), (cx - half_size, cy + half_size),
                (cx - half_size, cy - half_size) 
            ]
            wkt_polygon = f"POLYGON(({', '.join([f'{p[0]:.2f} {p[1]:.2f}' for p in poly_coords])}))"
            optimal_epsg = TARGET_CRS.split(':')[-1] 
            lon, lat = 0.0, 0.0 
            if transformer_to_4326:
                try: lon, lat = transformer_to_4326.transform(cx, cy)
                except Exception as e_trans: print(f"Could not transform point ({cx},{cy}) for dummy CSV: {e_trans}")
            defect_data_list.append({
                'id': defect_id, 'label_name': label_name, 'latitude': lat, 'longitude': lon,
                'geometry_wkt': wkt_polygon, 'optimal_epsg_code': optimal_epsg
            })
        df_defects = pd.DataFrame(defect_data_list)
        df_defects.to_csv(csv_file_path, index=False)
        print(f"Dummy defect CSV created with {len(df_defects)} entries: {csv_file_path}")

    main()