# In pipline/step6_generate_cut_obj_model.py

import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon # Ensure these are imported
from shapely.ops import unary_union
import numpy as np # For np.linalg.norm in convert_stl_to_obj
import pyproj
import traceback
import copy # Not strictly used in the final version but good for potential deep copies

# CadQuery and STL/OBJ related libraries
import cadquery as cq
from cadquery import exporters
from stl import mesh # For reading STL in converter
# numpy is already imported above

# Use the shared plotting helper if available
from helpers import plot_geometries


# --- Helper: Create CQ Solid from a single Shapely Polygon ---
def create_cq_solid_from_shapely_poly(shapely_polygon, extrusion_height):
    if not isinstance(shapely_polygon, Polygon) or not shapely_polygon.is_valid or shapely_polygon.is_empty:
        # print("      create_cq_solid_from_shapely_poly: Invalid input polygon.")
        return None
    try:
        exterior_coords = list(shapely_polygon.exterior.coords)
        interior_coords_list = [list(interior.coords) for interior in shapely_polygon.interiors]

        outer_wire_points = exterior_coords
        if tuple(outer_wire_points[0]) != tuple(outer_wire_points[-1]): # Ensure closed
            outer_wire_points.append(outer_wire_points[0])
        if len(outer_wire_points) < 4: # Need at least 3 distinct points + closing point
            # print(f"      create_cq_solid_from_shapely_poly: Not enough points for outer wire ({len(outer_wire_points)}).")
            return None
        
        outer_vectors = [cq.Vector(p[0], p[1], 0) for p in outer_wire_points]
        outer_wire = cq.Wire.makePolygon(outer_vectors)
        if not outer_wire or not outer_wire.isValid():
            # print("      create_cq_solid_from_shapely_poly: Failed to create valid outer wire.")
            return None

        inner_wires = []
        for i, interior_coords in enumerate(interior_coords_list):
            try:
                inner_wire_points = interior_coords
                if tuple(inner_wire_points[0]) != tuple(inner_wire_points[-1]):
                    inner_wire_points.append(inner_wire_points[0])
                if len(inner_wire_points) < 4: continue # Skip invalid inner holes

                inner_vectors = [cq.Vector(p[0], p[1], 0) for p in inner_wire_points]
                inner_wire = cq.Wire.makePolygon(inner_vectors)
                if inner_wire and inner_wire.isValid():
                    inner_wires.append(inner_wire)
            except Exception as hole_err:
                print(f"      Warning: CQ Solid - Hole {i+1} creation failed: {hole_err}")
        
        solid_face = None
        try:
            solid_face = cq.Face.makeFromWires(outer_wire, inner_wires if inner_wires else [])
        except Exception: # If fails with holes, try without
            # print("      Warning: makeFromWires with holes failed, trying without holes.")
            try:
                solid_face = cq.Face.makeFromWires(outer_wire, [])
            except Exception as face_err_no_holes:
                # print(f"      Error: makeFromWires without holes also failed: {face_err_no_holes}")
                return None
        
        if solid_face is None or not isinstance(solid_face, cq.Face) or not solid_face.isValid():
            # print("      Error: Failed to create a valid CQ Face from wires.")
            return None

        # Extrude the face
        try:
            # Create a workplane, add the face, then extrude
            extruded_solid_wp = cq.Workplane("XY").add(solid_face).extrude(extrusion_height)
            
            if extruded_solid_wp.vals() and extruded_solid_wp.solids().vals():
                final_solid_candidate = extruded_solid_wp.val() # This might be a Compound or a Solid

                # Prefer a direct Solid if possible
                if isinstance(final_solid_candidate, cq.Solid) and final_solid_candidate.isValid():
                    return final_solid_candidate
                # If it's a compound with exactly one solid, extract that solid
                if isinstance(final_solid_candidate, cq.Compound) and len(final_solid_candidate.Solids()) == 1:
                    single_solid_from_compound = final_solid_candidate.Solids()[0]
                    if single_solid_from_compound.isValid():
                        return single_solid_from_compound
                # If it's a valid compound (even with multiple solids, though less ideal for a single poly extrusion)
                if isinstance(final_solid_candidate, cq.Compound) and final_solid_candidate.isValid() and len(final_solid_candidate.Solids()) > 0:
                     # print(f"      Warning: Extrusion of polygon resulted in a Compound with {len(final_solid_candidate.Solids())} solids. Returning compound.")
                     return final_solid_candidate # Accept valid compound
                
                # print(f"      Warning: CQ Solid - Extrusion resulted in an unexpected type ({type(final_solid_candidate)}) or invalid/empty solid after extrusion.")
            return None # Extrusion failed or result not usable
        except Exception as extrude_err:
            print(f"      Error: CQ Solid - Extrusion process failed: {extrude_err}")
            return None
    except Exception as e:
        print(f"      Error: General CQ Solid creation from polygon failed: {e}")
        return None

# --- Helper: Create CQ Solid from Shapely Polygon or MultiPolygon ---
# (This function tries to combine parts of a MultiPolygon into a single tool if possible)
def create_cq_solid(shapely_geom, extrusion_height, is_tool_multipolygon=False, tool_plot_prefix="tool"):
    cq_solid_result = None
    geom_type = shapely_geom.geom_type
    # print(f"    Creating CQ solid from {geom_type} (Height: {extrusion_height}m)...")

    if geom_type == 'Polygon':
        cq_solid_result = create_cq_solid_from_shapely_poly(shapely_geom, extrusion_height)
    
    elif geom_type == 'MultiPolygon':
        all_parts_solids = []
        for i, poly_part in enumerate(shapely_geom.geoms):
            # print(f"      Processing part {i+1}/{len(shapely_geom.geoms)} of MultiPolygon for {tool_plot_prefix}...")
            part_solid = create_cq_solid_from_shapely_poly(poly_part, extrusion_height)
            if part_solid and part_solid.isValid():
                all_parts_solids.append(part_solid)
            else:
                pass # print(f"      Warning: Part {i+1} of MultiPolygon ({tool_plot_prefix}) failed to create a valid solid. Skipping this part.")
        
        if not all_parts_solids:
            print(f"    Error: No valid solid parts created from MultiPolygon for {tool_plot_prefix}.")
            return None

        if len(all_parts_solids) == 1:
            cq_solid_result = all_parts_solids[0]
            # print(f"      MultiPolygon for {tool_plot_prefix} resulted in a single valid part solid.")
        else:
            # print(f"      Attempting to combine {len(all_parts_solids)} valid part solids for {tool_plot_prefix} MultiPolygon...")
            # Try to combine/fuse into a single object if it's for a tool
            # If is_tool_multipolygon is True and for cracks, returning a list might be handled by caller.
            # For now, always try to combine for simplicity in the cutting logic.
            
            # Using cq.Workplane().add().combine()
            wp_for_combine = cq.Workplane("XY")
            for s_part in all_parts_solids:
                wp_for_combine = wp_for_combine.add(s_part)
            
            try:
                combined_object = wp_for_combine.combine(glue=True).val() # .val() gets the resulting Shape
                if combined_object and combined_object.isValid():
                    # Check if it's a single solid or a compound of solids
                    if isinstance(combined_object, cq.Solid):
                        cq_solid_result = combined_object
                        # print(f"        Successfully combined MultiPolygon parts for {tool_plot_prefix} into a single Solid.")
                    elif isinstance(combined_object, cq.Compound) and len(combined_object.Solids()) > 0:
                        cq_solid_result = combined_object # Accept a valid compound
                        # print(f"        Combined MultiPolygon parts for {tool_plot_prefix} into a Compound with {len(combined_object.Solids())} solids.")
                    else:
                        # print(f"        Combine operation for {tool_plot_prefix} MultiPolygon resulted in an unexpected type ({type(combined_object)}) or empty compound.")
                        cq_solid_result = None # Indicate failure to get a usable combined tool
                else:
                    # print(f"        Combine operation for {tool_plot_prefix} MultiPolygon failed or resulted in invalid solid.")
                    cq_solid_result = None
            except Exception as e_combine:
                print(f"        Exception during combine for {tool_plot_prefix} MultiPolygon: {e_combine}. Treating as failed combination.")
                traceback.print_exc()
                cq_solid_result = None
            
            if cq_solid_result is None and len(all_parts_solids) > 0:
                print(f"      Warning: Failed to combine MultiPolygon parts for {tool_plot_prefix} into a single usable tool. Using only the first valid part as a fallback.")
                cq_solid_result = all_parts_solids[0] # Fallback to first part if combine fails
    else:
        print(f"    Error: Unsupported geometry type for CQ solid creation: {geom_type}")
        return None

    if cq_solid_result is None or not cq_solid_result.isValid():
        # print(f"    Error: Failed to create a final valid CQ solid for the input geometry ({tool_plot_prefix}).")
        return None
    
    # print(f"      CQ solid creation process completed for {tool_plot_prefix} (type: {type(cq_solid_result).__name__}).")
    return cq_solid_result


# --- load_and_prepare_geometry function (as provided in the previous answer) ---
# (Make sure this is the version that accepts is_crack_geometry, crack_buffer_distance, min_crack_area_m2)
def load_and_prepare_geometry(gml_path_str, target_crs,
                              simplify_tolerance,
                              plot_flag, plot_dir, plot_prefix,
                              is_crack_geometry=False, # This flag and related params remain for generality
                              crack_buffer_distance=0.005,
                              min_crack_area_m2=0.001):
    gml_path = Path(gml_path_str)
    print(f"  Processing GML: {gml_path.name} (Target CRS: {target_crs})")
    all_polygons_for_union = [] 
    raw_polygons_for_plotting = [] 

    target_crs_obj_parsed = None
    try:
        target_crs_obj_parsed = pyproj.CRS.from_user_input(target_crs)
    except Exception as e_crs:
        print(f"  Error parsing target_crs '{target_crs}': {e_crs}. Cannot proceed.")
        return None

    try:
        gdf = gpd.read_file(gml_path)
        if gdf.empty:
            print(f"  Warning: GML file {gml_path.name} is empty or contains no features.")
            return None

        source_crs_gml = gdf.crs
        if source_crs_gml and not source_crs_gml.equals(target_crs_obj_parsed):
            print(f"    Reprojecting GML from {source_crs_gml.srs if source_crs_gml else 'Unknown'} to {target_crs_obj_parsed.srs}...")
            try: gdf = gdf.to_crs(target_crs_obj_parsed)
            except Exception as e_reproject: print(f"    Error reprojecting GML {gml_path.name}: {e_reproject}"); return None
        elif not source_crs_gml:
            print(f"    GML {gml_path.name} has no CRS defined. Assuming it is already in target_crs: {target_crs_obj_parsed.srs}")
            gdf.crs = target_crs_obj_parsed 

        for geom_idx, geom_row_series in gdf.iterrows(): 
            geom = geom_row_series.geometry
            if geom is None or geom.is_empty: continue

            current_geom = geom
            if not current_geom.is_valid: current_geom = current_geom.buffer(0)
            if not current_geom.is_valid or current_geom.is_empty: continue
            
            geoms_to_process_from_current = []
            if current_geom.geom_type == 'Polygon': geoms_to_process_from_current.append(current_geom)
            elif current_geom.geom_type == 'MultiPolygon': geoms_to_process_from_current.extend(list(current_geom.geoms))
            
            for poly_item_idx, poly_item in enumerate(geoms_to_process_from_current):
                if poly_item is None or poly_item.is_empty or poly_item.geom_type != 'Polygon': continue
                final_poly_to_consider = poly_item
                if not final_poly_to_consider.is_valid: final_poly_to_consider = final_poly_to_consider.buffer(0)
                if not final_poly_to_consider.is_valid or final_poly_to_consider.is_empty: continue
                
                raw_polygons_for_plotting.append(final_poly_to_consider) 

                if is_crack_geometry: # This logic will not be hit for cracks from this module anymore
                    if final_poly_to_consider.area < min_crack_area_m2: continue
                    buffered_crack = final_poly_to_consider.buffer(crack_buffer_distance, cap_style=1, join_style=1)
                    if buffered_crack.is_valid and not buffered_crack.is_empty and buffered_crack.area > 1e-9: 
                        all_polygons_for_union.append(buffered_crack)
                    elif final_poly_to_consider.is_valid and not final_poly_to_consider.is_empty and final_poly_to_consider.area >= min_crack_area_m2:
                         all_polygons_for_union.append(final_poly_to_consider)
                else: 
                    all_polygons_for_union.append(final_poly_to_consider)

        if not all_polygons_for_union:
            print(f"  Error: No valid polygons (meeting criteria) found in {gml_path.name} for unioning.")
            return None

        if plot_flag and raw_polygons_for_plotting:
            plot_title_raw = f"Input (Raw Valid Parts): {plot_prefix}"
            if is_crack_geometry: plot_title_raw += f" (Pre-Buffer {crack_buffer_distance:.3f}m)"
            plot_geometries(raw_polygons_for_plotting, target_crs_obj_parsed, plot_title_raw, plot_dir, f"plot_{plot_prefix}_00_input_raw_parts")
        
        num_polys_after_filter = len(all_polygons_for_union)
        log_msg_count = f"    {num_polys_after_filter} valid polygons selected for union from {gml_path.name}"
        if is_crack_geometry: log_msg_count += f" (after filtering & {crack_buffer_distance:.3f}m buffering for cracks)"
        print(log_msg_count)

        if plot_flag and is_crack_geometry and all_polygons_for_union:
            plot_geometries(all_polygons_for_union, target_crs_obj_parsed, f"Input (Post-Filter/Buffer): {plot_prefix}", plot_dir, f"plot_{plot_prefix}_01_input_post_buffer")
        elif plot_flag and not is_crack_geometry and all_polygons_for_union:
            plot_geometries(all_polygons_for_union, target_crs_obj_parsed, f"Input (Parts for Union): {plot_prefix}", plot_dir, f"plot_{plot_prefix}_01_input_parts_for_union")

    except FileNotFoundError: print(f"  Error: GML file not found at {gml_path_str}"); return None
    except Exception as e: print(f"  Error reading/processing GML {gml_path.name}: {e}"); traceback.print_exc(); return None

    merged_geom = None
    try:
        if not all_polygons_for_union: print(f"  Error: No polygons available for union for {plot_prefix}."); return None
        print(f"    Performing unary_union on {len(all_polygons_for_union)} polygons for {plot_prefix}...")
        merged_raw = unary_union(all_polygons_for_union)
        if not merged_raw.is_valid: merged_raw = merged_raw.buffer(0) 
        if merged_raw.is_empty or not merged_raw.is_valid:
            print(f"  Error: Merged geometry for {plot_prefix} is invalid or empty even after buffer(0) on union."); return None
        merged_geom = merged_raw 
        current_simplify_tolerance = simplify_tolerance
        if is_crack_geometry:
            simplify_for_cracks = min(simplify_tolerance, 0.005) 
            if simplify_for_cracks < current_simplify_tolerance: current_simplify_tolerance = simplify_for_cracks
            print(f"    Using simplify tolerance for cracks ({plot_prefix}): {current_simplify_tolerance:.4f}")

        if current_simplify_tolerance > 0:
            simplified = merged_geom.simplify(current_simplify_tolerance, preserve_topology=True)
            simplified_buffered = simplified.buffer(0)
            if simplified_buffered and not simplified_buffered.is_empty and simplified_buffered.is_valid:
                merged_geom = simplified_buffered
                print(f"    Simplification (tolerance: {current_simplify_tolerance:.4f}) and buffer(0) successfully applied to {plot_prefix}.")
            else: print(f"    Warning: Simplification/buffer(0) for {plot_prefix} failed. Using pre-simplification merged geometry.")
        else: print(f"    Skipping simplification for {plot_prefix} as tolerance is <= 0.")
        
        if merged_geom.geom_type == 'GeometryCollection':
            print(f"    Result for {plot_prefix} is GeometryCollection. Extracting and re-unioning Polygons/MultiPolygons...")
            extracted_polygons = [g for g in merged_geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon') and g.is_valid and not g.is_empty]
            if not extracted_polygons: print(f"    Error: GeometryCollection for {plot_prefix} contains no valid Polygons/MultiPolygons after extraction."); return None
            merged_geom = unary_union(extracted_polygons) 
            if not merged_geom.is_valid: merged_geom = merged_geom.buffer(0) 
            if not merged_geom.is_valid or merged_geom.is_empty: print(f"    Error: Failed to create a valid geometry after re-unioning GeometryCollection for {plot_prefix}."); return None
            print(f"    Re-merged {plot_prefix} after GeometryCollection extraction. New type: {merged_geom.geom_type}")

        if merged_geom is None or merged_geom.is_empty or not merged_geom.is_valid: print(f"  Error: Final geometry for {plot_prefix} is invalid or empty before final type check."); return None
        if merged_geom.geom_type not in ['Polygon', 'MultiPolygon']: print(f"  Error: Final geometry for {plot_prefix} has unexpected type: {merged_geom.geom_type}."); return None
        print(f"    Successfully prepared final geometry for {plot_prefix}. Type: {merged_geom.geom_type}, Area: {merged_geom.area:.4f} m^2")
        if plot_flag: plot_geometries(merged_geom, target_crs_obj_parsed, f"Final Prepared Shape: {plot_prefix}", plot_dir, f"plot_{plot_prefix}_02_final_prepared_shape")
        return merged_geom
    except Exception as e: print(f"  Error during merging/simplifying for {plot_prefix}: {e}"); traceback.print_exc(); return None

# --- convert_stl_to_obj function (as previously provided, no changes needed here) ---
def calculate_planar_uv(vertices):
    min_coords = np.min(vertices, axis=0); max_coords = np.max(vertices, axis=0)
    range_coords = max_coords - min_coords
    range_x = range_coords[0] if range_coords[0] > 1e-6 else 1.0
    range_y = range_coords[1] if range_coords[1] > 1e-6 else 1.0
    uvs = np.zeros((vertices.shape[0], 2))
    uvs[:, 0] = (vertices[:, 0] - min_coords[0]) / range_x
    uvs[:, 1] = (vertices[:, 1] - min_coords[1]) / range_y
    return uvs

def convert_stl_to_obj(stl_path, obj_path, mtl_filename,
                       material_top="TexturedTop",
                       material_bottom="UntexturedSides",
                       material_sides="UntexturedSides",
                       generate_vt=True,
                       z_tolerance=0.01):
    if not os.path.exists(stl_path): print(f"  Error: STL not found: '{stl_path}'"); return False
    try: your_mesh = mesh.Mesh.from_file(stl_path)
    except Exception as e: print(f"  Error loading STL '{stl_path}': {e}"); return False
    num_faces = len(your_mesh.vectors)
    if num_faces == 0: print(f"  Warning: STL '{stl_path}' has no faces."); return False
    print(f"  Converting STL: {Path(stl_path).name} ({num_faces} faces) -> OBJ: {Path(obj_path).name}")

    all_points = your_mesh.vectors.reshape(-1, 3)
    unique_vertices, inverse_indices = np.unique(all_points, axis=0, return_inverse=True)
    face_vertex_indices = inverse_indices.reshape(num_faces, 3) + 1

    obj_normals_list = [[0.0, 0.0, 1.0],[0.0, 0.0, -1.0]] # vn 1 (top), vn 2 (bottom)
    side_normal_map = {}; next_side_normal_idx = 3
    top_face_indices = []; bottom_face_indices = []; side_faces_by_normal_idx = {}
    normal_up_threshold = 1.0 - z_tolerance; normal_down_threshold = -1.0 + z_tolerance
    face_normals = your_mesh.normals

    for i in range(num_faces):
        normal = face_normals[i]; nz = normal[2]
        if nz > normal_up_threshold: top_face_indices.append(i) # Uses vn 1
        elif nz < normal_down_threshold:
            bottom_face_indices.append(i); face_vertex_indices[i] = face_vertex_indices[i][::-1] # Uses vn 2, flip verts
        else: # Side face
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 1e-6: norm_unit = normal / norm_mag
            else: norm_unit = np.array([1.0, 0.0, 0.0]) # Default if zero normal
            
            normal_tuple = tuple(round(x, 4) for x in norm_unit) # Round for grouping
            if normal_tuple not in side_normal_map:
                side_normal_map[normal_tuple] = next_side_normal_idx
                obj_normals_list.append(list(norm_unit)); next_side_normal_idx += 1
            norm_idx = side_normal_map[normal_tuple]
            if norm_idx not in side_faces_by_normal_idx: side_faces_by_normal_idx[norm_idx] = []
            side_faces_by_normal_idx[norm_idx].append(i)

    generated_vts = []
    if generate_vt and len(unique_vertices) > 0:
        print("    Generating planar UV coordinates for top face...")
        generated_vts = calculate_planar_uv(unique_vertices)

    print(f"  Writing OBJ file with materials: {obj_path}")
    try:
        with open(obj_path, 'w') as f:
            f.write(f"# Converted from: {os.path.basename(stl_path)}\n")
            f.write(f"# Vertices: {len(unique_vertices)}, Faces: {num_faces}\n")
            f.write(f"mtllib {mtl_filename}\n\n"); f.write(f"o {Path(obj_path).stem}\n\n")
            for v in unique_vertices: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            if generate_vt and len(generated_vts) > 0:
                for vt in generated_vts: f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
                f.write("\n")
            for vn_unit in obj_normals_list: # Already unit vectors or zero
                 f.write(f"vn {vn_unit[0]:.4f} {vn_unit[1]:.4f} {vn_unit[2]:.4f}\n")
            f.write("\n")
            if top_face_indices:
                f.write(f"usemtl {material_top}\n"); f.write("s 1\n")
                for face_idx in top_face_indices:
                    v_indices = face_vertex_indices[face_idx]
                    if generate_vt: f.write(f"f {v_indices[0]}/{v_indices[0]}/1 {v_indices[1]}/{v_indices[1]}/1 {v_indices[2]}/{v_indices[2]}/1\n")
                    else: f.write(f"f {v_indices[0]}//1 {v_indices[1]}//1 {v_indices[2]}//1\n")
                f.write("\n")
            if bottom_face_indices:
                f.write(f"usemtl {material_bottom}\n"); f.write("s 2\n")
                for face_idx in bottom_face_indices: v_indices = face_vertex_indices[face_idx]; f.write(f"f {v_indices[0]}//2 {v_indices[1]}//2 {v_indices[2]}//2\n")
                f.write("\n")
            if side_faces_by_normal_idx:
                f.write(f"usemtl {material_sides}\n"); f.write("s 3\n") # Using smooth group 3 for all sides now
                for normal_idx_val in sorted(side_faces_by_normal_idx.keys()): # Iterate through actual normal indices stored
                    for face_idx in side_faces_by_normal_idx[normal_idx_val]: # normal_idx_val is the actual vn index (e.g., 3, 4, ...)
                        v_indices = face_vertex_indices[face_idx]
                        f.write(f"f {v_indices[0]}//{normal_idx_val} {v_indices[1]}//{normal_idx_val} {v_indices[2]}//{normal_idx_val}\n")
                f.write("\n")
        print(f"  Successfully created OBJ with materials/UVs: {obj_path}")
        return True
    except Exception as e: print(f"Error writing OBJ file: {e}"); traceback.print_exc(); return False

# --- write_mtl_file function (as previously provided, no changes needed here) ---
def write_mtl_file(mtl_path, texture_filename, material_top, material_bottom, material_sides):
    print(f"  Writing MTL file: {mtl_path}")
    try:
        with open(mtl_path, 'w') as f:
            f.write(f"# Material library for {Path(mtl_path).stem}\n\n")
            f.write(f"newmtl {material_top}\n")
            f.write("Ka 1.0 1.0 1.0\n"); f.write("Kd 1.0 1.0 1.0\n")
            f.write("Ks 0.1 0.1 0.1\n"); f.write("Ns 10\n"); f.write("d 1.0\n")
            f.write("illum 2\n"); f.write(f"map_Kd {texture_filename}\n\n")
            if material_bottom != material_sides or material_bottom != material_top : # Ensure bottom is distinct if needed
                f.write(f"newmtl {material_bottom}\n")
                f.write("Ka 0.7 0.7 0.7\n"); f.write("Kd 0.7 0.7 0.7\n")
                f.write("Ks 0.0 0.0 0.0\n"); f.write("Ns 10\n"); f.write("d 1.0\n")
                f.write("illum 1\n\n")
            f.write(f"newmtl {material_sides}\n")
            f.write("Ka 0.8 0.8 0.8\n"); f.write("Kd 0.8 0.8 0.8\n")
            f.write("Ks 0.0 0.0 0.0\n"); f.write("Ns 10\n"); f.write("d 1.0\n")
            f.write("illum 1\n\n")
        print(f"  Successfully wrote MTL: {mtl_path}")
        return True
    except Exception as e: print(f"Error writing MTL file '{mtl_path}': {e}"); traceback.print_exc(); return False


# ================== MAIN FUNCTION FOR THIS STEP ==================
def generate_cut_obj_model(
    base_gml_path_str,
    tool_gml_path_str,
    # REMOVED CRACK RELATED PARAMETERS
    # cracks_gml_path_str,
    # crack_geom_buffer_m_param,
    # min_crack_area_m2_param,
    output_dir_str,
    target_crs,
    base_extrusion_height,
    tool_extrusion_height,
    # cracks_extrusion_height, # REMOVED
    simplify_tolerance,
    output_obj_filename,
    output_mtl_filename,
    texture_filename,
    material_top,
    material_bottom,
    material_sides,
    generate_vt,
    z_tolerance,
    show_plots, save_plots, plot_dpi):
    """
    Generates a textured OBJ model by cutting a tool shape from a base shape.
    Crack cutting has been removed from this function.
    """
    print("\n=== STEP 6: Generating Cut OBJ Model with Texture Info ===")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_obj_path = output_dir / output_obj_filename
    output_mtl_path = output_dir / output_mtl_filename
    intermediate_stl_path = output_dir / f"_{Path(output_obj_filename).stem}_intermediate.stl"

    # --- 1. Load Base Geometry (edges.gml) ---
    print("\n--- Loading Base Geometry (edges.gml) ---")
    base_geom = load_and_prepare_geometry(
        base_gml_path_str, target_crs, simplify_tolerance,
        save_plots, output_dir, "base_shape_cut",
        is_crack_geometry=False # Ensure this is False
    )
    if base_geom is None: print("  Error: Base geometry (edges.gml) failed to load."); return False

    # --- 2. Load Tool Geometry (roi.gml) ---
    print("\n--- Loading Tool Geometry (roi.gml) ---")
    tool_geom = load_and_prepare_geometry(
        tool_gml_path_str, target_crs, simplify_tolerance,
        save_plots, output_dir, "tool_shape_cut",
        is_crack_geometry=False # Ensure this is False
    )
    if tool_geom is None: print("  Error: Tool geometry (roi.gml) failed to load."); return False

    # --- 3. Cracks Geometry loading and processing REMOVED for 3D cutting ---
    # The cracks_gml_path_str, crack_geom_buffer_m_param, etc. are no longer passed.
    # cracks_geom = None
    # cracks_tool_solid_cq = None
    print("  Info: 3D crack cutting is REMOVED from this step. Cracks are only marked on texture in Step 5c.")


    # --- 4. Create CQ Solids ---
    print("\n--- Creating CadQuery Solids ---")
    base_solid_cq = create_cq_solid(base_geom, base_extrusion_height, tool_plot_prefix="Base")
    if not (base_solid_cq and base_solid_cq.isValid()): print("  Error: Failed to create valid base CQ solid."); return False

    tool_solid_cq = create_cq_solid(tool_geom, tool_extrusion_height, tool_plot_prefix="ROI_Tool")
    if not (tool_solid_cq and tool_solid_cq.isValid()): print("  Error: Failed to create valid ROI tool CQ solid."); return False

    # --- 5. Perform Boolean Cuts ---
    print("\n--- Performing Boolean Cuts ---")
    current_solid_for_cutting = base_solid_cq
    
    # Cut with ROI tool
    print("  Attempting cut: Base - ROI_Tool")
    try:
        roi_cut_result = current_solid_for_cutting.cut(tool_solid_cq)
        if roi_cut_result and roi_cut_result.isValid():
            current_solid_for_cutting = roi_cut_result
            print("    ROI cut successful.")
        else:
            print("    ROI cut failed or produced invalid solid. Trying with cleaned inputs...")
            cleaned_base_for_roi = current_solid_for_cutting.clean()
            cleaned_tool_roi = tool_solid_cq.clean()
            if cleaned_base_for_roi.isValid() and cleaned_tool_roi.isValid():
                roi_cut_result_cleaned = cleaned_base_for_roi.cut(cleaned_tool_roi)
                if roi_cut_result_cleaned and roi_cut_result_cleaned.isValid():
                    current_solid_for_cutting = roi_cut_result_cleaned
                    print("    ROI cut successful after cleaning.")
                else:
                    print("    ERROR: ROI cut failed even after cleaning. Proceeding with pre-ROI-cut solid.")
            else:
                print("    ERROR: Could not clean solids for ROI cut. Proceeding with pre-ROI-cut solid.")
    except Exception as e_roi_cut:
        print(f"    ERROR during ROI boolean cut: {e_roi_cut}. Proceeding with pre-ROI-cut solid.")
        traceback.print_exc()

    # Crack cutting logic has been removed.
    # The current_solid_for_cutting after ROI cut is now the final geometry for export.

    final_cut_solid = current_solid_for_cutting
    if not (final_cut_solid and final_cut_solid.isValid()):
        print("  ERROR: Final solid after all cut operations is invalid or None. Cannot export.")
        return False

    # --- 6. Export Intermediate STL ---
    print("\n--- Exporting Intermediate STL ---")
    stl_export_success = False
    try:
        exporters.export(final_cut_solid, str(intermediate_stl_path), opt={"binary": True, "tolerance": 0.01, "angularTolerance": 0.1})
        print(f"  Exported: {intermediate_stl_path.name}")
        stl_export_success = True
    except Exception as e_stl_export:
        print(f"  Error exporting final cut solid to STL: {e_stl_export}")
        traceback.print_exc()

    # --- 7. Convert Cut STL to Final Textured OBJ ---
    obj_conversion_success = False
    if stl_export_success:
        print("\n--- Converting Cut STL to Final Textured OBJ ---")
        obj_conversion_success = convert_stl_to_obj(
            stl_path=str(intermediate_stl_path),
            obj_path=str(output_obj_path),
            mtl_filename=output_mtl_filename,
            material_top=material_top,
            material_bottom=material_bottom,
            material_sides=material_sides,
            generate_vt=generate_vt,
            z_tolerance=z_tolerance
        )
    else: print("  Skipping OBJ conversion due to STL export failure.")

    # --- 8. Write MTL File ---
    mtl_creation_success = False
    if obj_conversion_success:
        print("\n--- Writing MTL File ---")
        mtl_creation_success = write_mtl_file(
            mtl_path=output_mtl_path,
            texture_filename=texture_filename,
            material_top=material_top,
            material_bottom=material_bottom,
            material_sides=material_sides
        )
    else: print("  Skipping MTL file generation because OBJ conversion failed.")

    # --- 9. Cleanup ---
    print("\n--- Cleaning up Intermediate STL File ---")
    if intermediate_stl_path.exists():
        print(f"  NOTE: Intermediate STL kept for debugging: {intermediate_stl_path}")
    else: print(f"    Intermediate file {intermediate_stl_path.name} not found (may indicate earlier STL export error).")

    print(f"\n--- Finished Textured Cut OBJ Generation ({output_obj_path.name}) ---")
    return obj_conversion_success and mtl_creation_success