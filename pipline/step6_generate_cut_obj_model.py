# -*- coding: utf-8 -*-
# step6_generate_cut_obj_model.py

import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np
import pyproj
import traceback
import copy

# CadQuery and STL/OBJ related libraries
import cadquery as cq
from cadquery import exporters
from stl import mesh # For reading STL in converter
import numpy as np

# Use the shared plotting helper if available
from helpers import plot_geometries

# --- Reused Helper: Create CQ Solid ---
# (Keep create_cq_solid_from_shapely_poly as before)
def create_cq_solid_from_shapely_poly(shapely_polygon, extrusion_height):
    # ... (implementation from previous answer) ...
    if not isinstance(shapely_polygon, Polygon) or not shapely_polygon.is_valid or shapely_polygon.is_empty: return None
    try:
        exterior_coords=list(shapely_polygon.exterior.coords); interior_coords_list=[list(interior.coords) for interior in shapely_polygon.interiors]
        outer_wire_points=exterior_coords
        if tuple(outer_wire_points[0])!=tuple(outer_wire_points[-1]): outer_wire_points.append(outer_wire_points[0])
        if len(outer_wire_points)<3: return None
        outer_vectors=[cq.Vector(p[0], p[1], 0) for p in outer_wire_points]; outer_wire=cq.Wire.makePolygon(outer_vectors)
        if not outer_wire: return None
        inner_wires=[]
        for i, interior_coords in enumerate(interior_coords_list):
            try:
                inner_wire_points=interior_coords
                if tuple(inner_wire_points[0])!=tuple(inner_wire_points[-1]): inner_wire_points.append(inner_wire_points[0])
                if len(inner_wire_points)<3: continue
                inner_vectors=[cq.Vector(p[0], p[1], 0) for p in inner_wire_points]; inner_wire=cq.Wire.makePolygon(inner_vectors)
                if inner_wire: inner_wires.append(inner_wire)
            except Exception as hole_err: print(f"Warning: CQ Solid - Hole {i+1} failed: {hole_err}")
        solid_face=None
        try: solid_face=cq.Face.makeFromWires(outer_wire, inner_wires if inner_wires else [])
        except Exception: pass
        if solid_face is None or not isinstance(solid_face, cq.Face):
            try: solid_face=cq.Face.makeFromWires(outer_wire, [])
            except Exception: return None
        if solid_face is None or not isinstance(solid_face, cq.Face): return None
        try:
            extruded_solid_wp=cq.Workplane("XY").add(solid_face).extrude(extrusion_height)
            if extruded_solid_wp.vals() and extruded_solid_wp.solids().vals():
                final_solid=extruded_solid_wp.val()
                if isinstance(final_solid, cq.Shape) and final_solid.isValid(): return final_solid
                if isinstance(final_solid, cq.Compound) and len(final_solid.Solids())==1:
                    candidate=final_solid.Solids()[0];
                    if candidate.isValid(): return candidate
                print(f"Warning: CQ Solid - Extrusion invalid/unexpected type ({type(final_solid)})")
            return None
        except Exception as extrude_err: print(f"Error: CQ Solid - Extrusion failed: {extrude_err}"); return None
    except Exception as e: print(f"Error: CQ Solid creation failed: {e}"); return None


# --- Helper Function to Load, Merge, and Simplify GML ---
# (Keep load_and_prepare_geometry as before)
def load_and_prepare_geometry(gml_path_str, target_crs, simplify_tolerance, plot_flag, plot_dir, plot_prefix):
    """Loads GML, merges, simplifies, and returns a single Shapely geometry."""
    gml_path = Path(gml_path_str)
    if not gml_path.is_file(): print(f"  Error: GML file not found: {gml_path}"); return None
    print(f"  Processing GML: {gml_path.name}"); all_polygons=[]; target_crs_obj=None
    try:
        gdf=gpd.read_file(gml_path)
        if gdf.empty: print("  Warning: GML is empty."); return None
        target_crs_obj_gml=gdf.crs; target_crs_obj_target=pyproj.CRS.from_user_input(target_crs)
        if target_crs_obj_gml and not target_crs_obj_gml.equals(target_crs_obj_target): print(f"    Reprojecting GML from {target_crs_obj_gml.srs} to {target_crs}..."); gdf=gdf.to_crs(target_crs_obj_target)
        elif not target_crs_obj_gml: gdf.crs=target_crs_obj_target
        target_crs_obj=gdf.crs
        for geom in gdf.geometry:
            if geom is None or geom.is_empty: continue
            geom_type=geom.geom_type
            if geom_type=='Polygon': poly_to_add=geom if geom.is_valid else geom.buffer(0);
            if poly_to_add.is_valid and not poly_to_add.is_empty and poly_to_add.geom_type=='Polygon': all_polygons.append(poly_to_add)
            elif geom_type=='MultiPolygon':
                for poly in geom.geoms:
                    if poly is None or poly.is_empty: continue; poly_to_add=poly if poly.is_valid else poly.buffer(0)
                    if poly_to_add.is_valid and not poly_to_add.is_empty and poly_to_add.geom_type=='Polygon': all_polygons.append(poly_to_add)
        if not all_polygons: print(f"  Error: No valid Polygons found in {gml_path.name}."); return None
        print(f"    Found {len(all_polygons)} valid polygons.")
        if plot_flag: plot_geometries(all_polygons, target_crs_obj, f"Input: {plot_prefix}", plot_dir, f"plot_{plot_prefix}_01_input")
    except Exception as e: print(f"  Error reading GML {gml_path.name}: {e}"); return None

    # Merge/Simplify
    merged_geom = None # Initialize before try block
    try:
        merged_raw = unary_union(all_polygons).buffer(0)
        if merged_raw.is_empty or not merged_raw.is_valid:
            print("Error: Merged geometry invalid/empty after unary_union+buffer(0)."); return None

        merged_geom = merged_raw # Start with valid merged result

        if simplify_tolerance > 0:
            simplified = merged_geom.simplify(simplify_tolerance, preserve_topology=True).buffer(0)
            if simplified and not simplified.is_empty and simplified.is_valid:
                merged_geom = simplified
                print("    Simplification applied.")
            else:
                 print("    Warning: Simplification failed or resulted in invalid geometry. Using unsimplified.")
                 # merged_geom remains the valid unsimplified one

        # --- Revised GeometryCollection Handling ---
        if merged_geom.geom_type == 'GeometryCollection':
            print("    Result is GeometryCollection. Extracting Polygons/MultiPolygons...")
            # Define polygons_from_collection *only* within this block's scope
            extracted_polygons = [g for g in merged_geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon') and g.is_valid and not g.is_empty]
            if not extracted_polygons:
                print("    Error: GeometryCollection contains no valid Polygons/MultiPolygons.")
                return None
            # If extraction successful, perform union and overwrite merged_geom
            merged_geom = unary_union(extracted_polygons).buffer(0)
            print(f"    Re-merged after extraction. New type: {merged_geom.geom_type}")
            # No else needed, if it wasn't a GeometryCollection, merged_geom is already set

        # --- Final Check ---
        # Now merged_geom should *always* be defined if we reach here
        if merged_geom is None or merged_geom.is_empty or not merged_geom.is_valid or merged_geom.geom_type not in ['Polygon', 'MultiPolygon']:
             print(f"Error: Final geometry is invalid, empty, or has unexpected type ({merged_geom.geom_type if merged_geom else 'None'}).")
             return None

        print(f"    Final Merged/Simplified type: {merged_geom.geom_type}")
        if plot_flag: plot_geometries(merged_geom, target_crs_obj, f"Merged: {plot_prefix}", plot_dir, f"plot_{plot_prefix}_02_merged")
        return merged_geom # Return the final valid merged_geom

    except Exception as e:
        print(f"  Error during merging/simplifying: {e}")
        traceback.print_exc()
        return None


# --- Helper Function to Create CQ Solid from Shapely Polygon/MultiPolygon ---
# (Keep create_cq_solid as before)
def create_cq_solid(shapely_geom, extrusion_height):
    # ... (implementation from previous answer) ...
    cq_solid = None; geom_type = shapely_geom.geom_type; print(f"    Creating CQ solid from {geom_type} (Height: {extrusion_height}m)...")
    if geom_type == 'Polygon': cq_solid = create_cq_solid_from_shapely_poly(shapely_geom, extrusion_height)
    elif geom_type == 'MultiPolygon':
        all_parts = [create_cq_solid_from_shapely_poly(poly, extrusion_height) for poly in shapely_geom.geoms]; valid_parts = [p for p in all_parts if p and p.isValid()]
        if not valid_parts: raise ValueError("No valid parts from MultiPolygon")
        current_solid = valid_parts[0]
        for next_part in valid_parts[1:]:
            try: current_solid = current_solid.union(next_part)
            except: print("Warning: Union of parts failed. Skipping part."); continue
        cq_solid = current_solid
    else: print(f"Error: Unsupported geometry type for CQ solid creation: {geom_type}")
    if cq_solid is None or not cq_solid.isValid(): print("    Error: Failed to create valid CQ solid."); return None
    print("      CQ solid created successfully."); return cq_solid


# --- NEW/REPLACED: Advanced STL to OBJ Converter + Helpers ---
def calculate_planar_uv(vertices):
    """ Generates simple planar UV coordinates based on XY bounding box. """
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
    """ Converts STL to OBJ with materials, UVs (optional), and normals. """
    if not os.path.exists(stl_path): print(f"  Error: STL not found: '{stl_path}'"); return False
    try: your_mesh = mesh.Mesh.from_file(stl_path)
    except Exception as e: print(f"  Error loading STL '{stl_path}': {e}"); return False
    num_faces = len(your_mesh.vectors)
    if num_faces == 0: print(f"  Warning: STL '{stl_path}' has no faces."); return False # Changed: Fail if empty
    print(f"  Converting STL: {Path(stl_path).name} ({num_faces} faces) -> OBJ: {Path(obj_path).name}")

    all_points = your_mesh.vectors.reshape(-1, 3)
    unique_vertices, inverse_indices = np.unique(all_points, axis=0, return_inverse=True)
    face_vertex_indices = inverse_indices.reshape(num_faces, 3) + 1

    obj_normals_list = [[0.0, 0.0, 1.0],[0.0, 0.0, -1.0]]
    side_normal_map = {}; next_side_normal_idx = 3
    top_face_indices = []; bottom_face_indices = []; side_faces_by_normal_idx = {}
    normal_up_threshold = 1.0 - z_tolerance; normal_down_threshold = -1.0 + z_tolerance
    face_normals = your_mesh.normals

    for i in range(num_faces):
        normal = face_normals[i]; nz = normal[2]
        if nz > normal_up_threshold: top_face_indices.append(i)
        elif nz < normal_down_threshold:
            bottom_face_indices.append(i); face_vertex_indices[i] = face_vertex_indices[i][::-1]
        else:
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 1e-6:
                norm_unit = normal / norm_mag; norm_unit[2] = 0.0; norm_mag_xy = np.linalg.norm(norm_unit)
                if norm_mag_xy > 1e-6: norm_unit /= norm_mag_xy
                else: norm_unit = np.array([1.0, 0.0, 0.0]) # Default side normal
                normal_tuple = tuple(round(x, 5) for x in norm_unit)
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
            f.write(f"mtllib {mtl_filename}\n\n") # Reference MTL file
            f.write(f"o {Path(obj_path).stem}\n\n") # Object name from OBJ filename

            for v in unique_vertices: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            if generate_vt and len(generated_vts) > 0:
                for vt in generated_vts: f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
                f.write("\n")
            for vn_idx, vn in enumerate(obj_normals_list):
                 # Make sure normals are unit vectors before writing
                 vn_vec = np.array(vn)
                 vn_mag = np.linalg.norm(vn_vec)
                 if vn_mag > 1e-6: vn_unit = vn_vec / vn_mag
                 else: vn_unit = vn_vec # Keep zero vector if it was zero
                 # Avoid writing zero vectors as normals if possible, though technically allowed
                 # if np.linalg.norm(vn_unit) > 1e-6:
                 f.write(f"vn {vn_unit[0]:.4f} {vn_unit[1]:.4f} {vn_unit[2]:.4f}\n")
                 # else:
                 #    print(f"Warning: Skipping potentially zero normal vector at index {vn_idx+1}")
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
                f.write(f"usemtl {material_sides}\n"); f.write("s 3\n")
                for normal_idx in sorted(side_faces_by_normal_idx.keys()):
                    for face_idx in side_faces_by_normal_idx[normal_idx]: v_indices = face_vertex_indices[face_idx]; f.write(f"f {v_indices[0]}//{normal_idx} {v_indices[1]}//{normal_idx} {v_indices[2]}//{normal_idx}\n")
                f.write("\n")
        print(f"  Successfully created OBJ with materials/UVs: {obj_path}")
        return True
    except Exception as e: print(f"Error writing OBJ file: {e}"); traceback.print_exc(); return False


# --- NEW: Helper Function to Write MTL File ---
def write_mtl_file(mtl_path, texture_filename, material_top, material_bottom, material_sides):
    """Writes the MTL file defining materials used in the OBJ."""
    print(f"  Writing MTL file: {mtl_path}")
    try:
        with open(mtl_path, 'w') as f:
            f.write(f"# Material library for {Path(mtl_path).stem}\n\n")

            # --- Top Material (Textured) ---
            f.write(f"newmtl {material_top}\n")
            f.write("Ka 1.0 1.0 1.0\n")  # Ambient color
            f.write("Kd 1.0 1.0 1.0\n")  # Diffuse color (texture overrides)
            f.write("Ks 0.1 0.1 0.1\n")  # Specular color
            f.write("Ns 10\n")           # Specular exponent
            f.write("d 1.0\n")           # Opacity (1=opaque)
            f.write("illum 2\n")         # Illumination model (diffuse & specular)
            f.write(f"map_Kd {texture_filename}\n\n") # Texture map for diffuse color

            # --- Bottom Material (Untextured) ---
            # Only write if different from Sides material
            if material_bottom != material_sides:
                f.write(f"newmtl {material_bottom}\n")
                f.write("Ka 0.7 0.7 0.7\n") # Medium gray
                f.write("Kd 0.7 0.7 0.7\n")
                f.write("Ks 0.0 0.0 0.0\n") # No specular
                f.write("Ns 10\n")
                f.write("d 1.0\n")
                f.write("illum 1\n\n")     # Diffuse only

            # --- Sides Material (Untextured) ---
            f.write(f"newmtl {material_sides}\n")
            f.write("Ka 0.8 0.8 0.8\n") # Light gray
            f.write("Kd 0.8 0.8 0.8\n")
            f.write("Ks 0.0 0.0 0.0\n") # No specular
            f.write("Ns 10\n")
            f.write("d 1.0\n")
            f.write("illum 1\n\n")     # Diffuse only
        print(f"  Successfully wrote MTL: {mtl_path}")
        return True
    except Exception as e:
        print(f"Error writing MTL file '{mtl_path}': {e}")
        traceback.print_exc()
        return False


# ================== MAIN FUNCTION FOR THIS STEP ==================

def generate_cut_obj_model(
    base_gml_path_str,
    tool_gml_path_str,
    output_dir_str,
    target_crs,
    base_extrusion_height,
    tool_extrusion_height,
    simplify_tolerance,
    output_obj_filename,
    output_mtl_filename,   # <<< NEW: MTL filename parameter
    texture_filename,      # <<< NEW: Texture filename (relative path for MTL)
    material_top,          # <<< NEW
    material_bottom,       # <<< NEW
    material_sides,        # <<< NEW
    generate_vt,           # <<< NEW
    z_tolerance,           # <<< NEW
    show_plots, save_plots, plot_dpi):
    """
    Generates a textured OBJ model by cutting one extruded shape from another.
    """
    print("\n=== STEP 6: Generating Cut OBJ Model with Texture Info ===")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_obj_path = output_dir / output_obj_filename
    output_mtl_path = output_dir / output_mtl_filename # Full path for MTL writer
    intermediate_stl_path = output_dir / f"_{Path(output_obj_filename).stem}_intermediate.stl"

    # Use local plotting flags if helpers.py version wasn't imported
    global SHOW_PLOTS_LOCAL, SAVE_PLOTS_LOCAL, PLOT_DPI_LOCAL
    SHOW_PLOTS_LOCAL = show_plots
    SAVE_PLOTS_LOCAL = save_plots
    PLOT_DPI_LOCAL = plot_dpi

    # --- 1. Load Base Geometry ---
    print("\n--- Loading Base Geometry ---")
    base_geom = load_and_prepare_geometry(base_gml_path_str, target_crs, simplify_tolerance, save_plots, output_dir, "base_shape_cut")
    if base_geom is None: return False

    # --- 2. Load Tool Geometry ---
    print("\n--- Loading Tool Geometry ---")
    tool_geom = load_and_prepare_geometry(tool_gml_path_str, target_crs, simplify_tolerance, save_plots, output_dir, "tool_shape_cut")
    if tool_geom is None: return False

    # --- 3. Create CQ Solids ---
    print("\n--- Creating CadQuery Solids ---")
    base_solid_cq = create_cq_solid(base_geom, base_extrusion_height)
    if base_solid_cq is None: return False
    tool_solid_cq = create_cq_solid(tool_geom, tool_extrusion_height)
    if tool_solid_cq is None: return False

    # --- 4. Perform Boolean Cut ---
    print("\n--- Performing Boolean Cut (Base - Tool) ---")
    cut_result_solid = None
    try:
        cut_result_solid = base_solid_cq.cut(tool_solid_cq)
        if not cut_result_solid or not cut_result_solid.isValid():
            print("  Warning: Cut invalid. Trying cleaned inputs...");
            cleaned_base = base_solid_cq.clean(); cleaned_tool = tool_solid_cq.clean()
            if cleaned_base.isValid() and cleaned_tool.isValid(): cut_result_solid = cleaned_base.cut(cleaned_tool)
            if not cut_result_solid or not cut_result_solid.isValid(): print("  Error: Cut failed even after cleaning."); return False
            else: print("  Cut successful after cleaning.")
        else: print("  Boolean cut successful.")
    except Exception as e_cut: print(f"  ERROR during boolean cut: {e_cut}"); traceback.print_exc(); return False

    # --- 5. Export Intermediate STL ---
    print("\n--- Exporting Intermediate STL ---")
    stl_export_success = False
    if cut_result_solid and cut_result_solid.isValid():
        try:
            exporters.export(cut_result_solid, str(intermediate_stl_path), opt={"binary": True})
            print(f"  Exported: {intermediate_stl_path.name}")
            stl_export_success = True
        except Exception as e: print(f"  Error exporting STL: {e}"); traceback.print_exc()
    else: print("  Error: No valid cut result solid.")

    # --- 6. Convert Cut STL to Final Textured OBJ ---
    obj_conversion_success = False
    if stl_export_success:
        print("\n--- Converting Cut STL to Final Textured OBJ ---")
        obj_conversion_success = convert_stl_to_obj( # <<< CALL ADVANCED CONVERTER
            stl_path=str(intermediate_stl_path),
            obj_path=str(output_obj_path),
            mtl_filename=output_mtl_filename,   # Pass MTL filename for mtllib line
            material_top=material_top,
            material_bottom=material_bottom,
            material_sides=material_sides,
            generate_vt=generate_vt,             # Pass UV flag
            z_tolerance=z_tolerance              # Pass Z tolerance
        )
    else: print("  Skipping OBJ conversion.")

    # --- 7. Write MTL File ---
    mtl_creation_success = False
    if obj_conversion_success: # Only write MTL if OBJ was created
        print("\n--- Writing MTL File ---")
        mtl_creation_success = write_mtl_file(
            mtl_path=output_mtl_path,
            texture_filename=texture_filename, # Use the relative texture filename
            material_top=material_top,
            material_bottom=material_bottom,
            material_sides=material_sides
        )
    else:
        print("  Skipping MTL file generation because OBJ conversion failed.")


    # --- 8. Cleanup ---
    print("\n--- Cleaning up Intermediate STL File ---")
    if intermediate_stl_path.exists():
        try: intermediate_stl_path.unlink(); print(f"  Deleted: {intermediate_stl_path.name}")
        except OSError as e: print(f"  Warning: Could not delete {intermediate_stl_path.name}: {e}")
    else: print(f"    Intermediate file not found: {intermediate_stl_path.name}")

    print(f"\n--- Finished Textured Cut OBJ Generation ({output_obj_path.name}) ---")
    # Return True only if both OBJ and MTL were successfully created
    return obj_conversion_success and mtl_creation_success