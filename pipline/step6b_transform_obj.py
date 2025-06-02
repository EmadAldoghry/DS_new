# pipline/step6b_transform_obj.py
import os
import sys
import io # For string-based OBJ processing
from pathlib import Path

# --- Configuration ---
# Z_ADDITIONAL_OFFSET is passed as a function argument
# FLOAT_PRECISION is implicitly handled by f-string formatting to .6f

def _transform_obj_content(obj_content: str, z_additional_offset_val: float) -> tuple[str | None, float | None, float | None, float | None]:
    """
    Transforms vertex coordinates in an OBJ string and returns transformation parameters.
    ... (rest of docstring as before) ...

    Returns:
        tuple: (transformed_obj_str, offset_x_world, offset_y_world, original_min_z_world)
               Returns (None, None, None, None) if an error occurs.
    """

    vertices_data = []
    other_lines_after_vertices = []
    original_header_non_comments = []
    vertex_lines_processed = False

    min_x_orig, max_x_orig = float('inf'), float('-inf')
    min_y_orig, max_y_orig = float('inf'), float('-inf')
    min_z_orig = float('inf')

    for line in obj_content.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            if vertex_lines_processed:
                other_lines_after_vertices.append(line)
            else:
                original_header_non_comments.append(line)
            continue

        if stripped_line.startswith('v '):
            vertex_lines_processed = True
            try:
                parts = stripped_line.split()
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                vertices_data.append({'x': x, 'y': y, 'z': z, 'original_line': line})
                min_x_orig = min(min_x_orig, x)
                max_x_orig = max(max_x_orig, x)
                min_y_orig = min(min_y_orig, y)
                max_y_orig = max(max_y_orig, y)
                min_z_orig = min(min_z_orig, z)
            except (IndexError, ValueError) as e:
                print(f"  Warning: Could not parse vertex line, skipping transformation for it: '{line}' - Error: {e}")
                other_lines_after_vertices.append(line)
        elif not vertex_lines_processed:
             if not stripped_line.startswith('#'):
                 original_header_non_comments.append(line)
        else:
            other_lines_after_vertices.append(line)

    if not vertices_data:
        print("  Error: No geometric vertices ('v' lines) found. Cannot perform transformations.")
        return None, None, None, None

    if min_x_orig == float('inf') or min_y_orig == float('inf') or \
       max_x_orig == float('-inf') or max_y_orig == float('-inf') or \
       min_z_orig == float('inf'):
         print("  Error: Could not determine bounding box or min_z. Cannot transform.")
         return None, None, None, None

    print(f"  Original Bounding Box (World Coords): X=[{min_x_orig:.6f}, {max_x_orig:.6f}], Y=[{min_y_orig:.6f}, {max_y_orig:.6f}]")
    print(f"  Original Min Z (World Coords): {min_z_orig:.6f}")

    bb_width = max_x_orig - min_x_orig
    bb_height = max_y_orig - min_y_orig

    # These are the world coordinates of the origin of the new local frame
    local_frame_origin_x_world = 0.0
    local_frame_origin_y_world = 0.0

    if bb_width <= bb_height:
        local_frame_origin_x_world = (min_x_orig + max_x_orig) / 2.0
        local_frame_origin_y_world = min_y_orig
        print(f"  Info: Bbox X-dim ({bb_width:.6f}) <= Y-dim ({bb_height:.6f}). Local Frame Origin (World): X at X-mid, Y at Y-min.")
    else:
        local_frame_origin_x_world = min_x_orig
        local_frame_origin_y_world = (min_y_orig + max_y_orig) / 2.0
        print(f"  Info: Bbox Y-dim ({bb_height:.6f}) < X-dim ({bb_width:.6f}). Local Frame Origin (World): X at X-min, Y at Y-mid.")

    z_shift_to_make_min_z_zero = -min_z_orig # This shift makes the object's min_z touch 0 in world coords
    
    print(f"  Calculated Local Frame Origin (World Coords): X_offset={local_frame_origin_x_world:.6f}, Y_offset={local_frame_origin_y_world:.6f}")
    print(f"  Object vertices will be shifted by Z_shift_to_make_min_z_zero={z_shift_to_make_min_z_zero:.6f} and then by z_additional_offset_val={z_additional_offset_val:.6f}.")
    
    final_obj_base_z_local = 0.0 + z_additional_offset_val # The Z value of the object's base in the new local frame


    output_buffer = io.StringIO()
    output_buffer.write("# 3D Model: model.obj (Transformed to Local Coordinates)\n")
    output_buffer.write(f"# Local Frame Origin (World Coords): X={local_frame_origin_x_world:.6f}, Y={local_frame_origin_y_world:.6f}, Z_approx={min_z_orig:.6f}\n")
    output_buffer.write(f"# Object Base Z in Local Frame: {final_obj_base_z_local:.6f}\n")


    for line in original_header_non_comments:
         output_buffer.write(line + "\n")
    if vertices_data and original_header_non_comments and original_header_non_comments[-1].strip() != "":
        output_buffer.write("\n")

    output_buffer.write("# Vertices (X Y Z) in Local Coordinates\n")
    actual_new_min_z_local = float('inf')
    for v_data in vertices_data:
        new_x_local = v_data['x'] - local_frame_origin_x_world
        new_y_local = v_data['y'] - local_frame_origin_y_world
        # Z processing: 1. Shift to bring min_z_orig to 0. 2. Add the final desired offset.
        new_z_local = (v_data['z'] - min_z_orig) + z_additional_offset_val
        actual_new_min_z_local = min(actual_new_min_z_local, new_z_local)
        output_buffer.write(f"v {new_x_local:.6f} {new_y_local:.6f} {new_z_local:.6f}\n")
    
    print(f"  New Min Z in Local Frame after transformation: {actual_new_min_z_local:.6f} (Targeted: {final_obj_base_z_local:.6f})")

    if vertices_data and other_lines_after_vertices:
        if not output_buffer.getvalue().endswith("\n\n") and not output_buffer.getvalue().endswith("\r\n\r\n"):
             if not output_buffer.getvalue().endswith("\n"): output_buffer.write("\n")
             if not other_lines_after_vertices[0].strip() == "":
                output_buffer.write("\n")

    for line in other_lines_after_vertices:
        output_buffer.write(line + "\n")

    return output_buffer.getvalue(), local_frame_origin_x_world, local_frame_origin_y_world, min_z_orig


def transform_obj_file(input_obj_path_str: str, output_obj_path_str: str, z_additional_offset_val: float) -> tuple[bool, float | None, float | None, float | None]:
    """
    Reads an OBJ file, transforms its vertex coordinates, and writes a new OBJ file.
    Uses the _transform_obj_content function for the core logic.

    Args:
        input_obj_path_str: Path to the input OBJ file.
        output_obj_path_str: Path to save the transformed OBJ file.
        z_additional_offset_val: Additional Z offset to apply after grounding the object's original min_z to 0.

    Returns:
        tuple: (success_flag, local_frame_origin_x_world, local_frame_origin_y_world, original_min_z_world)
               On failure, (False, None, None, None).
    """
    print(f"Reading OBJ file from: {input_obj_path_str}")
    try:
        with open(input_obj_path_str, 'r', encoding='utf-8') as infile:
            original_obj_data = infile.read()
    except FileNotFoundError:
        print(f"  Error: Input file '{input_obj_path_str}' not found.")
        return False, None, None, None
    # ... (rest of error handling for read as before)

    print("\n  --- Performing OBJ Transformation ---")
    transformed_obj_data, offset_x, offset_y, orig_min_z = _transform_obj_content(original_obj_data, z_additional_offset_val)

    if transformed_obj_data is None:
        print("  OBJ content transformation failed. No output file generated.")
        return False, None, None, None

    # ... (writing logic as before) ...
    print(f"\n  --- Writing transformed OBJ to: {output_obj_path_str} ---")
    try:
        output_dir = os.path.dirname(output_obj_path_str)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"  Created output directory: {output_dir}")

        with open(output_obj_path_str, 'w', encoding='utf-8') as outfile:
            outfile.write(transformed_obj_data)
        print("  Transformation writing complete.")
        return True, offset_x, offset_y, orig_min_z # Return success and offsets
    except Exception as e:
        print(f"  An error occurred while writing transformed file '{output_obj_path_str}': {e}")
        return False, None, None, None


if __name__ == "__main__":
    # Example for standalone testing (adjust paths as needed)
    script_dir = Path(__file__).parent
    base_output_dir = script_dir.parent / "output_project" / "cut_model_output"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    dummy_input_obj_path = base_output_dir / "final_cut_road_model.obj"
    dummy_output_obj_path = base_output_dir / "model.obj" 
    default_z_offset = 0.05

    # ... (dummy OBJ creation as before) ...
    if not dummy_input_obj_path.exists():
        print(f"Creating dummy OBJ file for testing: {dummy_input_obj_path}")
        with open(dummy_input_obj_path, "w") as f:
            f.write("# Dummy OBJ file for testing step6b_transform_obj.py\n")
            f.write("mtllib final_cut_road_model.mtl\n")
            f.write("o TestObject\n")
            # Original world coordinates
            f.write("v 1000.0 2000.0 50.0\n") # min_x=1000, min_y=2000, min_z=50
            f.write("v 1010.0 2000.0 50.0\n") # max_x=1010 (width=10)
            f.write("v 1010.0 2020.0 50.0\n") # max_y=2020 (height=20)
            f.write("v 1000.0 2020.0 50.0\n")
            f.write("v 1000.0 2000.0 51.0\n") # max_z=51
            f.write("v 1010.0 2000.0 51.0\n")
            f.write("v 1010.0 2020.0 51.0\n")
            f.write("v 1000.0 2020.0 51.0\n")
            f.write("vt 0.0 0.0\n") # ... rest of dummy content
            f.write("f 1// 2// 3// 4//\n") 
            # Expected local frame origin (world): X-dim (10) <= Y-dim (20)
            # local_frame_origin_x_world = (1000+1010)/2 = 1005
            # local_frame_origin_y_world = 2000
            # original_min_z_world = 50
            # Expected output vertices (example for v 1000.0 2000.0 50.0):
            # vx = 1000 - 1005 = -5
            # vy = 2000 - 2000 = 0
            # vz = (50 - 50) + 0.05 = 0.05
    
    print(f"Attempting to transform: {dummy_input_obj_path} -> {dummy_output_obj_path}")
    print(f"Using Z additional offset: {default_z_offset}")

    success, ox, oy, oz = transform_obj_file(
        str(dummy_input_obj_path), 
        str(dummy_output_obj_path),
        z_additional_offset_val=default_z_offset
    )
    if success:
        print(f"Standalone test transformation successful: {dummy_output_obj_path}")
        print(f"  Returned Local Frame Origin (World Coords): X={ox:.2f}, Y={oy:.2f}")
        print(f"  Returned Original Min Z (World Coords): Z={oz:.2f}")
    else:
        print(f"Standalone test transformation FAILED.")
        sys.exit(1)