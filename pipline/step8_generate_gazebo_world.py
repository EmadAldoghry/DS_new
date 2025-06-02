# pipline/step8_generate_gazebo_world.py
import os
import shutil
from pathlib import Path
import numpy as np
import pyproj
import traceback # For detailed error printing

def get_obj_vertices(obj_file_path):
    """Reads an OBJ file and returns a list of its geometric vertices."""
    vertices = []
    try:
        with open(obj_file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    try:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except (IndexError, ValueError) as e:
                        print(f"  Warning: Could not parse vertex line: {line.strip()} - {e}")
    except FileNotFoundError:
        print(f"  Error: OBJ file for vertex extraction not found: {obj_file_path}")
        return None
    except Exception as e:
        print(f"  Error reading OBJ file {obj_file_path}: {e}")
        return None
    return vertices

def calculate_and_transform_origin_for_gazebo(
    original_obj_file_path,
    obj_crs_str,
    target_latlon_crs_str="EPSG:4326"
    ):
    """Calculates and transforms Gazebo world origin."""
    print(f"  Calculating Gazebo origin from bounding box of: {original_obj_file_path}")
    print(f"    Assuming original OBJ coordinates are in CRS: {obj_crs_str}")
    print(f"    Target CRS for Gazebo latitude/longitude: {target_latlon_crs_str}")

    vertices = get_obj_vertices(original_obj_file_path)
    if not vertices or len(vertices) < 3:
        print("  Error: Not enough vertices in original OBJ to calculate bounding box.")
        return None, None

    vertices_np = np.array(vertices)
    min_coords_obj_crs = np.min(vertices_np, axis=0)
    max_coords_obj_crs = np.max(vertices_np, axis=0)
    dim_x_obj_crs = max_coords_obj_crs[0] - min_coords_obj_crs[0]
    dim_y_obj_crs = max_coords_obj_crs[1] - min_coords_obj_crs[1]

    print(f"    Original OBJ BBox Min (in {obj_crs_str}): ({min_coords_obj_crs[0]:.2f}, {min_coords_obj_crs[1]:.2f})")
    print(f"    Original OBJ BBox Max (in {obj_crs_str}): ({max_coords_obj_crs[0]:.2f}, {max_coords_obj_crs[1]:.2f})")
    print(f"    Original OBJ BBox Dimensions (X, Y in {obj_crs_str}): ({dim_x_obj_crs:.2f}, {dim_y_obj_crs:.2f})")

    origin_x_obj_crs, origin_y_obj_crs = None, None
    if dim_x_obj_crs <= dim_y_obj_crs:
        origin_x_obj_crs = (min_coords_obj_crs[0] + max_coords_obj_crs[0]) / 2.0
        origin_y_obj_crs = min_coords_obj_crs[1]
        print(f"    X-dim is shorter/equal. Calculated origin in {obj_crs_str}: (X={origin_x_obj_crs:.2f}, Y={origin_y_obj_crs:.2f}).")
    else:
        origin_x_obj_crs = min_coords_obj_crs[0]
        origin_y_obj_crs = (min_coords_obj_crs[1] + max_coords_obj_crs[1]) / 2.0
        print(f"    Y-dim is shorter. Calculated origin in {obj_crs_str}: (X={origin_x_obj_crs:.2f}, Y={origin_y_obj_crs:.2f}).")

    try:
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_user_input(obj_crs_str),
            pyproj.CRS.from_user_input(target_latlon_crs_str),
            always_xy=True
        )
        transformed_lon, transformed_lat = transformer.transform(origin_x_obj_crs, origin_y_obj_crs)
        print(f"    Transformed origin to {target_latlon_crs_str}: (Lon={transformed_lon:.6f}, Lat={transformed_lat:.6f})")
        return transformed_lat, transformed_lon
    except Exception as e:
        print(f"  ERROR: Failed to transform origin coordinates from {obj_crs_str} to {target_latlon_crs_str}: {e}")
        traceback.print_exc()
        return None, None


def create_gazebo_model_and_world(
    transformed_obj_file_path: str, # Path to the final (local coords) OBJ
    mtl_file_path: str,
    texture_file_path: str,
    original_obj_for_origin_calc_path: str,
    original_obj_crs_str: str,
    output_dir_str: str, # Base directory for Gazebo files
    gazebo_model_name="pipeline_model", # Name used for the model in the world and for the subfolder
    gazebo_world_filename="pipeline_world.world"
    ):
    """
    Creates a Gazebo world file with an embedded model definition.
    Model assets (OBJ, MTL, Texture) are copied to a subdirectory named after gazebo_model_name.
    The world origin (latitude, longitude) is derived from the original OBJ's bbox.
    """
    print(f"\n--- Generating Gazebo World with Embedded Model ---")
    output_gazebo_dir = Path(output_dir_str)
    output_gazebo_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Calculate and Transform Gazebo World Origin ---
    latitude_deg, longitude_deg = calculate_and_transform_origin_for_gazebo(
        original_obj_file_path=original_obj_for_origin_calc_path,
        obj_crs_str=original_obj_crs_str,
        target_latlon_crs_str="EPSG:4326"
    )
    if latitude_deg is None or longitude_deg is None:
        print("  ERROR: Could not determine/transform Gazebo world origin. Aborting Gazebo file generation.")
        return False
    print(f"  Using Gazebo world origin (EPSG:4326) - Latitude: {latitude_deg:.7f}, Longitude: {longitude_deg:.7f}")

    # --- 2. Prepare file paths for model assets ---
    # The model assets will be placed in a subdirectory relative to the world file
    # to keep the URI in the world file cleaner.
    model_assets_subdir = Path(gazebo_model_name) # e.g., "pipeline_road_model"
    full_model_assets_path = output_gazebo_dir / model_assets_subdir
    full_model_assets_path.mkdir(parents=True, exist_ok=True)
    print(f"  Gazebo model assets will be stored in: {full_model_assets_path.resolve()}")

    # Get just the filenames
    obj_filename_asset = Path(transformed_obj_file_path).name
    mtl_filename_asset = Path(mtl_file_path).name
    texture_filename_asset = Path(texture_file_path).name

    # Define target paths for copying assets
    target_obj_asset_path = full_model_assets_path / obj_filename_asset
    target_mtl_asset_path = full_model_assets_path / mtl_filename_asset
    target_texture_asset_path = full_model_assets_path / texture_filename_asset

    # Define the URI path to be used in the world file's <mesh><uri>
    # This will be relative to where Gazebo looks for models (GAZEBO_MODEL_PATH)
    # or relative to the world file if Gazebo can resolve it that way.
    # For robustness, if the output_gazebo_dir is added to GAZEBO_MODEL_PATH,
    # then 'pipeline_road_model/model.obj' would work.
    mesh_uri_in_world = f"{model_assets_subdir.name}/{obj_filename_asset}" # e.g., "pipeline_road_model/model.obj"

    # --- 3. Copy model assets (OBJ, MTL, Texture) to the asset subdirectory ---
    try:
        shutil.copyfile(transformed_obj_file_path, target_obj_asset_path)
        shutil.copyfile(mtl_file_path, target_mtl_asset_path)
        shutil.copyfile(texture_file_path, target_texture_asset_path)
        print(f"    Copied OBJ, MTL, Texture to {full_model_assets_path}")
    except Exception as e:
        print(f"  Error copying model asset files: {e}")
        traceback.print_exc()
        return False

    # --- 4. Create Gazebo World File with Embedded Model ---
    world_file_path = output_gazebo_dir / gazebo_world_filename
    # The model name in the <model name='...'> tag can be different from gazebo_model_name (subfolder)
    # but using the same is often less confusing. Let's use a distinct one for clarity if needed,
    # or just gazebo_model_name.
    inline_model_name_in_world = f"{gazebo_model_name}_instance" # Or just gazebo_model_name

    world_content = f"""<sdf version='1.10'>
<world name='default'>
  <plugin name='gz::sim::systems::Physics' filename='gz-sim-physics-system'/>
  <plugin name='gz::sim::systems::UserCommands' filename='gz-sim-user-commands-system'/>
  <plugin name='gz::sim::systems::SceneBroadcaster' filename='gz-sim-scene-broadcaster-system'/>
  <plugin name='gz::sim::systems::Contact' filename='gz-sim-contact-system'/>
  <plugin
    filename="gz-sim-sensors-system"
    name="gz::sim::systems::Sensors">
    <render_engine>ogre2</render_engine>
  </plugin>
  <plugin
    filename="gz-sim-imu-system"
    name="gz::sim::systems::Imu"/>
  <plugin
    filename="gz-sim-navsat-system"
    name="gz::sim::systems::NavSat"/>

  <physics name='1ms' type='ignored'>
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
  </physics>
  <gravity>0 0 -9.81</gravity>
  <magnetic_field>5.5645e-06 2.28758e-05 -4.23884e-05</magnetic_field>
  <atmosphere type='adiabatic'/>
  <scene>
    <ambient>0.4 0.4 0.4 1</ambient>
    <background>0.7 0.7 0.7 1</background>
    <shadows>true</shadows>
  </scene>

  <spherical_coordinates>
    <surface_model>EARTH_WGS84</surface_model>
    <world_frame_orientation>ENU</world_frame_orientation>
    <latitude_deg>{latitude_deg if latitude_deg is not None else 0.0:.7f}</latitude_deg>
    <longitude_deg>{longitude_deg if longitude_deg is not None else 0.0:.7f}</longitude_deg>
    <elevation>0</elevation>
    <heading_deg>0</heading_deg>
  </spherical_coordinates>

  <light name='sun' type='directional'>
    <pose>0 0 10 0 0 0</pose>
    <cast_shadows>true</cast_shadows>
    <intensity>0.8</intensity>
    <direction>-0.5 0.1 -0.9</direction>
    <diffuse>0.8 0.8 0.8 1</diffuse>
    <specular>0.2 0.2 0.2 1</specular>
  </light>

  <model name='ground_plane'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
      </collision>
      <visual name='visual'>
        <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
        <material><ambient>0.7 0.7 0.7 1</ambient><diffuse>0.7 0.7 0.7 1</diffuse><specular>0.7 0.7 0.7 1</specular></material>
      </visual>
    </link>
  </model>

  <model name='{inline_model_name_in_world}'>
    <static>true</static>
    <pose>0 0 0 0 0 0</pose>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          <mesh>
            <uri>{mesh_uri_in_world}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </visual>
      <collision name='collision'>
        <geometry>
          <mesh>
            <uri>{mesh_uri_in_world}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </collision>
      <inertial>
        <mass>1000</mass> <!-- Adjusted mass to be more substantial -->
        <inertia> <!-- Generic box inertia, adjust if more accuracy needed -->
          <ixx>100</ixx> <ixy>0</ixy> <ixz>0</ixz>
          <iyy>100</iyy> <iyz>0</iyz> <izz>100</izz>
        </inertia>
      </inertial>
    </link>
  </model>

</world>
</sdf>
"""
    try:
        with open(world_file_path, "w") as f: f.write(world_content)
        print(f"    Created Gazebo world file with embedded model: {world_file_path.resolve()}")
        print(f"  To use this world, ensure Gazebo can find model assets in '{full_model_assets_path.resolve()}'")
        print(f"  This typically means '{output_gazebo_dir.resolve()}' should be in your GAZEBO_MODEL_PATH,")
        print(f"  or Gazebo is launched from a directory where '{model_assets_subdir.name}/' is a direct subdirectory.")
        return True
    except Exception as e:
        print(f"  Error writing Gazebo world file: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Example for standalone testing
    test_base_output_dir = Path("..") / "output_project" # Adjust if your script is not in pipline/
    # Assume output_project/cut_model_output/ exists from previous steps
    source_model_dir = test_base_output_dir / "cut_model_output"

    test_transformed_obj = source_model_dir / "model.obj" # Final local coords OBJ
    test_mtl_file = source_model_dir / "final_cut_road_model.mtl" # Original MTL name
    test_texture_file = source_model_dir / "road_texture.png" # (Potentially defect-marked) texture
    test_original_obj_for_origin = source_model_dir / "final_cut_road_model.obj" # Untransformed OBJ

    test_original_obj_crs = "EPSG:25832"
    test_gazebo_output_dir = test_base_output_dir / "gazebo_output_embedded" # New output dir for this test
    test_model_name_param = "my_pipeline_road" # This will be the subfolder name for assets

    print(f"--- Testing Gazebo World Generation with Embedded Model ---")
    files_exist = True
    for f_path_str in [test_transformed_obj, test_mtl_file, test_texture_file, test_original_obj_for_origin]:
        f_path = Path(f_path_str)
        if not f_path.exists():
            print(f"ERROR: Test input file not found: {f_path}")
            files_exist = False
            # Create dummy files if they don't exist for the test to proceed somewhat
            if not f_path.parent.exists(): f_path.parent.mkdir(parents=True, exist_ok=True)
            with open(f_path, 'w') as dummy_f: dummy_f.write(f"# Dummy file: {f_path.name}\n")
            if f_path.suffix == '.obj':
                 with open(f_path, 'w') as dummy_f:
                    dummy_f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n") # Minimal OBJ
            print(f"Created dummy file: {f_path}")


    # Create a dummy original OBJ for origin calculation if it truly doesn't exist
    # This is crucial for the calculate_and_transform_origin_for_gazebo function.
    if not Path(test_original_obj_for_origin).exists() or Path(test_original_obj_for_origin).stat().st_size < 10:
        print(f"Re-creating/Ensuring dummy original OBJ for origin test: {test_original_obj_for_origin}")
        Path(test_original_obj_for_origin).parent.mkdir(parents=True, exist_ok=True)
        with open(test_original_obj_for_origin, "w") as f:
            f.write("v 350000.0 5600000.0 0.0\n")
            f.write("v 350010.0 5600000.0 0.0\n")
            f.write("v 350010.0 5600005.0 0.0\n")
            f.write("v 350000.0 5600005.0 0.0\n")
            f.write("f 1 2 3 4\n")

    success = create_gazebo_model_and_world(
        transformed_obj_file_path=str(test_transformed_obj),
        mtl_file_path=str(test_mtl_file),
        texture_file_path=str(test_texture_file),
        original_obj_for_origin_calc_path=str(test_original_obj_for_origin),
        original_obj_crs_str=test_original_obj_crs,
        output_dir_str=str(test_gazebo_output_dir),
        gazebo_model_name=test_model_name_param, # This creates 'my_pipeline_road' subfolder
        gazebo_world_filename=f"{test_model_name_param}_embedded.world"
    )
    if success:
        print(f"Gazebo world (embedded model) test successful. Output in: {test_gazebo_output_dir}")
    else:
        print(f"Gazebo world (embedded model) test failed.")