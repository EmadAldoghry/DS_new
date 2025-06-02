# pipline/step7b_generate_waypoints_yaml.py
import pandas as pd
import yaml
import pyproj
from pathlib import Path
import math # For math.cos, math.sin, math.radians

# Pure Python Euler to Quaternion conversion (common robotics sequence: ZYX intrinsic or XYZ extrinsic)
# Returns [qx, qy, qz, qw]
def euler_to_quaternion(roll, pitch, yaw):
    """Converts Euler angles (roll, pitch, yaw in radians) to quaternion [x, y, z, w]."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]

def generate_waypoints_yaml(
    csv_path_str: str,
    source_crs_str: str,           # CRS of input CSV (e.g., EPSG:4326)
    intermediate_crs_str: str,     # CRS for initial projection (e.g., TARGET_CRS from orchestrator)
    obj_local_frame_origin_world_xy: tuple[float, float], # (x_world, y_world) of local frame origin from Step 6b
    waypoint_z_in_local_frame: float, # The Z value for all waypoints in the local frame (e.g., TRANSFORM_Z_ADDITIONAL_OFFSET)
    output_yaml_path_str: str,
    default_orientation_euler_rad: list[float] = [0.0, 0.0, 0.0], # [roll, pitch, yaw] in RADIANS
    map_frame_id: str = "map"
):
    """
    Generates a YAML file with waypoints for Nav2 from a CSV of defect coordinates.
    The defect coordinates are transformed into the OBJ's local coordinate system.
    The YAML structure mirrors a list of PoseStamped messages.
    """
    print("\n--- Running: Waypoint YAML Generation ---")
    csv_path = Path(csv_path_str)
    output_yaml_path = Path(output_yaml_path_str)
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read CSV
    try:
        df = pd.read_csv(csv_path)
        if not {'longitude', 'latitude'}.issubset(df.columns):
            raise ValueError("CSV must contain 'longitude' and 'latitude' columns.")
    except Exception as e:
        print(f"ERROR: Failed to read or validate CSV {csv_path}: {e}")
        return False

    # 2. Setup CRS transformation
    try:
        transformer_to_intermediate = pyproj.Transformer.from_crs(
            pyproj.CRS.from_user_input(source_crs_str),
            pyproj.CRS.from_user_input(intermediate_crs_str),
            always_xy=True  # Input (lon, lat), output (x, y) based on intermediate_crs_str
        )
    except Exception as e:
        print(f"ERROR setting up CRS transformation from {source_crs_str} to {intermediate_crs_str}: {e}")
        return False

    # 3. Prepare waypoints
    poses_data = [] # This will be a list of PoseStamped-like dicts
    lx_world_origin, ly_world_origin = obj_local_frame_origin_world_xy

    # Convert default orientation from Euler (radians) to Quaternion
    try:
        q = euler_to_quaternion(default_orientation_euler_rad[0],
                                default_orientation_euler_rad[1],
                                default_orientation_euler_rad[2])
        default_orientation_q_dict = {'x': q[0], 'y': q[1], 'z': q[2], 'w': q[3]}
    except Exception as e:
        print(f"Warning: Could not convert Euler to Quaternion: {e}. Using default identity quaternion.")
        default_orientation_q_dict = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}

    print(f"  Transforming {len(df)} defect coordinates to local frame...")
    for index, row in df.iterrows():
        try:
            lon, lat = float(row['longitude']), float(row['latitude'])

            # Project to intermediate CRS (these are "world" coordinates relative to the untransformed OBJ)
            x_intermediate, y_intermediate = transformer_to_intermediate.transform(lon, lat)

            # Transform to local OBJ frame (the frame Nav2 map and Gazebo model use)
            x_local = x_intermediate - lx_world_origin
            y_local = y_intermediate - ly_world_origin
            # All waypoints will be at the specified Z height in the local frame
            z_local = waypoint_z_in_local_frame

            # Create a dictionary matching PoseStamped structure
            pose_stamped_dict = {
                'header': {
                    'frame_id': map_frame_id,
                    'stamp': {'sec': 0, 'nanosec': 0} # Stamp can be zero for waypoints definition
                },
                'pose': {
                    'position': {'x': round(x_local, 4), 'y': round(y_local, 4), 'z': round(z_local, 4)},
                    'orientation': default_orientation_q_dict
                }
            }
            poses_data.append(pose_stamped_dict)
            print(f"    Input (lon,lat): ({lon:.4f}, {lat:.4f}) -> Intermediate (x,y): ({x_intermediate:.2f}, {y_intermediate:.2f}) -> Local (x,y,z): ({x_local:.2f}, {y_local:.2f}, {z_local:.2f})")

        except (ValueError, TypeError) as e:
            print(f"  Skipping row {index} due to invalid coordinates/transformation error: {e}")
            continue

    if not poses_data:
        print("  No valid waypoints generated.")
        return False

    # 4. Write YAML file
    # The top-level key 'poses' matches the field in nav2_msgs/action/FollowWaypoints.Goal
    final_yaml_structure = {'poses': poses_data}

    try:
        with open(output_yaml_path, 'w') as f:
            yaml.dump(final_yaml_structure, f, sort_keys=False, default_flow_style=None, indent=2)
        print(f"  Successfully saved {len(poses_data)} waypoints to: {output_yaml_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to write YAML file {output_yaml_path}: {e}")
        return False

if __name__ == '__main__':
    # Example Usage (for testing this module independently)
    print("--- Testing Waypoint YAML Generation ---")
    
    # Create a dummy CSV for testing
    dummy_csv_content = "latitude,longitude\n51.87101,7.28630\n51.87115,7.28645"
    dummy_csv_path = Path("dummy_defect_coords_for_waypoints_test.csv")
    with open(dummy_csv_path, "w") as f:
        f.write(dummy_csv_content)

    test_output_dir = Path("output_test_waypoints")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = test_output_dir / "test_nav_waypoints.yaml"

    # These dummy values simulate what would come from the orchestrator / Step 6b
    # Example: Original OBJ was in EPSG:25832
    # A point (lat,lon) = (51.87101, 7.28630) might transform to
    # EPSG:25832 (x,y) = (377029.0, 5748034.0) - This is a hypothetical value
    # If obj_local_frame_origin_world_xy (from Step 6b) was (377000.0, 5748000.0)
    # Then local x = 377029.0 - 377000.0 = 29.0
    # And local y = 5748034.0 - 5748000.0 = 34.0
    # And waypoint_z_in_local_frame = 0.1 (example)

    success = generate_waypoints_yaml(
        csv_path_str=str(dummy_csv_path),
        source_crs_str='EPSG:4326',       # CRS of the dummy CSV
        intermediate_crs_str='EPSG:25832',# Target CRS for pipeline
        obj_local_frame_origin_world_xy=(377000.0, 5748000.0), # Example: origin of local frame in world coords
        waypoint_z_in_local_frame=0.1,    # Example: Z height of waypoints in local frame
        output_yaml_path_str=str(output_yaml),
        default_orientation_euler_rad=[0.0, 0.0, math.radians(0)], # Roll, Pitch, Yaw (in radians)
        map_frame_id="robot_map_local"    # Example frame ID
    )

    if success:
        print(f"\nTest waypoints YAML generated: {output_yaml.resolve()}")
        with open(output_yaml, 'r') as f_read:
            print("\nYAML Content:")
            print(f_read.read())
    else:
        print("\nTest waypoints YAML generation FAILED.")

    # Clean up dummy CSV
    dummy_csv_path.unlink(missing_ok=True)
    print("--- Test Finished ---")