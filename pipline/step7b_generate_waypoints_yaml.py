# pipline/step7b_generate_waypoints_yaml.py
import pandas as pd
import yaml
import pyproj
from pathlib import Path
import math # For math.cos, math.sin, math.radians
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
import traceback
from typing import Union # For type hinting

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

def _get_polygon_centerline(polygon: Polygon, min_centerline_segment_length: float) -> Union[LineString, None]:
    """
    Approximates the centerline of an elongated polygon using its minimum rotated rectangle.
    Args:
        polygon: The input Shapely Polygon.
        min_centerline_segment_length: Minimum length for a valid resulting centerline segment.
    Returns:
        A LineString representing the centerline, or None if not found or too short.
    """
    if not polygon.is_valid or polygon.is_empty:
        return None
    
    try:
        mrr = polygon.minimum_rotated_rectangle
        if not mrr or not mrr.is_valid or mrr.is_empty or mrr.geom_type != 'Polygon':
            # print(f"  Debug: MRR for polygon is invalid or not a Polygon (type: {mrr.geom_type if mrr else 'None'}). Area: {polygon.area if polygon else 'N/A'}")
            return None

        mrr_coords = list(mrr.exterior.coords) 
        if len(mrr_coords) < 5: # Should be 5 points for a closed rectangle
            # print(f"  Debug: MRR exterior has too few coords: {len(mrr_coords)}")
            return None

        p0, p1, p2, p3 = mrr_coords[0], mrr_coords[1], mrr_coords[2], mrr_coords[3]
        
        side1_len = Point(p0).distance(Point(p1))
        side2_len = Point(p1).distance(Point(p2))

        short_side_mid1, short_side_mid2 = None, None
        if side1_len < side2_len:
            short_side_mid1 = Point((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            short_side_mid2 = Point((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
        else:
            short_side_mid1 = Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            short_side_mid2 = Point((p3[0] + p0[0]) / 2, (p3[1] + p0[1]) / 2)
            
        if short_side_mid1 is None or short_side_mid2 is None: return None # Should not happen if MRR is valid
        
        # Check if midpoints are too close (can happen for very thin slivers)
        if short_side_mid1.distance(short_side_mid2) < 1e-6:
             # print(f"  Debug: MRR short side midpoints are too close. Polygon area: {polygon.area}")
             return None

        centerline_candidate = LineString([short_side_mid1, short_side_mid2])
        
        # Intersect this candidate with the original polygon
        actual_centerline_geom = centerline_candidate.intersection(polygon)
        
        if actual_centerline_geom.is_empty: return None
        
        final_line_segment = None
        if actual_centerline_geom.geom_type == 'LineString':
            if actual_centerline_geom.length >= min_centerline_segment_length:
                final_line_segment = actual_centerline_geom
        elif actual_centerline_geom.geom_type == 'MultiLineString':
            longest_line = max(actual_centerline_geom.geoms, key=lambda line: line.length, default=None)
            if longest_line and longest_line.length >= min_centerline_segment_length:
                final_line_segment = longest_line
        
        return final_line_segment
    except Exception as e:
        # print(f"  Warning: Error in _get_polygon_centerline for polygon (area {polygon.area:.3f}): {e}")
        # traceback.print_exc()
        return None


def generate_waypoints_yaml(
    cracks_gml_path_str: str,
    intermediate_crs_str: str,     # CRS for processing GML geometries (e.g., TARGET_CRS from orchestrator)
    cracks_gml_fallback_crs_str: str, # Fallback CRS if GML has no CRS info
    obj_local_frame_origin_world_xy: tuple[float, float], # (x_world, y_world) of local frame origin from Step 6b
    waypoint_z_in_local_frame: float, # The Z value for all waypoints in the local frame
    output_yaml_path_str: str,
    waypoint_interval_m: float,      # Interval for sampling points along centerlines
    default_orientation_euler_rad: list[float] = [0.0, 0.0, 0.0], # [roll, pitch, yaw] in RADIANS
    map_frame_id: str = "map"
):
    """
    Generates a YAML file with waypoints for Nav2 from centerlines of crack polygons in a GML file.
    """
    print("\n--- Running: Waypoint YAML Generation (from Cracks GML Centerlines) ---")
    output_yaml_path = Path(output_yaml_path_str)
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    min_len_for_centerline_processing = 0.05 * waypoint_interval_m # Heuristic: small fraction of interval

    # 1. Load Cracks GML
    try:
        gdf_cracks = gpd.read_file(cracks_gml_path_str)
        if gdf_cracks.empty:
            print(f"  Info: Cracks GML file {cracks_gml_path_str} is empty. No waypoints will be generated.")
            # Write an empty poses list to YAML to signify no waypoints
            with open(output_yaml_path, 'w') as f:
                yaml.dump({'poses': []}, f, sort_keys=False, default_flow_style=None, indent=2)
            print(f"  Empty waypoints YAML saved to: {output_yaml_path}")
            return True 
    except Exception as e:
        print(f"ERROR: Failed to read Cracks GML {cracks_gml_path_str}: {e}")
        traceback.print_exc()
        return False

    # 2. Ensure crack geometries are in the intermediate_crs_str
    target_intermediate_crs_obj = pyproj.CRS.from_user_input(intermediate_crs_str)
    try:
        if gdf_cracks.crs:
            if not gdf_cracks.crs.equals(target_intermediate_crs_obj):
                print(f"    Reprojecting cracks from {gdf_cracks.crs.srs} to {target_intermediate_crs_obj.srs}...")
                gdf_cracks = gdf_cracks.to_crs(target_intermediate_crs_obj)
        else:
            print(f"    Warning: Cracks GML {Path(cracks_gml_path_str).name} has no CRS. Assuming fallback: {cracks_gml_fallback_crs_str}")
            fallback_crs_obj = pyproj.CRS.from_user_input(cracks_gml_fallback_crs_str)
            gdf_cracks.crs = fallback_crs_obj
            if not fallback_crs_obj.equals(target_intermediate_crs_obj):
                print(f"    Reprojecting cracks from fallback {fallback_crs_obj.srs} to {target_intermediate_crs_obj.srs}...")
                gdf_cracks = gdf_cracks.to_crs(target_intermediate_crs_obj)
    except Exception as e_proj:
        print(f"  Error reprojecting crack geometries to {intermediate_crs_str}: {e_proj}")
        traceback.print_exc()
        return False

    # 3. Prepare waypoints from centerlines
    all_waypoint_candidates_world_crs = [] # List of Shapely Points in intermediate_crs
    lx_world_origin, ly_world_origin = obj_local_frame_origin_world_xy

    print(f"  Processing {len(gdf_cracks)} features from GML for centerlines...")
    processed_polygons = 0
    for index, row_series in gdf_cracks.iterrows():
        geom = row_series.geometry
        if geom is None or geom.is_empty or not geom.is_valid:
            continue

        polygons_in_feature = []
        if geom.geom_type == 'Polygon':
            polygons_in_feature.append(geom)
        elif geom.geom_type == 'MultiPolygon':
            polygons_in_feature.extend(list(g for g in geom.geoms if g.is_valid and not g.is_empty and g.geom_type == 'Polygon'))
        
        for poly_idx, crack_polygon in enumerate(polygons_in_feature):
            processed_polygons += 1
            centerline = _get_polygon_centerline(crack_polygon, min_len_for_centerline_processing)
            if centerline and centerline.length >= min_len_for_centerline_processing :
                sampled_points_on_line = []
                # Add start point
                sampled_points_on_line.append(centerline.interpolate(0))
                
                # Add intermediate points if line is long enough for interval
                if centerline.length > waypoint_interval_m: # Ensure at least one interval fits
                    # Start sampling from waypoint_interval_m up to length - half_interval (to avoid too close to end)
                    num_intervals_fit = math.floor(centerline.length / waypoint_interval_m)
                    for i in range(1, int(num_intervals_fit) +1 ): # Iterate from 1 up to number of full intervals
                        dist = i * waypoint_interval_m
                        if dist < centerline.length: # Check if distance is strictly within line length
                             sampled_points_on_line.append(centerline.interpolate(dist))
                
                # Add end point, ensuring it's different from the last sampled point if any
                end_point_line = centerline.interpolate(centerline.length)
                if not sampled_points_on_line or not end_point_line.equals_exact(sampled_points_on_line[-1], tolerance=1e-4):
                    sampled_points_on_line.append(end_point_line)
                
                all_waypoint_candidates_world_crs.extend(sampled_points_on_line)
    
    print(f"  Processed {processed_polygons} polygons, found {len(all_waypoint_candidates_world_crs)} raw waypoint candidates from centerlines.")

    # Deduplicate points based on rounded coordinates to avoid micro-movements
    final_unique_waypoints_world_crs = []
    seen_coords_tuples_for_wp = set()
    for p_world in all_waypoint_candidates_world_crs:
        coord_tuple_world = (round(p_world.x, 3), round(p_world.y, 3)) # Rounded to mm
        if coord_tuple_world not in seen_coords_tuples_for_wp:
            final_unique_waypoints_world_crs.append(p_world)
            seen_coords_tuples_for_wp.add(coord_tuple_world)
    
    print(f"  Found {len(final_unique_waypoints_world_crs)} unique waypoint candidates after deduplication.")


    # 4. Convert default orientation from Euler (radians) to Quaternion
    try:
        q = euler_to_quaternion(default_orientation_euler_rad[0],
                                default_orientation_euler_rad[1],
                                default_orientation_euler_rad[2])
        default_orientation_q_dict = {'x': q[0], 'y': q[1], 'z': q[2], 'w': q[3]}
    except Exception as e:
        print(f"Warning: Could not convert Euler to Quaternion: {e}. Using default identity quaternion.")
        default_orientation_q_dict = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}

    # 5. Transform points to local OBJ frame and create PoseStamped list
    poses_data = []
    for p_world_unique in final_unique_waypoints_world_crs:
        x_intermediate, y_intermediate = p_world_unique.x, p_world_unique.y
        x_local = x_intermediate - lx_world_origin
        y_local = y_intermediate - ly_world_origin
        z_local = waypoint_z_in_local_frame

        pose_stamped_dict = {
            'header': {
                'frame_id': map_frame_id,
                'stamp': {'sec': 0, 'nanosec': 0} 
            },
            'pose': {
                'position': {'x': round(x_local, 4), 'y': round(y_local, 4), 'z': round(z_local, 4)},
                'orientation': default_orientation_q_dict
            }
        }
        poses_data.append(pose_stamped_dict)

    if not poses_data:
        print("  No valid waypoints generated after processing all cracks.")
        # Still save an empty list as per earlier logic for empty GML
        with open(output_yaml_path, 'w') as f:
            yaml.dump({'poses': []}, f, sort_keys=False, default_flow_style=None, indent=2)
        print(f"  Empty waypoints YAML saved to: {output_yaml_path}")
        return True


    # 6. Write YAML file
    final_yaml_structure = {'poses': poses_data}
    try:
        with open(output_yaml_path, 'w') as f:
            yaml.dump(final_yaml_structure, f, sort_keys=False, default_flow_style=None, indent=2)
        print(f"  Successfully saved {len(poses_data)} waypoints (from cracks) to: {output_yaml_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to write YAML file {output_yaml_path}: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("--- Testing Waypoint YAML Generation from Cracks GML Centerlines ---")
    
    test_output_dir = Path("output_test_waypoints_from_gml")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy GML for cracks
    dummy_cracks_gml_content = """<?xml version="1.0" encoding="utf-8" ?>
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2" xmlns:ogr="http://ogr.maptools.org/">
  <gml:featureMember>
    <ogr:crack_feature_1 srsName="urn:ogc:def:crs:EPSG::25832">
      <ogr:geometryProperty><gml:Polygon>
        <gml:exterior><gml:LinearRing>
          <gml:posList>377010.0 5748010.0 377020.0 5748011.0 377020.0 5748012.0 377010.0 5748011.0 377010.0 5748010.0</gml:posList>
        </gml:LinearRing></gml:exterior>
      </gml:Polygon></ogr:geometryProperty>
    </ogr:crack_feature_1>
  </gml:featureMember>
  <gml:featureMember>
    <ogr:crack_feature_2 srsName="urn:ogc:def:crs:EPSG::25832">
      <ogr:geometryProperty><gml:Polygon>
        <gml:exterior><gml:LinearRing>
          <gml:posList>377030.0 5748020.0 377030.5 5748030.0 377031.0 5748030.0 377030.5 5748020.0 377030.0 5748020.0</gml:posList>
        </gml:LinearRing></gml:exterior>
      </gml:Polygon></ogr:geometryProperty>
    </ogr:crack_feature_2>
  </gml:featureMember>
</gml:FeatureCollection>
    """
    dummy_cracks_gml_path = test_output_dir / "dummy_cracks_for_waypoints_test.gml"
    with open(dummy_cracks_gml_path, "w") as f:
        f.write(dummy_cracks_gml_content)

    output_yaml = test_output_dir / "test_nav_waypoints_from_gml.yaml"

    success = generate_waypoints_yaml(
        cracks_gml_path_str=str(dummy_cracks_gml_path),
        intermediate_crs_str='EPSG:25832', 
        cracks_gml_fallback_crs_str='EPSG:25832',
        obj_local_frame_origin_world_xy=(377000.0, 5748000.0), 
        waypoint_z_in_local_frame=0.1,    
        output_yaml_path_str=str(output_yaml),
        waypoint_interval_m=2.0, # Sample points every 2 meters along centerline
        default_orientation_euler_rad=[0.0, 0.0, math.radians(0)],
        map_frame_id="map_local_test"    
    )

    if success:
        print(f"\nTest waypoints YAML (from GML) generated: {output_yaml.resolve()}")
        if output_yaml.exists():
            with open(output_yaml, 'r') as f_read:
                print("\nYAML Content:")
                print(f_read.read())
    else:
        print("\nTest waypoints YAML (from GML) generation FAILED.")

    # dummy_cracks_gml_path.unlink(missing_ok=True) # Keep for inspection if needed
    print("--- Test Finished ---")