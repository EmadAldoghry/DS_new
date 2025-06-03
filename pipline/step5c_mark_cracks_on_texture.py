# pipline/step5c_mark_cracks_on_texture.py
import geopandas as gpd
import pyproj
from shapely.ops import transform as shapely_transform
import cv2
import numpy as np
from pathlib import Path
import traceback

def mark_cracks_from_gml_on_texture(
    base_texture_path_str: str,
    texture_affine_transform, # rasterio.transform.Affine object
    texture_crs_pyproj_obj,   # pyproj.CRS object of the texture
    cracks_gml_path_str: str,
    texture_fallback_crs_str: str, # Fallback CRS if GML has no CRS info
    crack_color_bgr: tuple = (0, 0, 0) # Default black
) -> bool:
    """
    Loads a texture image, reads crack polygons from a GML file,
    projects them onto the texture, draws them, and overwrites the texture file.
    """
    print("\n--- Running: Step 5c - Marking Crack Polygons (from GML) on Texture ---")
    base_texture_path = Path(base_texture_path_str)
    cracks_gml_path = Path(cracks_gml_path_str)

    if not base_texture_path.exists():
        print(f"  Error: Base texture file not found: {base_texture_path}")
        return False
    if not cracks_gml_path.exists():
        print(f"  Error: Cracks GML file not found: {cracks_gml_path}")
        return False

    # 1. Load the base texture image
    try:
        image_cv2 = cv2.imread(str(base_texture_path))
        if image_cv2 is None:
            print(f"  Error: Could not load image from {base_texture_path}")
            return False
        img_height, img_width = image_cv2.shape[:2]
        print(f"  Loaded texture: {base_texture_path.name} ({img_width}x{img_height}) for crack marking.")
    except Exception as e:
        print(f"  Error loading texture image: {e}")
        traceback.print_exc()
        return False

    # 2. Read crack polygons from GML
    try:
        gdf_cracks = gpd.read_file(cracks_gml_path)
        if gdf_cracks.empty:
            print(f"  Warning: No features found in cracks GML file: {cracks_gml_path}")
            return True # No cracks to mark, not an error for this step
    except Exception as e:
        print(f"  Error reading cracks GML {cracks_gml_path}: {e}")
        traceback.print_exc()
        return False

    # 3. Ensure crack polygons are in the texture's CRS
    try:
        if gdf_cracks.crs:
            if not gdf_cracks.crs.equals(texture_crs_pyproj_obj):
                print(f"    Reprojecting cracks from {gdf_cracks.crs.srs} to {texture_crs_pyproj_obj.srs}...")
                gdf_cracks = gdf_cracks.to_crs(texture_crs_pyproj_obj)
        else:
            print(f"    Warning: Cracks GML has no CRS. Assuming fallback: {texture_fallback_crs_str}")
            fallback_crs_obj = pyproj.CRS.from_user_input(texture_fallback_crs_str)
            gdf_cracks.crs = fallback_crs_obj # Set CRS before potential transform
            if not fallback_crs_obj.equals(texture_crs_pyproj_obj):
                print(f"    Reprojecting cracks from fallback {fallback_crs_obj.srs} to {texture_crs_pyproj_obj.srs}...")
                gdf_cracks = gdf_cracks.to_crs(texture_crs_pyproj_obj)
    except Exception as e_proj:
        print(f"  Error reprojecting crack geometries: {e_proj}")
        traceback.print_exc()
        return False
    
    # 4. Process and draw each crack polygon
    cracks_drawn = 0
    for index, row_series in gdf_cracks.iterrows():
        geom = row_series.geometry # Get the geometry object from the row Series
        if geom is None or geom.is_empty or not geom.is_valid:
            print(f"  Warning: Invalid or empty crack geometry at index {index}. Skipping.")
            continue

        polygons_to_draw = []
        if geom.geom_type == 'Polygon':
            polygons_to_draw.append(geom)
        elif geom.geom_type == 'MultiPolygon':
            polygons_to_draw.extend(list(geom.geoms))
        else:
            print(f"  Warning: Crack geometry at index {index} is not Polygon or MultiPolygon ({geom.geom_type}). Skipping.")
            continue

        for poly_idx, crack_polygon_texture_crs in enumerate(polygons_to_draw):
            if not crack_polygon_texture_crs.is_valid or crack_polygon_texture_crs.is_empty:
                 print(f"  Warning: Sub-polygon (index {index}, part {poly_idx}) invalid/empty. Skipping.")
                 continue
            try:
                exterior_coords_world = list(crack_polygon_texture_crs.exterior.coords)
                
                pixel_coords_list = []
                for wx, wy in exterior_coords_world:
                    # Transform world coords to pixel coords
                    # The ~ operator on an Affine transform gives its inverse
                    col, row = ~texture_affine_transform * (wx, wy)
                    pixel_coords_list.append([int(round(col)), int(round(row))])
                
                if len(pixel_coords_list) < 3:
                    print(f"  Warning: Not enough pixel coordinates for crack polygon (index {index}, part {poly_idx}). Skipping.")
                    continue

                crack_poly_pixel_np = np.array([pixel_coords_list], dtype=np.int32)
                cv2.fillPoly(image_cv2, [crack_poly_pixel_np], crack_color_bgr)
                # Optionally draw outlines if desired, e.g., cv2.polylines(...)
                cracks_drawn += 1

            except Exception as e_draw:
                print(f"  Warning: Could not process or draw crack polygon (index {index}, part {poly_idx}): {e_draw}")
                traceback.print_exc()
                continue
    
    print(f"  Drew {cracks_drawn} crack polygons from GML onto the texture.")

    # 5. Save the modified texture (overwriting the original)
    try:
        cv2.imwrite(str(base_texture_path), image_cv2)
        print(f"  Successfully saved modified (crack-marked) texture to: {base_texture_path}")
        return True
    except Exception as e_save:
        print(f"  Error saving modified texture: {e_save}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("--- Testing Step 5c: Mark Crack GML Polygons on Texture ---")
    
    test_output_dir = Path("output_test_step5c")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy GML for cracks
    dummy_cracks_gml_content = """<?xml version="1.0" encoding="utf-8" ?>
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2" xmlns:ogr="http://ogr.maptools.org/">
  <gml:featureMember>
    <ogr:crack_feature>
      <ogr:geometryProperty><gml:Polygon srsName="urn:ogc:def:crs:EPSG::25832">
        <gml:exterior><gml:LinearRing>
          <gml:posList>396005.0 5693994.0 396006.0 5693994.0 396006.0 5693995.0 396005.0 5693995.0 396005.0 5693994.0</gml:posList>
        </gml:LinearRing></gml:exterior>
      </gml:Polygon></ogr:geometryProperty>
    </ogr:crack_feature>
  </gml:featureMember>
  <gml:featureMember>
    <ogr:crack_feature>
      <ogr:geometryProperty><gml:Polygon srsName="urn:ogc:def:crs:EPSG::25832">
        <gml:exterior><gml:LinearRing>
          <gml:posList>396010.0 5693990.0 396010.5 5693990.0 396010.5 5693990.5 396010.0 5693990.5 396010.0 5693990.0</gml:posList>
        </gml:LinearRing></gml:exterior>
      </gml:Polygon></ogr:geometryProperty>
    </ogr:crack_feature>
  </gml:featureMember>
</gml:FeatureCollection>
    """
    dummy_cracks_gml_path = test_output_dir / "dummy_cracks_for_texture_test.gml"
    with open(dummy_cracks_gml_path, "w") as f:
        f.write(dummy_cracks_gml_content)

    dummy_texture_path = test_output_dir / "dummy_base_texture_for_cracks_test.png"
    dummy_image_array = np.full((100, 150, 3), (200, 200, 200), dtype=np.uint8) # Light gray
    cv2.imwrite(str(dummy_texture_path), dummy_image_array)

    from rasterio.transform import from_origin
    dummy_texture_transform = from_origin(396000, 5694000, 0.1, 0.1) # X_orig, Y_orig_TOP, pix_w, pix_h
    dummy_texture_crs = pyproj.CRS.from_epsg(25832)
    dummy_fallback_crs = "EPSG:25832"

    success = mark_cracks_from_gml_on_texture(
        base_texture_path_str=str(dummy_texture_path),
        texture_affine_transform=dummy_texture_transform,
        texture_crs_pyproj_obj=dummy_texture_crs,
        cracks_gml_path_str=str(dummy_cracks_gml_path),
        texture_fallback_crs_str=dummy_fallback_crs,
        crack_color_bgr=(0,0,0) # Black
    )

    if success:
        print(f"Test successful. Modified texture saved to: {dummy_texture_path.resolve()}")
        print("Please visually inspect the image to confirm black polygons.")
    else:
        print("Test failed.")
    
    # dummy_cracks_gml_path.unlink(missing_ok=True)
    # dummy_texture_path.unlink(missing_ok=True) # Keep for inspection
    print("--- Test Finished ---")