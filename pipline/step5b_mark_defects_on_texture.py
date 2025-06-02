# pipline/step5b_mark_defects_on_texture.py
import pandas as pd
import pyproj
from shapely import wkt # For loading WKT polygons
from shapely.ops import transform as shapely_transform # For transforming Shapely geometries
import cv2 # OpenCV for image manipulation
import numpy as np
from pathlib import Path
import rasterio.transform # For Affine transform methods

def mark_defects_on_texture(
    base_texture_path_str: str,
    texture_affine_transform, # rasterio.transform.Affine object
    texture_crs_pyproj_obj,   # pyproj.CRS object of the texture
    csv_path_str: str,
    # defect_source_crs_str is no longer needed globally, as each row has its own CRS
    # defect_size_m is no longer needed
    defect_color_bgr: tuple = (0, 0, 0) # Black in BGR for OpenCV
) -> bool:
    """
    Loads a texture image, reads defect polygons (WKT) from a CSV,
    projects them onto the texture, draws them as black polygons,
    and overwrites the original texture file.

    Args:
        base_texture_path_str: Path to the texture image to modify.
        texture_affine_transform: Affine transform of the texture.
        texture_crs_pyproj_obj: pyproj.CRS object of the texture.
        csv_path_str: Path to the CSV file with defect polygons (geometry_wkt, optimal_epsg_code).
        defect_color_bgr: Color to draw defects (BGR format for OpenCV).

    Returns:
        True if successful, False otherwise.
    """
    print("\n--- Running: Step 5b - Marking Defect Polygons on Texture ---")
    base_texture_path = Path(base_texture_path_str)
    csv_path = Path(csv_path_str)

    if not base_texture_path.exists():
        print(f"  Error: Base texture file not found: {base_texture_path}")
        return False
    if not csv_path.exists():
        print(f"  Error: Defect CSV file not found: {csv_path}")
        return False

    # 1. Load the base texture image
    try:
        image_cv2 = cv2.imread(str(base_texture_path))
        if image_cv2 is None:
            print(f"  Error: Could not load image from {base_texture_path}")
            return False
        img_height, img_width = image_cv2.shape[:2]
        print(f"  Loaded texture: {base_texture_path.name} ({img_width}x{img_height})")
    except Exception as e:
        print(f"  Error loading texture image: {e}")
        return False

    # 2. Read defect CSV
    try:
        df_defects = pd.read_csv(csv_path)
        required_cols = {'geometry_wkt', 'optimal_epsg_code'}
        if not required_cols.issubset(df_defects.columns):
            missing = required_cols - set(df_defects.columns)
            raise ValueError(f"Defect CSV must contain columns: {missing}")
    except Exception as e:
        print(f"  Error reading or validating defect CSV {csv_path}: {e}")
        return False

    # 3. Process and draw each defect polygon
    defects_drawn = 0
    # Cache transformers to avoid re-creating them for the same EPSG code
    transformer_cache = {}

    for index, row in df_defects.iterrows():
        try:
            wkt_string = row['geometry_wkt']
            defect_poly_original_crs_str = f"EPSG:{row['optimal_epsg_code']}"

            # Load WKT polygon
            defect_polygon_orig_crs = wkt.loads(wkt_string)
            if not defect_polygon_orig_crs.is_valid or defect_polygon_orig_crs.is_empty:
                print(f"  Warning: Invalid or empty WKT polygon at row {index}. Skipping.")
                continue

            # Get or create transformer
            if defect_poly_original_crs_str not in transformer_cache:
                try:
                    transformer_cache[defect_poly_original_crs_str] = pyproj.Transformer.from_crs(
                        pyproj.CRS.from_user_input(defect_poly_original_crs_str),
                        texture_crs_pyproj_obj,
                        always_xy=True
                    )
                except Exception as e_crs:
                    print(f"  Warning: Could not create transformer for {defect_poly_original_crs_str} at row {index}: {e_crs}. Skipping.")
                    continue
            
            transformer = transformer_cache[defect_poly_original_crs_str]

            # Transform defect polygon to texture's CRS
            defect_polygon_texture_crs = shapely_transform(transformer.transform, defect_polygon_orig_crs)
            
            if not defect_polygon_texture_crs.is_valid or defect_polygon_texture_crs.is_empty:
                print(f"  Warning: Defect polygon became invalid/empty after transformation at row {index}. Skipping.")
                continue
            
            # Extract exterior coordinates (and interior if it's a Polygon with holes, though less likely for defects)
            # For simplicity, we'll just draw the exterior of the transformed polygon.
            # If you have MultiPolygons in WKT, this would need to iterate through `geom.geoms`
            if defect_polygon_texture_crs.geom_type == 'Polygon':
                exterior_coords_world = list(defect_polygon_texture_crs.exterior.coords)
            elif defect_polygon_texture_crs.geom_type == 'MultiPolygon':
                # Handle MultiPolygon: draw each part. For simplicity, let's take the largest.
                # A more robust solution would iterate and draw all.
                print(f"  Note: Defect at row {index} is a MultiPolygon. Processing largest part.")
                largest_part = max(defect_polygon_texture_crs.geoms, key=lambda p: p.area, default=None)
                if largest_part:
                    exterior_coords_world = list(largest_part.exterior.coords)
                else:
                    print(f"  Warning: Empty MultiPolygon at row {index}. Skipping.")
                    continue
            else:
                print(f"  Warning: Defect at row {index} is not a Polygon or MultiPolygon after transform ({defect_polygon_texture_crs.geom_type}). Skipping.")
                continue

            # Convert world polygon vertices to pixel coordinates
            pixel_coords_list = []
            for xw, yw in exterior_coords_world:
                col, row = ~texture_affine_transform * (xw, yw)
                pixel_coords_list.append([int(round(col)), int(round(row))])
            
            if len(pixel_coords_list) < 3:
                 print(f"  Warning: Not enough pixel coordinates for polygon at row {index}. Skipping.")
                 continue

            defect_poly_pixel_np = np.array([pixel_coords_list], dtype=np.int32)
            cv2.fillPoly(image_cv2, [defect_poly_pixel_np], defect_color_bgr)
            defects_drawn += 1

        except Exception as e:
            label = row.get('label_name', f'row {index}')
            print(f"  Warning: Could not process or draw defect '{label}': {e}")
            import traceback
            traceback.print_exc() # More detailed error for debugging
            continue
    
    print(f"  Drew {defects_drawn} defect polygons onto the texture.")

    # 4. Save the modified texture (overwriting the original)
    try:
        cv2.imwrite(str(base_texture_path), image_cv2)
        print(f"  Successfully saved modified (defect-marked) texture to: {base_texture_path}")
        return True
    except Exception as e:
        print(f"  Error saving modified texture: {e}")
        return False

if __name__ == '__main__':
    # --- Example Usage (for testing this module independently) ---
    print("--- Testing Step 5b: Mark Defect WKT Polygons on Texture ---")
    
    # Create a dummy CSV for defects with WKT
    dummy_csv_content = (
        "id,label_name,geometry_wkt,optimal_epsg_code\n"
        "1,DEFECT_A,\"POLYGON((396005.0 5693994.0, 396006.0 5693994.0, 396006.0 5693995.0, 396005.0 5693995.0, 396005.0 5693994.0))\",25832\n"
        "2,DEFECT_B,\"POLYGON((396010.0 5693990.0, 396010.5 5693990.0, 396010.5 5693990.5, 396010.0 5693990.5, 396010.0 5693990.0))\",25832"
    )
    dummy_csv_path = Path("dummy_defect_wkt_for_texture_test.csv")
    with open(dummy_csv_path, "w") as f:
        f.write(dummy_csv_content)

    dummy_texture_path = Path("dummy_base_texture_for_defect_wkt_test.png")
    dummy_image_array = np.full((100, 150, 3), (200, 200, 200), dtype=np.uint8) # Light gray
    cv2.imwrite(str(dummy_texture_path), dummy_image_array)

    from rasterio.transform import from_origin
    dummy_texture_transform = from_origin(396000, 5694000, 0.1, 0.1) # X_orig, Y_orig_TOP, pixel_width, pixel_height
    dummy_texture_crs = pyproj.CRS.from_epsg(25832)

    # Defect A: (396005, 5693994) to (396006, 5693995)
    # Pixel for (396005, 5693994): col=(396005-396000)/0.1=50, row=(5694000-5693994)/0.1=60
    # Pixel for (396006, 5693995): col=(396006-396000)/0.1=60, row=(5694000-5693995)/0.1=50

    success = mark_defects_on_texture(
        base_texture_path_str=str(dummy_texture_path),
        texture_affine_transform=dummy_texture_transform,
        texture_crs_pyproj_obj=dummy_texture_crs,
        csv_path_str=str(dummy_csv_path),
        defect_color_bgr=(0,0,255) # Draw in RED for testing visibility
    )

    if success:
        print(f"Test successful. Modified texture saved to: {dummy_texture_path.resolve()}")
        print("Please visually inspect the image to confirm red polygons.")
    else:
        print("Test failed.")

    dummy_csv_path.unlink(missing_ok=True)
    # dummy_texture_path.unlink(missing_ok=True) # Keep for inspection
    print("--- Test Finished ---")