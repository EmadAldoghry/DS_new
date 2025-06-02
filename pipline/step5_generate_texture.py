# pipline/step5_generate_texture.py
from pathlib import Path
# import fiona # Not strictly needed if using GeoPandas for GML read
import rasterio
import rasterio.mask
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from shapely.geometry import shape, mapping # mapping might be used if fiona was used
from shapely.ops import transform as shapely_transform, unary_union
import numpy as np
from PIL import Image
import pyproj
import traceback
import requests
from io import BytesIO
import geopandas as gpd

# Use the shared plotting helper if available
from helpers import plot_geometries, plot_image_array # Ensure helpers.py is accessible

def generate_texture_from_polygon(
    polygon_gml_path_str,
    output_dir_str,             # Directory where the final OBJ/MTL/PNG will reside
    output_texture_filename,    # Just the filename for the texture (e.g., "base_texture.png")
    wms_url, wms_layer, wms_version, wms_format,
    wms_width, wms_height, target_wms_crs_str,
    wms_bbox_padding, polygon_crs_fallback_str,
    fill_color_rgb,
    show_plots=True, save_plots=True, plot_dpi=150):
    """
    Reads polygons from GML, downloads WMS, crops, and saves texture PNG
    directly into the specified output_dir_str.

    Returns:
        tuple: (path_to_texture_str, texture_affine_transform, texture_crs_pyproj_obj)
               Returns (None, None, None) on failure.
    """
    print("\n--- Running: Step 5 - Texture Generation ---")
    polygon_path = Path(polygon_gml_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_texture_path = output_dir / output_texture_filename
    
    target_crs_obj = None # Initialize to be accessible in return on early failure
    try:
        target_crs_obj = pyproj.CRS.from_user_input(target_wms_crs_str)
    except Exception as e_crs_init:
        print(f"  Error: Invalid target_wms_crs_str '{target_wms_crs_str}': {e_crs_init}")
        return None, None, None

    # Create a subdirectory *within* the main output for texture-specific plots
    texture_plot_dir = output_dir / "texture_plots"
    if save_plots: texture_plot_dir.mkdir(parents=True, exist_ok=True)

    if not polygon_path.is_file():
        print(f"  Error: Polygon GML file not found: {polygon_path}"); return None, None, None

    # --- Step 1: Read Polygons & Reproject ---
    print(f"  Step 1: Reading Polygons from {polygon_path.name}...")
    geoms_for_processing = [] # Will store Shapely geometries in target_crs_obj
    input_crs_for_plot = None # Store original CRS for initial plot if desired
    try:
        gdf = gpd.read_file(polygon_path)
        if gdf.empty:
            print("  Warning: Polygon GML is empty."); return None, None, None

        input_crs_for_plot = gdf.crs # Store for potential plotting of original
        
        if gdf.crs:
            if not gdf.crs.equals(target_crs_obj):
                print(f"    Reprojecting source polygons from {gdf.crs.srs} to {target_crs_obj.srs}...")
                gdf = gdf.to_crs(target_crs_obj)
            else:
                print(f"    Source polygons already in target CRS {target_crs_obj.srs}.")
        else:
            print(f"    Warning: Polygon GML has no CRS. Assuming fallback: {polygon_crs_fallback_str}")
            try:
                fallback_crs_obj = pyproj.CRS.from_user_input(polygon_crs_fallback_str)
                input_crs_for_plot = fallback_crs_obj # Update for plotting
                gdf.crs = fallback_crs_obj
                if not fallback_crs_obj.equals(target_crs_obj):
                    print(f"    Reprojecting source polygons from fallback {fallback_crs_obj.srs} to {target_crs_obj.srs}...")
                    gdf = gdf.to_crs(target_crs_obj)
                else:
                    print(f"    Fallback CRS matches target CRS {target_crs_obj.srs}.")
            except Exception as fallback_err:
                print(f"    Error setting/using fallback CRS '{polygon_crs_fallback_str}': {fallback_err}"); return None, None, None
        
        # gdf is now in target_crs_obj
        for geom in gdf.geometry:
            if geom is None or geom.is_empty: continue
            cleaned = geom if geom.is_valid else geom.buffer(0)
            if cleaned.is_valid and not cleaned.is_empty and cleaned.geom_type in ['Polygon', 'MultiPolygon']:
                geoms_for_processing.append(cleaned)
        
        if not geoms_for_processing:
            print("  Error: No valid Polygon/MultiPolygon found after reading/reprojection."); return None, None, None
        print(f"    Found {len(geoms_for_processing)} valid Polygon/MultiPolygon features for texture base.")

        plot_geometries(geoms_for_processing, target_crs_obj,
                        f"Step 5 Pre-Merge: Input Polygons for Texture (Target CRS: {target_crs_obj.srs})",
                        texture_plot_dir, "texture_plot_01_input_reproj",
                        show_plots=show_plots, save_plots=save_plots, plot_dpi=plot_dpi)

    except Exception as e:
        print(f"  Error in Step 1 (Read GML for Texture): {e}"); traceback.print_exc(); return None, None, None

    # --- Step 2: Merge Polygons (for WMS BBOX calculation) ---
    print("  Step 2: Merging geometries for BBOX...")
    merged_geom_for_bbox = None
    try:
        merged_geom_for_bbox = unary_union(geoms_for_processing).buffer(0) # Ensure validity
        if merged_geom_for_bbox.is_empty or not merged_geom_for_bbox.is_valid:
            print("  Error: Merged geometry for BBOX is invalid/empty."); return None, None, None
        print(f"    Merged geometry type for BBOX: {merged_geom_for_bbox.geom_type}")
        plot_geometries(merged_geom_for_bbox, target_crs_obj,
                        "Step 5: Merged Polygon for Texture BBOX",
                        texture_plot_dir, "texture_plot_02_merged",
                        show_plots=show_plots, save_plots=save_plots, plot_dpi=plot_dpi)
    except Exception as e:
        print(f"  Error in Step 2 (Merge for Texture BBOX): {e}"); traceback.print_exc(); return None, None, None

    # --- Step 3: Download WMS ---
    print(f"  Step 3: Downloading WMS Layer '{wms_layer}'...")
    wms_image_array, wms_transform, wms_meta, wms_bounds_tuple = None, None, None, None
    try:
        min_x, min_y, max_x, max_y = merged_geom_for_bbox.bounds
        # Calculate WMS request BBOX with padding
        wms_req_bbox = (min_x - wms_bbox_padding, 
                        min_y - wms_bbox_padding, 
                        max_x + wms_bbox_padding, 
                        max_y + wms_bbox_padding)
        wms_bounds_tuple = wms_req_bbox # Store for plotting

        wms_params = {
            'service': 'WMS', 'request': 'GetMap', 'version': wms_version, 
            'layers': wms_layer, 'styles': '',
            'crs': target_wms_crs_str, # This should be the CRS of the WMS server, which is target_crs_obj.srs
            'bbox': f"{wms_req_bbox[0]},{wms_req_bbox[1]},{wms_req_bbox[2]},{wms_req_bbox[3]}",
            'width': str(wms_width), 'height': str(wms_height), 'format': wms_format
        }
        print(f"    WMS Request URL (approx): {requests.Request('GET', wms_url, params=wms_params).prepare().url}")

        response = requests.get(wms_url, params=wms_params, timeout=180)
        response.raise_for_status() # Will raise HTTPError for bad responses (4xx or 5xx)
        print(f"    WMS Response Status: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")

        with Image.open(BytesIO(response.content)) as img:
            target_mode = 'RGB'
            if img.mode != target_mode:
                print(f"    Converting WMS image from mode {img.mode} to {target_mode}")
                img = img.convert(target_mode)
            
            wms_image_array = np.array(img) # H, W, C
            if wms_image_array.ndim == 3 and wms_image_array.shape[2] == 3: # Check for 3 channels
                wms_image_array = np.transpose(wms_image_array, (2, 0, 1)) # C, H, W for rasterio
            elif wms_image_array.ndim == 2: # Grayscale image
                print("    WMS returned grayscale, converting to 3-band RGB.")
                wms_image_array = np.stack([wms_image_array]*3, axis=0) # C, H, W
            else:
                raise ValueError(f"Unexpected WMS image array shape: {wms_image_array.shape}")
        
        if wms_image_array.shape[0] != 3: # Ensure 3 bands after processing
            raise ValueError(f"Failed to process WMS image into 3-band array (final shape: {wms_image_array.shape})")

        wms_transform = from_bounds(*wms_req_bbox, wms_image_array.shape[2], wms_image_array.shape[1]) # width, height from array
        wms_meta = {
            'driver': 'MEM', # In-memory raster
            'height': wms_image_array.shape[1], 
            'width': wms_image_array.shape[2],
            'count': 3, # Expect 3 bands (RGB)
            'dtype': wms_image_array.dtype,
            'crs': target_crs_obj, # CRS of the WMS image
            'transform': wms_transform
        }
        plot_image_array(wms_image_array, wms_transform, target_crs_obj, "Step 5: Full WMS Download",
                         texture_plot_dir, "texture_plot_03a_wms_full",
                         show_plots=show_plots, save_plots=save_plots, plot_dpi=plot_dpi)
        plot_geometries(merged_geom_for_bbox, target_crs_obj, "Step 5: Polygon on WMS Bounds", texture_plot_dir,
                        "texture_plot_03b_poly_on_wms", raster_bounds_tuple=wms_bounds_tuple,
                        show_plots=show_plots, save_plots=save_plots, plot_dpi=plot_dpi)
    except requests.exceptions.RequestException as e_req:
        print(f"  Error in Step 3 (WMS Request/Download): {e_req}"); traceback.print_exc(); return None, None, None
    except Exception as e:
        print(f"  Error in Step 3 (WMS Processing): {e}"); traceback.print_exc(); return None, None, None

    # --- Step 4: Crop WMS Image & Fill ---
    print("  Step 4: Cropping WMS image...")
    out_transform_cropped = None # Initialize to be accessible for return
    try:
        with MemoryFile() as memfile:
            with memfile.open(**wms_meta) as src_wms: # src_wms is a rasterio dataset reader
                src_wms.write(wms_image_array)
                
                # Use the original list of valid polygons for masking (geoms_for_processing)
                # This preserves internal holes correctly if they existed in the source GML
                # Convert Shapely geoms to GeoJSON-like mapping for rasterio.mask
                shapely_geoms_for_mask = [g for g in geoms_for_processing if g.is_valid and not g.is_empty]
                if not shapely_geoms_for_mask:
                    raise ValueError("No valid geoms in geoms_for_processing for masking.")
                
                # rasterio.mask.mask expects a list of GeoJSON-like geometry dicts
                geojson_geoms_for_mask = [mapping(g) for g in shapely_geoms_for_mask]

                out_image_masked, out_transform_cropped_temp = rasterio.mask.mask(
                    src_wms, 
                    geojson_geoms_for_mask, 
                    crop=True,     # Crop the output raster to the extent of the shapes
                    filled=False,  # Keep nodata values where shapes don't cover
                    nodata=0       # Value to use for nodata if not specified in src
                )
                out_transform_cropped = out_transform_cropped_temp # Assign to broader scope

        if out_image_masked.shape[1] == 0 or out_image_masked.shape[2] == 0:
            raise ValueError("Cropped image has zero dimensions. Check polygon overlap with WMS.")

        # Fill masked area (where mask is True) with fill_color_rgb
        fill_value_rgb_np = np.array(fill_color_rgb, dtype=out_image_masked.dtype) # Match dtype
        fill_value_array_reshaped = fill_value_rgb_np.reshape((3, 1, 1)) # Reshape for broadcasting (C, H, W)
        
        # out_image_masked is a NumPy MaskedArray. 
        # .data gives the underlying data, .mask is a boolean array (True where masked)
        out_image_filled = np.where(out_image_masked.mask, fill_value_array_reshaped, out_image_masked.data)
        
        # Ensure uint8 RGB for PIL
        out_image_final_for_pil = out_image_filled
        if out_image_final_for_pil.dtype != np.uint8:
            if np.issubdtype(out_image_final_for_pil.dtype, np.floating): # If float, scale 0-1 to 0-255
                out_image_final_for_pil = (out_image_final_for_pil * 255).clip(0, 255)
            out_image_final_for_pil = out_image_final_for_pil.astype(np.uint8)
        
        if out_image_final_for_pil.shape[0] != 3:
            raise ValueError(f"Cropped image does not have 3 bands after processing (shape: {out_image_final_for_pil.shape})")

        # Convert from (C, H, W) to (H, W, C) for PIL
        img_to_save_pil_transposed = np.transpose(out_image_final_for_pil, (1, 2, 0))
        pil_img = Image.fromarray(img_to_save_pil_transposed, 'RGB')
        pil_img.save(output_texture_path)
        print(f"    Cropped/filled texture saved: {output_texture_path}")

        plot_image_array(out_image_filled, out_transform_cropped, target_crs_obj, 
                         "Step 5: Final Cropped Texture",
                         texture_plot_dir, "texture_plot_04_cropped",
                         show_plots=show_plots, save_plots=save_plots, plot_dpi=plot_dpi)
        
        # Return the full path to the texture, its transform, and its CRS object
        return str(output_texture_path), out_transform_cropped, target_crs_obj

    except Exception as e:
        print(f"  Error in Step 4 (Crop/Save Texture): {e}"); traceback.print_exc()
        return None, None, None