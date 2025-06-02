# -*- coding: utf-8 -*-
import os
from pathlib import Path
import geopandas as gpd
import pyproj
from owslib.wfs import WebFeatureService
from lxml import etree # For checking raw GML if GeoPandas fails

def fetch_clip_and_save_wfs(hull_polygon_gpkg_path_str, wfs_url, feature_types,
                            target_crs_str, out_dir_str, bbox_margin=0.0):
    """Fetches features from WFS, clips them to the hull, and saves as GML."""
    print(f"\n--- Running: Fetching and Clipping WFS Features ---")
    output_dir = Path(out_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    target_crs_obj = pyproj.CRS.from_user_input(target_crs_str)

    try:
        hull_gdf = gpd.read_file(hull_polygon_gpkg_path_str)
        if hull_gdf.empty:
            raise ValueError("Hull GeoPackage is empty.")
        hull_polygon = hull_gdf.geometry.iloc[0]
        if not hull_polygon.is_valid: hull_polygon = hull_polygon.buffer(0)
        if not hull_polygon.is_valid: raise ValueError("Hull polygon is invalid.")
        if hull_gdf.crs and not hull_gdf.crs.equals(target_crs_obj):
            print(f"  Reprojecting hull from {hull_gdf.crs} to {target_crs_str} for WFS query.")
            hull_gdf = hull_gdf.to_crs(target_crs_obj)
            hull_polygon = hull_gdf.geometry.iloc[0]
    except Exception as e:
        print(f"ERROR: Could not read or prepare hull polygon from {hull_polygon_gpkg_path_str}: {e}")
        return output_paths


    try:
        print(f"Connecting to WFS service: {wfs_url}")
        wfs = WebFeatureService(wfs_url, version="2.0.0", timeout=60)
    except Exception as e:
        print(f"ERROR: Could not connect to WFS service: {e}")
        return output_paths

    mask_gdf = gpd.GeoDataFrame([1], geometry=[hull_polygon], crs=target_crs_obj)
    minx, miny, maxx, maxy = hull_polygon.bounds
    bbox = (minx - bbox_margin, miny - bbox_margin, maxx + bbox_margin, maxy + bbox_margin)
    print(f"Using bounding box (with {bbox_margin}m margin): {bbox}")

    for typename in feature_types:
        print(f"\nProcessing WFS feature type: {typename}")
        feature_name_safe = typename.replace(':', '_').replace('/', '_')
        raw_path = output_dir / f"{feature_name_safe}_raw.gml"
        out_path = output_dir / f"{feature_name_safe}_clipped.gml"

        try:
            print(f"  Requesting features from WFS...")
            resp = wfs.getfeature(typename=typename, bbox=bbox, srsname=target_crs_obj.to_string(), outputFormat='text/xml; subtype=gml/3.2.1')
            with open(raw_path, 'wb') as f: f.write(resp.read())

            if not raw_path.exists() or raw_path.stat().st_size < 100: # Basic check
                print(f"  Downloaded file {raw_path} is empty or too small. Skipping clipping.")
                raw_path.unlink(missing_ok=True); continue
            print(f"  Reading GML file {raw_path} into GeoDataFrame...")
            try:
                gdf = gpd.read_file(str(raw_path))
            except Exception as read_err:
                print(f"  ERROR reading GML {raw_path}: {read_err}.")
                try: # Check if XML is well-formed at least
                    with open(raw_path, 'rb') as f_xml: etree.parse(f_xml)
                    print("    (XML parsing seems okay, GeoPandas read failed. File might be complex GML.)")
                except Exception as xml_err: print(f"    (XML parsing also failed: {xml_err})")
                continue

            if gdf.empty: print(f"  No features found for {typename} within the bounding box."); raw_path.unlink(missing_ok=True); continue
            print(f"  Found {len(gdf)} features in raw GML.")

            if gdf.crs is None: gdf.crs = target_crs_obj
            elif not pyproj.CRS(gdf.crs).equals(target_crs_obj): gdf = gdf.to_crs(target_crs_obj)

            gdf = gdf[gdf.geometry.is_valid].copy() # Keep only valid
            if gdf.empty: print("  No valid geometries remaining after validity check."); raw_path.unlink(missing_ok=True); continue

            clipped = gpd.clip(gdf, mask_gdf, keep_geom_type=False)
            if clipped.empty: print(f"  No features found within the precise hull for {typename}."); raw_path.unlink(missing_ok=True); continue
            
            clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.is_valid].copy()
            if clipped.empty: print(f"  No valid features left to save for {typename}."); raw_path.unlink(missing_ok=True); continue
            
            clipped.to_file(out_path, driver='GML', GML_FEATURE_COLLECTION=True, GML_ID='auto')
            output_paths[typename] = str(out_path)
            print(f"  Successfully saved {len(clipped)} features to {out_path}")
            raw_path.unlink(missing_ok=True) # Clean up raw file

        except Exception as e:
            print(f"  Unhandled ERROR processing {typename}: {e}")
            out_path.unlink(missing_ok=True); raw_path.unlink(missing_ok=True)
    return output_paths

if __name__ == '__main__':
    # Example Usage
    test_output_dir = Path("output_test_wfs")
    test_output_dir.mkdir(exist_ok=True)
    
    # Create a dummy hull GPKG for testing
    dummy_hull_path = test_output_dir / "dummy_hull.gpkg"
    # Approximate bounds for Cologne, Germany for a real WFS test
    # lat 50.94, lon 6.96. EPSG:25832 (UTM 32N) x: 357000, y: 5640000
    from shapely.geometry import Polygon
    poly = Polygon([(350000, 5630000), (360000, 5630000), (360000, 5640000), (350000, 5640000)])
    gdf_hull = gpd.GeoDataFrame([{'id':1, 'geometry': poly}], crs="EPSG:25832")
    gdf_hull.to_file(dummy_hull_path, driver="GPKG")

    wfs_paths = fetch_clip_and_save_wfs(
        hull_polygon_gpkg_path_str=str(dummy_hull_path),
        wfs_url='https://www.wfs.nrw.de/geobasis/wfs_nw_alkis_aaa-modell-basiert',
        feature_types=['adv:AX_Strassenverkehr'], # Keep it small for testing
        target_crs_str='EPSG:25832',
        out_dir_str=str(test_output_dir)
    )
    print("WFS Fetch Test Complete. Output paths:", wfs_paths)