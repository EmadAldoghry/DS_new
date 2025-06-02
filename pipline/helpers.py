# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from descartes import PolygonPatch
import geopandas as gpd
import pyproj
from shapely.geometry import Point, LineString, Polygon, MultiPoint, box
import rasterio
from rasterio.plot import show as rio_show
import numpy as np
from pathlib import Path

# =============================================================================
# Helper Plotting Function for GML Analysis
# =============================================================================
# helpers.py
def plot_analysis_step(title, step_num, total_steps, elements_to_plot,
                       output_dir, filename_prefix,
                       hull_polygon=None, hull_centroid=None,
                       show_plots=True, save_plots=True, plot_dpi=150):
    print(f"\nGenerating Plot {step_num}/{total_steps}: {title}")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"Step {step_num}/{total_steps}: {title}")
    ax.set_xlabel("X Coordinate (Projected)")
    ax.set_ylabel("Y Coordinate (Projected)")
    ax.set_aspect('equal', adjustable='box')
    legend_elements = []

    if hull_polygon and hull_polygon.is_valid:
        hx, hy = hull_polygon.exterior.xy
        ax.plot(hx, hy, color='gray', linestyle='--', linewidth=1, label='_nolegend_', zorder=0)
        if not any(le.get_label() == 'Convex Hull Outline' for le in legend_elements if isinstance(le, Line2D)):
            legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', lw=1, label='Convex Hull Outline'))

    if hull_centroid:
         ax.plot(hull_centroid.x, hull_centroid.y, 'k+', markersize=6, label='_nolegend_', zorder=1)
         if not any(le.get_label() == 'Hull Centroid' for le in legend_elements if isinstance(le, Line2D)):
             legend_elements.append(Line2D([0], [0], marker='+', color='black', markersize=6, linestyle='None', label='Hull Centroid'))

    print(f"DEBUG plot_analysis_step: Type of elements_to_plot: {type(elements_to_plot)}") # Outer debug
    if not isinstance(elements_to_plot, list):
        print(f"DEBUG plot_analysis_step: elements_to_plot IS NOT A LIST. Value: {elements_to_plot}")
        # Potentially raise an error or return early if this is unexpected
        # return

    for item_idx, item in enumerate(elements_to_plot):
        print(f"DEBUG plot_analysis_step: Processing item {item_idx}, type: {type(item)}") # Inner debug
        if not isinstance(item, dict):
            print(f"DEBUG plot_analysis_step: Item {item_idx} IS NOT A DICT. Value: {item}. Skipping this item.")
            continue # Skip this malformed item

        geoms = item.get('geoms', [])
        print(f"DEBUG plot_analysis_step: Item {item_idx} 'geoms' type: {type(geoms)}") # Debug for geoms

        # Ensure geoms is a list or similar iterable, handle None or single geom
        if isinstance(geoms, (Point, LineString, Polygon, MultiPoint)): # Check if geoms is already a single shapely geometry
            print(f"DEBUG plot_analysis_step: Item {item_idx} 'geoms' was a single Shapely object, wrapping in list.")
            geoms = [geoms]
        elif not isinstance(geoms, (list, tuple, gpd.GeoSeries)):
            print(f"DEBUG plot_analysis_step: Item {item_idx} 'geoms' is not list/tuple/GeoSeries (type: {type(geoms)}), setting to empty list.")
            geoms = []

        if not geoms:
            print(f"DEBUG plot_analysis_step: Item {item_idx} has no geoms after processing. Skipping.")
            continue

        color = item.get('color', 'black'); label = item.get('label', '_nolegend_'); lw = item.get('linewidth', 1.5)
        ls = item.get('linestyle', '-'); marker = item.get('marker', None); ms = item.get('markersize', 5)
        zorder = item.get('zorder', 2); alpha = item.get('alpha', 1.0)
        plotted_count = 0; is_points = False
        
        valid_geoms_in_list = []
        for geom_idx, g in enumerate(geoms):
            print(f"DEBUG plot_analysis_step: Item {item_idx}, Geom {geom_idx} type: {type(g)}")
            if g is not None and hasattr(g, 'is_valid') and hasattr(g, 'is_empty') and g.is_valid and not g.is_empty: # More robust check
                valid_geoms_in_list.append(g)
            else:
                print(f"DEBUG plot_analysis_step: Item {item_idx}, Geom {geom_idx} is invalid/empty or not a geometry. Value: {g}")

        if not valid_geoms_in_list:
            print(f"DEBUG plot_analysis_step: Item {item_idx} has no valid_geoms_in_list. Skipping.")
            continue
        # ... rest of the plotting logic for the item ...
        # (The original error was likely before this detailed geometry check, at item.get() or g.is_valid)

        # Determine if the primary geometry type is Point
        if all(isinstance(g, Point) for g in valid_geoms_in_list): is_points = True
        elif any(isinstance(g, Point) for g in valid_geoms_in_list): is_points = False # Treat as lines/polys if mixed

        if is_points:
            xs = [p.x for p in valid_geoms_in_list]; ys = [p.y for p in valid_geoms_in_list]; plotted_count = len(xs)
            ax.scatter(xs, ys, color=color, marker=marker if marker else 'o', s=ms**2, label='_nolegend_', zorder=zorder, edgecolors='face', linewidths=0.5, alpha=alpha)
            if label != '_nolegend_': legend_elements.append(Line2D([0], [0], marker=marker if marker else 'o', color='w', markerfacecolor=color, markersize=ms, linestyle='None', label=f'{label} ({plotted_count})', alpha=alpha))
        else:
            # Plot non-point geometries (Lines, Polygons, etc.)
            plotted_something = False
            for geom_plot_idx, geom in enumerate(valid_geoms_in_list): # Iterate over already validated geoms
                print(f"DEBUG plot_analysis_step: Item {item_idx}, Plotting valid geom {geom_plot_idx}, type: {type(geom)}")
                plotted_this = False
                if isinstance(geom, LineString):
                    x, y = geom.xy
                    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, solid_capstyle='round', label='_nolegend_', zorder=zorder, alpha=alpha)
                    plotted_this = True
                elif isinstance(geom, Polygon):
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, solid_capstyle='round', label='_nolegend_', zorder=zorder, alpha=alpha)
                    plotted_this = True
                elif isinstance(geom, MultiPoint):
                    # Ensure sub-geoms are Points if we are to scatter them
                    valid_sub_points = [p for p in geom.geoms if isinstance(p, Point) and p.is_valid]
                    if valid_sub_points:
                        xs = [p.x for p in valid_sub_points]; ys = [p.y for p in valid_sub_points]
                        ax.scatter(xs, ys, color=color, marker=marker if marker else 'o', s=ms**2, label='_nolegend_', zorder=zorder, edgecolors='face', linewidths=0.5, alpha=alpha)
                        plotted_count += len(xs)
                        plotted_something = True # Mark that something was plotted for this item
                        # continue # This 'continue' might be problematic if a MultiPoint is the only geom in valid_geoms_in_list and we want a legend entry for it
                elif isinstance(geom, (gpd.GeoSeries, list, tuple)) and all(isinstance(sub_geom, Point) for sub_geom in geom):
                     # This case might be redundant if geoms was already processed into individual points
                     valid_sub_points = [p for p in geom if isinstance(p, Point) and p.is_valid]
                     if valid_sub_points:
                         xs = [p.x for p in valid_sub_points]; ys = [p.y for p in valid_sub_points]
                         ax.scatter(xs, ys, color=color, marker=marker if marker else 'o', s=ms**2, label='_nolegend_', zorder=zorder, edgecolors='face', linewidths=0.5, alpha=alpha)
                         plotted_count += len(xs)
                         plotted_something = True
                         # continue
                elif hasattr(geom, 'geoms'): # Handle GeometryCollection or other Multi-types generally (e.g. MultiLineString, MultiPolygon)
                    for sub_geom_idx, sub_geom in enumerate(geom.geoms):
                         print(f"DEBUG plot_analysis_step: Item {item_idx}, Valid Geom {geom_plot_idx}, Sub-Geom {sub_geom_idx} type: {type(sub_geom)}")
                         if sub_geom and hasattr(sub_geom, 'is_valid') and sub_geom.is_valid and not sub_geom.is_empty:
                              if isinstance(sub_geom, LineString): x, y = sub_geom.xy; ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, solid_capstyle='round', label='_nolegend_', zorder=zorder, alpha=alpha); plotted_this = True
                              elif isinstance(sub_geom, Polygon): x, y = sub_geom.exterior.xy; ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, solid_capstyle='round', label='_nolegend_', zorder=zorder, alpha=alpha); plotted_this = True
                              elif isinstance(sub_geom, Point): ax.scatter([sub_geom.x], [sub_geom.y], color=color, marker=marker if marker else 'o', s=ms**2, label='_nolegend_', zorder=zorder, edgecolors='face', linewidths=0.5, alpha=alpha); plotted_this = True # This would make plotted_count increment incorrectly for legend

                if plotted_this: # If LineString, Polygon, or a sub-geometry was plotted
                    plotted_count += 1 # Increment count for distinct lines/polygons in the item
                    plotted_something = True

            if label != '_nolegend_' and plotted_something: # Only add legend if something was plotted for this item
                 first_plotted_geom_type = type(valid_geoms_in_list[0]) # Get type of first valid geom for legend style
                 
                 # Determine if the legend should be for points or lines/polygons
                 # This check needs to be consistent with how 'is_points' was determined earlier for the whole 'item'
                 # If the item was determined to be points (is_points=True), legend was already added.
                 # This 'else' block is for non-point items.
                 if isinstance(valid_geoms_in_list[0], MultiPoint) or \
                    (isinstance(valid_geoms_in_list[0], (gpd.GeoSeries, list, tuple)) and \
                     all(isinstance(p_leg, Point) for p_leg in valid_geoms_in_list[0] if isinstance(p_leg, Point))):
                     # This condition seems to be for when the item itself IS a collection of points (e.g., a MultiPoint geometry object)
                     # but was not caught by the initial `is_points = True` because it was mixed with other types or was a single MultiPoint.
                     # The `plotted_count` here would be the number of MultiPoint objects if `is_points` was false.
                     # This legend part needs careful review of `plotted_count` meaning.
                     # Let's assume `plotted_count` for non-point items means number of LineStrings/Polygons or distinct Multi-geometries.
                     # For a MultiPoint geometry, the scatter inside this loop handles plotting.
                     # If 'is_points' was True initially, this block is skipped.
                     # If 'is_points' was False, and we are here, valid_geoms_in_list[0] is NOT a Point.
                     # So this specific legend append might be for a case where `item['geoms']` was like `[MultiPoint(...)]`
                     
                     # Simpler: if the item primarily contains points (even within MultiPoints)
                     # it should have been handled by `is_points = True`.
                     # If we are here, it's primarily lines/polygons.
                     legend_elements.append(Line2D([0], [0], color=color, lw=lw, linestyle=ls, label=f'{label} ({plotted_count})', alpha=alpha))
                 else: # Default for LineString, Polygon, MultiLineString, MultiPolygon
                     legend_elements.append(Line2D([0], [0], color=color, lw=lw, linestyle=ls, label=f'{label} ({plotted_count})', alpha=alpha))


    if legend_elements: ax.legend(handles=legend_elements, loc='best', fontsize='small')
    plt.tight_layout()

    if save_plots:
        save_path = Path(output_dir) / f"{filename_prefix}_analysis_step{step_num}.png"
        plt.savefig(save_path, dpi=plot_dpi)
        print(f"  Plot saved to: {save_path}")
    if show_plots:
        plt.show()
    print(f"Plot {step_num} generation complete.")
    plt.close(fig)

# =============================================================================
# Helper Plotting Functions for Texture Generation
# =============================================================================
def plot_geometries(geoms, crs, title, output_dir, filename_base,
                    raster_bounds_tuple=None, highlight_geom=None,
                    show_plots=True, save_plots=True, plot_dpi=150):
    if not show_plots and not save_plots:
        return
    try:
        if not isinstance(geoms, list): geoms = [geoms]
        try: plot_crs = pyproj.CRS.from_user_input(crs).to_wkt()
        except: plot_crs = crs if isinstance(crs, str) else crs.srs
        gdf = gpd.GeoDataFrame({'geometry': geoms}, crs=plot_crs)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        gdf.plot(ax=ax, facecolor='lightblue', edgecolor='blue', alpha=0.6)
        if highlight_geom:
             highlight_list = highlight_geom if isinstance(highlight_geom, list) else [highlight_geom]
             gdf_highlight = gpd.GeoDataFrame({'geometry': highlight_list}, crs=plot_crs)
             gdf_highlight.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
        if raster_bounds_tuple:
            minx, miny, maxx, maxy = raster_bounds_tuple
            gdf_bounds = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs=plot_crs)
            gdf_bounds.plot(ax=ax, facecolor='none', edgecolor='gray', linestyle='--', label='Image Bounds')
            ax.legend()
        total_bounds = gdf.total_bounds
        if total_bounds is not None and len(total_bounds) == 4 and gdf.is_valid.all() and not gdf.is_empty.all():
             minx_g, miny_g, maxx_g, maxy_g = total_bounds
             x_pad = (maxx_g - minx_g) * 0.1 if (maxx_g - minx_g) > 0 else 10
             y_pad = (maxy_g - miny_g) * 0.1 if (maxy_g - miny_g) > 0 else 10
             ax.set_xlim(minx_g - x_pad, maxx_g + x_pad); ax.set_ylim(miny_g - y_pad, maxy_g + y_pad)
        else: print("  Warning: Could not determine valid bounds for plot limits.")
        ax.set_title(title)
        xlabel = f"X Coordinate ({plot_crs})" if isinstance(plot_crs, str) and len(plot_crs) < 50 else "X Coordinate"
        ylabel = f"Y Coordinate ({plot_crs})" if isinstance(plot_crs, str) and len(plot_crs) < 50 else "Y Coordinate"
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        if save_plots:
            save_path = Path(output_dir) / f"{filename_base}.png"
            plt.savefig(save_path, dpi=plot_dpi); print(f"  Plot saved to: {save_path}")
        if show_plots: plt.show()
        plt.close(fig)
    except Exception as e: print(f"Warning: Could not generate plot '{title}': {e}")

def plot_image_array(image_array, transform, crs, title, output_dir, filename_base,
                     show_plots=True, save_plots=True, plot_dpi=150):
    if not show_plots and not save_plots: return
    try:
        try: plot_crs_str = pyproj.CRS.from_user_input(crs).srs if isinstance(crs, (pyproj.CRS, str)) else "Unknown CRS"
        except: plot_crs_str = str(crs) if isinstance(crs, str) else "Invalid CRS"
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        rio_show(image_array, ax=ax, transform=transform, title=title)
        ax.set_xlabel(f"X Coordinate ({plot_crs_str})"); ax.set_ylabel(f"Y Coordinate ({plot_crs_str})")
        if save_plots:
            save_path = Path(output_dir) / f"{filename_base}.png"
            plt.savefig(save_path, dpi=plot_dpi); print(f"  Plot saved to: {save_path}")
        if show_plots: plt.show()
        plt.close(fig)
    except Exception as e: print(f"Warning: Could not generate plot '{title}': {e}")

# =============================================================================
# Helper Plotting Function for Alpha Shape
# =============================================================================
def plot_alpha_shape_result(original_points, alpha_shape_polygon, actual_alpha_display_val, # << Changed param name
                            target_crs_str, output_dir, filename_base,
                            show_plots=True, save_plots=True, plot_dpi=150):
    print("\nPlotting Alpha Shape...")
    fig, ax = plt.subplots(figsize=(12, 10))

    if original_points:
        x_coords = [pt.x for pt in original_points]
        y_coords = [pt.y for pt in original_points]
        ax.scatter(x_coords, y_coords, c='k', s=5, label=f'Input Points ({len(original_points)})', zorder=1)

    if alpha_shape_polygon and not alpha_shape_polygon.is_empty:
        label_alpha_val = str(actual_alpha_display_val)
        try:
            print("  Attempting to plot with descartes.PolygonPatch...")
            ax.add_patch(PolygonPatch(alpha_shape_polygon, fill=False, ec='green', linewidth=1.5, label=f'Alpha Shape (alpha≈{label_alpha_val})', zorder=2))
            if isinstance(alpha_shape_polygon, Polygon):
                hull_pts_exterior = alpha_shape_polygon.exterior.coords.xy
                ax.scatter(hull_pts_exterior[0], hull_pts_exterior[1], color='red', s=15, label='Alpha Shape Vertices', zorder=3)
                ax.add_patch(PolygonPatch(alpha_shape_polygon, fill=False, ec='green', linewidth=1.5, label=f'Alpha Shape (alpha: {label_alpha_val})', zorder=2))
            print("  Successfully plotted with descartes.PolygonPatch.")
        except IndexError: # Fallback for complex geometries
            print("  IndexError with PolygonPatch. Attempting Matplotlib PathPatch (outline only)...")
            geoms_to_plot = [alpha_shape_polygon] if isinstance(alpha_shape_polygon, Polygon) else list(alpha_shape_polygon.geoms)
            plotted_fallback = False
            for idx, geom in enumerate(geoms_to_plot):
                if isinstance(geom, Polygon) and geom.exterior:
                    try:
                        ext_coords = np.array(geom.exterior.coords)
                        if len(ext_coords) < 3: continue
                        codes = [MplPath.MOVETO] + [MplPath.LINETO]*(len(ext_coords)-2) + [MplPath.CLOSEPOLY]
                        path = MplPath(ext_coords, codes)
                        patch = PathPatch(path, facecolor='none', edgecolor='purple', linewidth=1.5, label=f'Alpha Shape Fallback {idx}' if idx > 0 else f'Alpha Shape Fallback (alpha≈{label_alpha_val})')
                        ax.add_patch(patch); plotted_fallback = True
                        for interior in geom.interiors:
                            int_coords = np.array(interior.coords)
                            if len(int_coords) < 3: continue
                            codes_int = [MplPath.MOVETO] + [MplPath.LINETO]*(len(int_coords)-2) + [MplPath.CLOSEPOLY]
                            path_int = MplPath(int_coords, codes_int)
                            ax.add_patch(PathPatch(path_int, facecolor='none', edgecolor='orange', linewidth=1.0))
                    except Exception as fallback_err: print(f"    Error during fallback plotting: {fallback_err}")
            if plotted_fallback: print("  Successfully plotted with Matplotlib PathPatch (outline).")
            else: print("  Fallback plotting also failed or no valid polygons found.")
        except Exception as e_plot: print(f"  An unexpected error occurred during plotting: {e_plot}")

    ax.set_title('Input Points and Calculated Alpha Shape')
    ax.set_xlabel(f'X Coordinate (Assumed CRS: {target_crs_str})')
    ax.set_ylabel(f'Y Coordinate (Assumed CRS: {target_crs_str})')
    ax.set_aspect('equal', adjustable='box')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_plots:
        save_path = Path(output_dir) / f"{filename_base}.png"
        plt.savefig(save_path, dpi=plot_dpi)
        print(f"  Plot saved to: {save_path}")
    if show_plots:
        plt.show()
    print("Alpha shape plot generation complete.")
    plt.close(fig)