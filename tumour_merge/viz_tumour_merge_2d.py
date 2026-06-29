"""
2D preview matching the current FBNO-style 3D tumour-merge scenario.

The panels use the same sample, panel times, geometry-A radius colour scale, and
pre/post-merge display logic as viz_tumour_merge_3d.py.  Pre-merge panels show
the two FBNO geometry-A tumour contours.  Merge/post-merge panels show the
stored FROST level-set union with the same radius-value mapping used in 3D.

Run: python viz_tumour_merge_2d.py -> tumour_merge_preview.png
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-frost")
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib import cm
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import viz_tumour_merge_3d as viz3d


OUT_PATH = os.path.join(HERE, "tumour_merge_preview.png")


def panel_title(d, t):
    title = "t=0" if t == 0 else f"t=0 to {t}"
    return f"{title}  ({int(d['n_components'][t])} comp.)"


def set_square_limits(ax, point_sets):
    xy = np.vstack([np.asarray(points) for points in point_sets if len(points)])
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.62 * max(xmax - xmin, ymax - ymin)
    half = max(half, 0.12)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def draw_valued_boundary(ax, points, values):
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    mean_value = float(np.mean(values))
    face_color = viz3d.FBNO_CMAP(viz3d.FBNO_NORM(mean_value))
    ax.add_patch(Polygon(points, closed=True, facecolor=face_color, edgecolor="none", alpha=0.78))

    closed = np.vstack([points, points[0]])
    closed_values = np.concatenate([values, values[:1]])
    segments = np.stack([closed[:-1], closed[1:]], axis=1)
    segment_values = 0.5 * (closed_values[:-1] + closed_values[1:])
    ax.add_collection(LineCollection(segments, colors="black", linewidths=3.0, alpha=0.95))
    lc = LineCollection(
        segments,
        cmap=viz3d.FBNO_CMAP,
        norm=viz3d.FBNO_NORM,
        linewidths=1.8,
        alpha=1.0,
    )
    lc.set_array(segment_values)
    ax.add_collection(lc)
    return lc


def draw_source_panel(ax, d, t, fixed_initial_separation=False):
    source_t = viz3d.T0_PANEL_END if fixed_initial_separation else t
    components = viz3d.source_surface_sequences(
        d,
        np.array([source_t]),
        fixed_initial_separation=fixed_initial_separation,
    )
    point_sets = []
    last_artist = None
    for sections, values in components:
        points = sections[0]
        vals = values[0]
        point_sets.append(points)
        last_artist = draw_valued_boundary(ax, points, vals)
    set_square_limits(ax, point_sets)
    return last_artist


def merged_radius_field(d, t):
    n = d["mask"].shape[-1]
    xs = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    field = np.full((n, n), np.nan, dtype=float)
    mask = d["mask"][t]
    x = xx[mask]
    y = yy[mask]
    t_values = np.full(len(x), float(t))
    scale = float(d["params"][2])
    sep = viz3d.separation_at_time(d, t_values)
    centres = viz3d.interpolate_centres(t_values)

    left_local = np.column_stack([x + sep, y])
    right_local = np.column_stack([x - sep, y])
    left_raw = left_local / scale + centres
    right_raw = right_local / scale + centres
    left_radius = np.sqrt(np.sum(left_raw ** 2, axis=1))
    right_radius = np.sqrt(np.sum(right_raw ** 2, axis=1))
    left_dist = np.sum(left_local ** 2, axis=1)
    right_dist = np.sum(right_local ** 2, axis=1)

    # Use a soft nearest-source blend for the flat filled field.  The 3D surface
    # can hide the left/right source switch; in 2D a hard switch draws an
    # artificial vertical seam through the merged tumour.
    temperature = 0.035
    logits = np.column_stack([-left_dist / temperature, -right_dist / temperature])
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    field[mask] = weights[:, 0] * left_radius + weights[:, 1] * right_radius
    return xs, field


def draw_merged_panel(ax, d, t):
    xs, field = merged_radius_field(d, t)
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=(-1, 1, -1, 1),
        cmap=viz3d.CMAP,
        norm=viz3d.FBNO_NORM,
        interpolation="bilinear",
    )
    ax.contour(xs, xs, d["phi"][t].T, levels=[0], colors="black", linewidths=3.0)
    ax.contour(xs, xs, d["phi"][t].T, levels=[0], colors="crimson", linewidths=1.5)
    point_sets = [np.asarray(comp) for comp in d["boundary_components"][t]]
    set_square_limits(ax, point_sets)
    return im


def draw_panel(ax, d, t):
    merge_step = int(d["merge_step"])
    if t == 0:
        artist = draw_source_panel(ax, d, t, fixed_initial_separation=True)
    elif t < merge_step:
        artist = draw_source_panel(ax, d, t)
    else:
        artist = draw_merged_panel(ax, d, t)

    ax.set_title(panel_title(d, t), fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    return artist


def main():
    data = np.load(viz3d.DATA_PATH, allow_pickle=True)
    d = viz3d.load_geometry_a_sample(data)
    times = viz3d.selected_times(d)

    fig, axs = plt.subplots(1, len(times), figsize=(17.5, 3.9))
    artist = None
    for ax, t in zip(axs, times):
        artist = draw_panel(ax, d, t)

    sm = cm.ScalarMappable(norm=viz3d.FBNO_NORM, cmap=viz3d.FBNO_CMAP)
    sm.set_array([])
    fig.subplots_adjust(left=0.02, right=0.91, bottom=0.08, top=0.78, wspace=0.08)
    cax = fig.add_axes([0.935, 0.18, 0.012, 0.54])
    cbar = fig.colorbar(sm, cax=cax, format="%.2f")
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks(np.linspace(viz3d.FBNO_GEOM_A_VMIN, viz3d.FBNO_GEOM_A_VMAX, 4))
    cbar.set_label("FBNO geometry-A radius value", fontsize=9)
    fig.suptitle("Geometry A two-tumour merge - 2D counterpart of the FBNO-style 3D scenario",
                 fontsize=12)
    fig.savefig(OUT_PATH, dpi=140)
    plt.close(fig)
    print("geometry A sample merge@t=", int(d["merge_step"]), "panels=", times)
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
