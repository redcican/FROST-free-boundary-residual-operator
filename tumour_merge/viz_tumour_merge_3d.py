"""
Geometry-A two-tumour merge in the FBNO panel-a 3D style.

The flat preview shows selected times in a 1x5 row.  This figure keeps that 1x5 progression, but
each panel is a cumulative space-time free-boundary surface in the same style as
FBNO/reproduce_tumour/result/panel_a_3d/panel_a_3d_combined.png: z is time, the boundary is stacked
vertically, the t=0 caps are red, and the surface is coloured with YlGnBu.

Run: python viz_tumour_merge_3d.py -> tumour_merge_3d.png
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-frost")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from skimage.measure import marching_cubes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "tumour_merge.npy")
OUT_PATH = os.path.join(HERE, "tumour_merge_3d.png")
FBNO_GEOM_A = os.path.join(HERE, "..", "..", "FBNO", "data", "Tumour_data", "domain_test.npy")
Z_MAX = 100.0
NUM_NEW_LAYERS = 50
MARCHING_CUBES_STEP = 1
CMAP = "YlGnBu"
FBNO_ELEV = 25
FBNO_AZIM = 45
FBNO_GEOM_A_VMIN = 0.006916818361336667
FBNO_GEOM_A_VMAX = 2.7956595167997826
FBNO_NORM = Normalize(FBNO_GEOM_A_VMIN, FBNO_GEOM_A_VMAX)
FBNO_CMAP = plt.get_cmap(CMAP)
T0_VISUAL_END = 2
T0_PANEL_END = 3
PAIR_FULL_SEPARATION = 2.55
PAIR_PREMERGE_SEP0 = 2.35
PAIR_PREMERGE_SEP1 = 1.35


def load_source_contours():
    data = np.load(FBNO_GEOM_A, allow_pickle=True)
    seen = {}
    for item in data:
        key = tuple(np.asarray(item["initial_domain"]).ravel().tolist())
        if key not in seen:
            seen[key] = np.asarray(item["processed_contours"])[:, :, :2].astype(float)
        if len(seen) == 3:
            break
    return list(seen.values())


SOURCE_CONTOURS = load_source_contours()
SOURCE_VALUES = [np.sqrt(np.sum(contours ** 2, axis=2)) for contours in SOURCE_CONTOURS]


def geometry_a_centres():
    return SOURCE_CONTOURS[0].mean(axis=1)


GEOM_A_CENTRES = geometry_a_centres()


def load_geometry_a_sample(data):
    candidates = [
        i for i, d in enumerate(data)
        if int(d["params"][0]) == 0
        and int(d["params"][1]) == 0
        and 0 < int(d["merge_step"]) < d["phi"].shape[0] - 1
    ]
    if not candidates:
        raise RuntimeError("No valid geometry-A A+A merge sample found in tumour_merge.npy")
    return data[min(candidates, key=lambda i: abs(int(data[i]["merge_step"]) - 10))]


def selected_times(d):
    tmax = d["phi"].shape[0] - 1
    ms = int(d["merge_step"])
    return [0, max(0, ms - 2), ms, min(tmax, ms + 2), tmax]


def upsample_to_time(vol, t_end):
    zt = np.linspace(0, t_end, NUM_NEW_LAYERS)
    lo = np.floor(zt).astype(int)
    hi = np.clip(lo + 1, 0, t_end)
    w = (zt - lo)[:, None, None]
    return (1.0 - w) * vol[lo] + w * vol[hi]


def interpolate_centres(t_values):
    idx = np.arange(len(GEOM_A_CENTRES))
    return np.column_stack([
        np.interp(t_values, idx, GEOM_A_CENTRES[:, 0]),
        np.interp(t_values, idx, GEOM_A_CENTRES[:, 1]),
    ])


def separation_at_time(d, t_values):
    sep0, sep1 = float(d["params"][3]), float(d["params"][4])
    tau = t_values / (d["phi"].shape[0] - 1)
    smooth = tau * tau * (3.0 - 2.0 * tau)
    return sep0 + (sep1 - sep0) * smooth


def rot_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def centered_sequence(source_index):
    contours = SOURCE_CONTOURS[source_index]
    return contours - contours.mean(axis=1, keepdims=True)


def transform_sequence(contours, scale, rotation, centres):
    return contours @ rot_matrix(rotation).T * scale + np.asarray(centres)[:, None, :]


def source_surface_sequences(d, times, fixed_initial_separation=False):
    g_a, g_b = int(d["params"][0]), int(d["params"][1])
    times = np.asarray(times, dtype=int)
    if fixed_initial_separation:
        sep = np.full(len(times), PAIR_FULL_SEPARATION)
    else:
        tau = times.astype(float) / max(1, int(d["merge_step"]) - 1)
        tau = np.clip(tau, 0.0, 1.0)
        smooth = tau * tau * (3.0 - 2.0 * tau)
        sep = PAIR_PREMERGE_SEP0 + (PAIR_PREMERGE_SEP1 - PAIR_PREMERGE_SEP0) * smooth
    centres_a = np.column_stack([-sep, np.zeros(len(times))])
    centres_b = np.column_stack([+sep, np.zeros(len(times))])
    seq_a = centered_sequence(g_a)[times] + centres_a[:, None, :]
    seq_b = centered_sequence(g_b)[times] + centres_b[:, None, :]
    val_a = SOURCE_VALUES[g_a][times]
    val_b = SOURCE_VALUES[g_b][times]
    return (seq_a, val_a), (seq_b, val_b)


def fbno_geometry_a_radius_values(xyz, t_values, d):
    """Map FROST-scaled merge coordinates back to FBNO geometry-A units for colouring."""
    scale = float(d["params"][2])
    sep = separation_at_time(d, t_values)
    left_local = np.column_stack([xyz[:, 0] + sep, xyz[:, 1]])
    right_local = np.column_stack([xyz[:, 0] - sep, xyz[:, 1]])
    use_left = np.sum(left_local ** 2, axis=1) <= np.sum(right_local ** 2, axis=1)
    local = np.where(use_left[:, None], left_local, right_local)
    raw = local / scale + interpolate_centres(t_values)
    return np.sqrt(np.sum(raw ** 2, axis=1))


def add_initial_caps(ax, d):
    for comp in d["boundary_components"][0]:
        comp = np.asarray(comp)
        if comp.ndim != 2 or len(comp) < 3:
            continue
        verts = [np.column_stack([comp[:, 0], comp[:, 1], np.zeros(len(comp))])]
        cap = Poly3DCollection(verts, alpha=0.50, facecolor="red", edgecolor="red", linewidths=0.4)
        ax.add_collection3d(cap)


def best_aligned_contour(contour, values, reference):
    options = [(contour, values)]
    if values is None:
        options.append((contour[::-1], None))
    else:
        options.append((contour[::-1], values[::-1]))
    best = None
    best_score = np.inf
    for pts, vals in options:
        for shift in range(len(pts)):
            rolled = np.roll(pts, shift, axis=0)
            score = float(np.mean(np.sum((rolled - reference) ** 2, axis=1)))
            if score < best_score:
                best_score = score
                best = (rolled, None if vals is None else np.roll(vals, shift))
    return best


def align_sections(sections, values=None):
    sections = [np.asarray(section, dtype=float) for section in sections]
    values = None if values is None else [np.asarray(v, dtype=float) for v in values]
    aligned_sections = [sections[0]]
    aligned_values = None if values is None else [values[0]]
    for i in range(1, len(sections)):
        section, vals = best_aligned_contour(
            sections[i],
            None if values is None else values[i],
            aligned_sections[-1],
        )
        aligned_sections.append(section)
        if aligned_values is not None:
            aligned_values.append(vals)
    return np.asarray(aligned_sections), None if aligned_values is None else np.asarray(aligned_values)


def add_red_cap(ax, section, z):
    verts = [np.column_stack([section[:, 0], section[:, 1], np.full(len(section), z)])]
    cap = Poly3DCollection(verts, alpha=0.50, facecolor="red", edgecolor="red", linewidths=0.4)
    ax.add_collection3d(cap)


def draw_contour_stack_surface(ax, sections, z_values, value_sections=None, t_values=None,
                               d=None, add_bottom_cap=True):
    sections = [np.asarray(section, dtype=float) for section in sections]
    z_values = np.asarray(z_values, dtype=float)
    if len(sections) == 0:
        return None
    if len(sections) == 1:
        dz = 0.35 * Z_MAX / 14.0
        z = float(z_values[0])
        z0, z1 = (max(0.0, z - dz), z) if z > 0.0 else (0.0, dz)
        sections = [sections[0], sections[0].copy()]
        z_values = np.array([z0, z1])
        if value_sections is not None:
            value_sections = [value_sections[0], value_sections[0]]
        if t_values is not None:
            t_values = [t_values[0], t_values[0]]

    sections, value_sections = align_sections(sections, value_sections)
    n_layers, npts = sections.shape[:2]
    kind = "cubic" if n_layers >= 4 else "linear"
    target_z = np.linspace(z_values.min(), z_values.max(), max(NUM_NEW_LAYERS, n_layers * 5))
    interp_sec = np.zeros((len(target_z), npts, 2))
    interp_val = None if value_sections is None else np.zeros((len(target_z), npts))
    for point_i in range(npts):
        ix = interp1d(z_values, sections[:, point_i, 0], kind=kind, fill_value="extrapolate")
        iy = interp1d(z_values, sections[:, point_i, 1], kind=kind, fill_value="extrapolate")
        interp_sec[:, point_i, 0] = ix(target_z)
        interp_sec[:, point_i, 1] = iy(target_z)
        if value_sections is not None:
            iv = interp1d(z_values, value_sections[:, point_i], kind=kind, fill_value="extrapolate")
            interp_val[:, point_i] = iv(target_z)

    all_points = np.array([
        [interp_sec[layer_i, point_i, 0], interp_sec[layer_i, point_i, 1], target_z[layer_i]]
        for layer_i in range(len(target_z))
        for point_i in range(npts)
    ])
    triangles = []
    for layer_i in range(len(target_z) - 1):
        for point_i in range(npts):
            next_i = (point_i + 1) % npts
            a = layer_i * npts + point_i
            b = layer_i * npts + next_i
            c = (layer_i + 1) * npts + point_i
            e = (layer_i + 1) * npts + next_i
            triangles.append([a, b, c])
            triangles.append([c, b, e])
    triangles = np.asarray(triangles)

    if interp_val is None:
        if d is None or t_values is None:
            raise ValueError("d and t_values are required when value_sections is omitted")
        layer_t = np.interp(target_z, z_values, np.asarray(t_values, dtype=float))
        all_t = np.repeat(layer_t, npts)
        vertex_value = fbno_geometry_a_radius_values(all_points, all_t, d)
    else:
        vertex_value = interp_val.reshape(-1)
    face_field = vertex_value[triangles].mean(axis=1)

    surf = ax.plot_trisurf(
        all_points[:, 0], all_points[:, 1], all_points[:, 2],
        triangles=triangles,
        alpha=0.7,
        linewidth=0.05,
        antialiased=True,
    )
    surf.set_array(face_field)
    surf.set_cmap(CMAP)
    surf.set_clim(FBNO_GEOM_A_VMIN, FBNO_GEOM_A_VMAX)

    stride = max(1, npts // 36)
    for point_i in range(0, npts, stride):
        ax.plot(interp_sec[:, point_i, 0], interp_sec[:, point_i, 1], target_z,
                color="black", linewidth=0.05, alpha=0.4)
    for section, z in zip(sections, z_values):
        closed = np.vstack([section, section[0]])
        ax.plot(closed[:, 0], closed[:, 1], np.full(len(closed), z),
                color="black", linewidth=0.05, alpha=0.4)
    if add_bottom_cap:
        add_red_cap(ax, sections[0], z_values[0])
    return surf


def draw_fbno_surface(ax, original_sections, original_values, original_z=None,
                      num_longitudinal_lines=36, do_roll=False):
    """Same contour-stack renderer used by FBNO/reproduce_tumour/reproduce_panel_a_3d.py."""
    original_sections = np.asarray(original_sections, dtype=float).copy()
    original_values = np.asarray(original_values, dtype=float)
    if original_z is None:
        original_z = np.linspace(0, Z_MAX, original_sections.shape[0])
    else:
        original_z = np.asarray(original_z, dtype=float)
    if do_roll:
        original_sections[0] = np.roll(original_sections[0], -150, 0)

    target_z = np.linspace(original_z.min(), original_z.max(), NUM_NEW_LAYERS)
    npts = original_sections.shape[1]
    interp_sec = np.zeros((NUM_NEW_LAYERS, npts, 2))
    interp_val = np.zeros((NUM_NEW_LAYERS, npts))
    for pi in range(npts):
        ix = interp1d(original_z, original_sections[:, pi, 0], kind="cubic", fill_value="extrapolate")
        iy = interp1d(original_z, original_sections[:, pi, 1], kind="cubic", fill_value="extrapolate")
        iv = interp1d(original_z, original_values[:, pi], kind="cubic", fill_value="extrapolate")
        interp_sec[:, pi, 0] = ix(target_z)
        interp_sec[:, pi, 1] = iy(target_z)
        interp_val[:, pi] = iv(target_z)

    all_points = np.array([
        [interp_sec[li, pi, 0], interp_sec[li, pi, 1], target_z[li]]
        for li in range(NUM_NEW_LAYERS)
        for pi in range(npts)
    ])
    triangles = []
    for li in range(NUM_NEW_LAYERS - 1):
        for pi in range(npts):
            npi = (pi + 1) % npts
            a = li * npts + pi
            b = li * npts + npi
            c = (li + 1) * npts + pi
            e = (li + 1) * npts + npi
            triangles.append([a, b, c])
            triangles.append([c, b, e])
    flatv = interp_val.flatten()
    tri_vals = np.array([(flatv[t[0]] + flatv[t[1]] + flatv[t[2]]) / 3 for t in triangles])

    surf = ax.plot_trisurf(
        all_points[:, 0],
        all_points[:, 1],
        all_points[:, 2],
        triangles=triangles,
        alpha=0.7,
        linewidth=0.05,
        antialiased=True,
    )
    surf.set_array(tri_vals)
    surf.set_cmap(CMAP)
    surf.set_clim(FBNO_GEOM_A_VMIN, FBNO_GEOM_A_VMAX)

    for pi in range(0, npts, max(1, npts // num_longitudinal_lines)):
        ax.plot(interp_sec[:, pi, 0], interp_sec[:, pi, 1], target_z,
                "k-", linewidth=0.05, alpha=0.4)
    for i, section in enumerate(original_sections):
        ax.plot(section[:, 0], section[:, 1], original_z[i], linewidth=0.05, color="black")
    verts = [list(zip(
        original_sections[0, :, 0],
        original_sections[0, :, 1],
        np.full_like(original_sections[0, :, 0], original_z[0]),
    ))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, color="red"))
    return surf


def add_source_surfaces(ax, d, times, fixed_initial_separation=False):
    z_values = Z_MAX * np.asarray(times, dtype=float) / (d["phi"].shape[0] - 1)
    surfaces = []
    for sections, values in source_surface_sequences(d, times, fixed_initial_separation):
        surfaces.append(draw_fbno_surface(ax, sections, values, original_z=z_values, do_roll=True))
    return surfaces[-1] if surfaces else None


def add_geometry_a_surface(ax, t_end):
    sections = SOURCE_CONTOURS[0][:t_end + 1]
    values = SOURCE_VALUES[0][:t_end + 1]
    z_values = np.linspace(0.0, Z_MAX * t_end / 14.0, t_end + 1)
    return draw_fbno_surface(ax, sections, values, original_z=z_values, do_roll=True)


def add_merged_surface(ax, d, t_start, t_end):
    sections, z_values, t_values = [], [], []
    for t in range(t_start, t_end + 1):
        comps = d["boundary_components"][t]
        if len(comps) != 1:
            continue
        sections.append(np.asarray(comps[0], dtype=float))
        z_values.append(Z_MAX * t / (d["phi"].shape[0] - 1))
        t_values.append(t)
    return draw_contour_stack_surface(
        ax, sections, z_values, t_values=t_values, d=d, add_bottom_cap=False
    )


def add_levelset_surface(ax, d, t_end):
    phiv = upsample_to_time(d["phi"].astype(float), t_end)
    phiv = gaussian_filter(phiv, sigma=(0.45, 0.60, 0.60))
    verts, faces, _, _ = marching_cubes(phiv, level=0.0, step_size=MARCHING_CUBES_STEP)

    n = d["phi"].shape[1]
    z_end = Z_MAX * t_end / (d["phi"].shape[0] - 1)
    xyz = np.column_stack([
        verts[:, 1] / (n - 1) * 2.0 - 1.0,
        verts[:, 2] / (n - 1) * 2.0 - 1.0,
        verts[:, 0] / (phiv.shape[0] - 1) * z_end,
    ])
    t_values = verts[:, 0] / (phiv.shape[0] - 1) * t_end
    vertex_value = fbno_geometry_a_radius_values(xyz, t_values, d)
    face_field = vertex_value[faces].mean(axis=1)
    surf = ax.plot_trisurf(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        triangles=faces,
        alpha=0.7,
        linewidth=0.05,
        antialiased=True,
    )
    surf.set_array(face_field)
    surf.set_cmap(CMAP)
    surf.set_clim(FBNO_GEOM_A_VMIN, FBNO_GEOM_A_VMAX)
    add_initial_caps(ax, d)
    add_boundary_rings(ax, d, t_end)
    return surf


def add_boundary_rings(ax, d, t_end):
    for t in range(t_end + 1):
        color = "red" if t == 0 else "black"
        lw = 0.05
        alpha = 0.40 if t != 0 else 0.80
        z = Z_MAX * t / (d["phi"].shape[0] - 1)
        for comp in d["boundary_components"][t]:
            comp = np.asarray(comp)
            if comp.ndim != 2 or len(comp) < 3:
                continue
            closed = np.vstack([comp, comp[0]])
            ax.plot(closed[:, 0], closed[:, 1], np.full(len(closed), z),
                    color=color, linewidth=lw, alpha=alpha)


def add_t0_visual_rings(ax, d):
    z_top = Z_MAX * T0_VISUAL_END / (d["phi"].shape[0] - 1)
    for z, color, alpha in [(0.0, "red", 0.85), (z_top, "black", 0.55)]:
        for comp in d["boundary_components"][0]:
            comp = np.asarray(comp)
            if comp.ndim != 2 or len(comp) < 3:
                continue
            closed = np.vstack([comp, comp[0]])
            ax.plot(closed[:, 0], closed[:, 1], np.full(len(closed), z),
                    color=color, linewidth=0.05, alpha=alpha)


def add_t0_complete_bodies(ax, d):
    """Render two complete independent initial tumour bodies instead of a degenerate z=0 slice."""
    z_top = Z_MAX * T0_VISUAL_END / (d["phi"].shape[0] - 1)
    for comp in d["boundary_components"][0]:
        comp = np.asarray(comp)
        if comp.ndim != 2 or len(comp) < 3:
            continue
        n = len(comp)
        bottom = np.column_stack([comp[:, 0], comp[:, 1], np.zeros(n)])
        top = np.column_stack([comp[:, 0], comp[:, 1], np.full(n, z_top)])
        ring_xyz = np.vstack([bottom, top])
        ring_vals = fbno_geometry_a_radius_values(ring_xyz, np.zeros(2 * n), d)

        tris = []
        tri_vals = []
        for i in range(n):
            j = (i + 1) % n
            tris.append([bottom[i], bottom[j], top[i]])
            tri_vals.append((ring_vals[i] + ring_vals[j] + ring_vals[n + i]) / 3.0)
            tris.append([top[i], bottom[j], top[j]])
            tri_vals.append((ring_vals[n + i] + ring_vals[j] + ring_vals[n + j]) / 3.0)

        side = Poly3DCollection(
            tris,
            alpha=0.7,
            linewidths=0.05,
            antialiased=True,
            facecolors=FBNO_CMAP(FBNO_NORM(tri_vals)),
        )
        ax.add_collection3d(side)

        top_val = fbno_geometry_a_radius_values(top, np.zeros(n), d).mean()
        top_cap = Poly3DCollection(
            [top],
            alpha=0.7,
            linewidths=0.05,
            facecolors=[FBNO_CMAP(FBNO_NORM(top_val))],
            antialiased=True,
        )
        ax.add_collection3d(top_cap)

        bottom_cap = Poly3DCollection([bottom], alpha=0.50, facecolor="red", edgecolor="red", linewidths=0.4)
        ax.add_collection3d(bottom_cap)

    add_t0_visual_rings(ax, d)
    sm = cm.ScalarMappable(norm=FBNO_NORM, cmap=FBNO_CMAP)
    sm.set_array([])
    return sm


def add_space_time_surface(ax, d, t_end):
    merge_step = int(d["merge_step"])

    if t_end == 0:
        return add_source_surfaces(
            ax,
            d,
            np.arange(T0_PANEL_END + 1),
            fixed_initial_separation=True,
        )
    if t_end < merge_step:
        return add_source_surfaces(ax, d, np.arange(t_end + 1))
    return add_levelset_surface(ax, d, t_end)


def panel_limits(d, t_end):
    points = []

    if t_end == 0:
        for sections, _ in source_surface_sequences(
            d, np.arange(d["phi"].shape[0]), fixed_initial_separation=True
        ):
            points.append(sections.reshape(-1, 2))
        zmax = Z_MAX
    else:
        merge_step = int(d["merge_step"])
        branch_end = min(t_end, merge_step - 1)
        if branch_end >= 0:
            for sections, _ in source_surface_sequences(d, np.arange(branch_end + 1)):
                points.append(sections.reshape(-1, 2))
        for t in range(merge_step, t_end + 1):
            comps = d["boundary_components"][t]
            if len(comps) != 1:
                continue
            comp = np.asarray(comps[0])
            if comp.ndim == 2 and len(comp) >= 3:
                points.append(comp)
        zmax = Z_MAX * t_end / (d["phi"].shape[0] - 1)

    if not points:
        for comp in d["boundary_components"][0]:
            comp = np.asarray(comp)
            if comp.ndim == 2 and len(comp) >= 3:
                points.append(comp)
    if not points:
        return (-1.0, 1.0), (-1.0, 1.0), (0.0, Z_MAX)

    xy = np.vstack(points)
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.68 * max(xmax - xmin, ymax - ymin)
    half = max(half, 0.18)
    return (cx - half, cx + half), (cy - half, cy + half), (0.0, max(zmax, 1.0))


def apply_fbno_axes(ax, d, t_end):
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=FBNO_ELEV, azim=FBNO_AZIM)
    ax.grid(False)


def draw_panel(fig, ax, d, t_end):
    surf = add_space_time_surface(ax, d, t_end)
    apply_fbno_axes(ax, d, t_end)
    title = "t=0" if t_end == 0 else f"t=0 to {t_end}"
    ax.set_title(f"{title}  ({int(d['n_components'][t_end])} comp.)", fontsize=10, pad=2)
    return surf


def main():
    data = np.load(DATA_PATH, allow_pickle=True)
    d = load_geometry_a_sample(data)
    times = selected_times(d)

    fig = plt.figure(figsize=(21.5, 5.8))
    positions = [
        [0.015, 0.13, 0.170, 0.67],
        [0.195, 0.13, 0.170, 0.67],
        [0.375, 0.13, 0.170, 0.67],
        [0.555, 0.13, 0.170, 0.67],
        [0.735, 0.13, 0.170, 0.67],
    ]
    for col, t_end in enumerate(times):
        ax = fig.add_axes(positions[col], projection="3d")
        draw_panel(fig, ax, d, t_end)
    sm = cm.ScalarMappable(norm=FBNO_NORM, cmap=FBNO_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=fig.add_axes([0.93, 0.20, 0.012, 0.50]), format="%.2f")
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks(np.linspace(FBNO_GEOM_A_VMIN, FBNO_GEOM_A_VMAX, 4))
    cbar.set_label("FBNO geometry-A radius value", fontsize=9)
    fig.suptitle(
        "Geometry A two-tumour merge — cumulative 3D space-time boundary surfaces "
        "(FBNO panel-a style)",
        fontsize=13,
    )
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print("geometry A sample merge@t=", int(d["merge_step"]), "panels=", times)
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
