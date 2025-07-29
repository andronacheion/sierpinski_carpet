#!/usr/bin/env python3
"""
Interactive Spherical Sierpiński Carpet – positive curvature (“bulging”)
"""

import os
import shutil
from math import log, cos, sin, pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def ask(prompt: str, default, cast=str):
    ans = input(f"{prompt} [default: {default}]: ").strip()
    return cast(ans) if ans else default

max_depth      = ask("Maximum recursion depth", 4, int)
radius         = ask("Sphere radius", 1.0, float)
color_mode     = ask("Color mode ('binary'/'generation')", "binary", str).lower()
show_sphere    = ask("Overlay wireframe sphere? ('yes'/'no')", "yes", str).lower() in {"y","yes","1"}
mirror_equator = ask("Mirror across equator? ('yes'/'no')", "no", str).lower() in {"y","yes","1"}
save_fmt       = ask("Save format ('png'/'svg')", "png", str).lower()
save_all       = ask("Save all gens or only final? ('all'/'final')", "all", str).lower()
resolution     = ask("Output resolution px (256/512/1024/2048/4096)", 1024, int)
arc_samples    = ask("Arc samples per edge (>=1)", 20, int)
proj_type      = ask("Projection ('ortho'/'persp')", "ortho", str).lower()
manual_view    = ask("Manual camera? ('yes' to set elev/azim)", "no", str).lower() in {"y","yes","1"}
view_elev      = ask("Camera elev (deg)", 10.0, float) if manual_view else None
view_azim      = ask("Camera azim (deg)", 20.0, float) if manual_view else None

FIGSIZE = 6
DPI     = resolution // FIGSIZE
OUT_DIR = f"spherical_sierpinski_{save_fmt}"
os.makedirs(OUT_DIR, exist_ok=True)

def to_sphere(x: float, y: float, R: float = 1.0):
    d = x*x + y*y + 1.0
    return np.array([2*x/d, 2*y/d, (x*x + y*y - 1)/d]) * R

def slerp(p: np.ndarray, q: np.ndarray, t: float, R: float):
    p_n = p/np.linalg.norm(p)
    q_n = q/np.linalg.norm(q)
    dot = np.clip(np.dot(p_n, q_n), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-9:
        return p_n * R
    return ((np.sin((1-t)*theta)*p_n + np.sin(t*theta)*q_n) / np.sin(theta)) * R

def rotate_point(x, y, theta):
    return x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta)

def spherical_polygon(ax, boundary_pts, facecolor="black", zorder=6):
    c = np.mean(boundary_pts, axis=0)
    c = c / np.linalg.norm(c) * radius
    tris = []
    n = len(boundary_pts)
    for i in range(n):
        tris.append([c, boundary_pts[i], boundary_pts[(i+1) % n]])
    poly = Poly3DCollection(tris, zorder=zorder)
    poly.set_facecolor(facecolor)
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

FIRST_BOUNDARY = None

def curved_square(ax, x0, y0, size, R, facecolor="black", samples=8, zorder=6, remember=False, mirror=False, rotation=pi/4):
    global FIRST_BOUNDARY
    c2d = [rotate_point(x, y, rotation) for (x, y) in [
        (x0, y0), (x0+size, y0), (x0+size, y0+size), (x0, y0+size)
    ]]
    c3d = [to_sphere(x, y, R) for (x, y) in c2d]
    boundary = []
    for i in range(4):
        p, q = c3d[i], c3d[(i+1)%4]
        ts = np.linspace(0.0, 1.0, samples+2)
        pts = [slerp(p, q, t, R) for t in ts]
        if i: pts = pts[1:]
        boundary.extend(pts)
    if remember and FIRST_BOUNDARY is None:
        FIRST_BOUNDARY = np.array(boundary)
    spherical_polygon(ax, boundary, facecolor=facecolor, zorder=zorder)
    if mirror:
        boundary_m = [np.array([p[0], p[1], -p[2]]) for p in boundary]
        spherical_polygon(ax, boundary_m, facecolor=facecolor, zorder=zorder)

def sierpinski(ax, x0, y0, size, depth, level, rotation=pi/4):
    if depth == 0:
        face = plt.cm.viridis(level/max_depth) if color_mode == "generation" else "black"
        samples = max(2, arc_samples // (2**max(level-1, 0)))
        curved_square(ax, x0, y0, size, radius, facecolor=face, samples=samples,
                      zorder=10-level, remember=(level==0), mirror=mirror_equator, rotation=rotation)
    else:
        new = size/3.0
        for dx in range(3):
            for dy in range(3):
                if dx == 1 and dy == 1:
                    continue
                sierpinski(ax, x0 + dx*new, y0 + dy*new, new, depth-1, level+1, rotation=rotation)

# --------- INITIAL PARAMETERS -----------
initial_size   = 0.95
initial_offset = -initial_size/2
gens = list(range(max_depth+1)) if save_all == "all" else [max_depth]

# ---------- MAIN LOOP ----------
for g in gens:
    fig = plt.figure(figsize=(FIGSIZE, FIGSIZE), dpi=DPI)
    ax  = fig.add_subplot(111, projection="3d")
    # No spherical grid, only the equator!
    if show_sphere:
        eq_u = np.linspace(0, 2*np.pi, 720)
        ax.plot3D(radius*np.cos(eq_u), radius*np.sin(eq_u), 0*eq_u,
                  color='black', linewidth=1.2, zorder=12)

    sierpinski(ax, initial_offset, initial_offset, initial_size, g, 0, rotation=pi/4)
    if manual_view:
        ax.view_init(elev=view_elev, azim=view_azim)
    else:
        if FIRST_BOUNDARY is not None:
            c = FIRST_BOUNDARY.mean(axis=0)
            c = c/np.linalg.norm(c)
            elev_auto = np.degrees(np.arcsin(c[2]))
            azim_auto = np.degrees(np.arctan2(c[1], c[0]))
            ax.view_init(elev=elev_auto, azim=azim_auto)
        else:
            ax.view_init(elev=10, azim=20)
    ax.set_proj_type(proj_type)
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-radius, radius]); ax.set_ylim([-radius, radius]); ax.set_zlim([-radius, radius])
    ax.set_axis_off()
    fig.patch.set_facecolor('white')       # White background for figure
    ax.set_facecolor('white')              # White background for axes
    plt.tight_layout(pad=0)
    fname = f"spherical_carpet_gen_{g}.{save_fmt}"
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight", pad_inches=0, transparent=False)
    plt.close(fig)
    print(f"Saved generation {g} -> {fname}")

with open(os.path.join(OUT_DIR, "fractal_dimension.txt"), "w") as fh:
    fh.write(f"D = log(8)/log(3) ≈ {log(8)/log(3):.5f}\n")
shutil.make_archive(OUT_DIR, "zip", OUT_DIR)
print(f"✔ Totul gata. Fișierele sunt în '{OUT_DIR}', arhivă '{OUT_DIR}.zip'.")
