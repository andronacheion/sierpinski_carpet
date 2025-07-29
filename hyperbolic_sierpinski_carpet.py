#!/usr/bin/env python3
"""
Hyperbolic Sierpinski Carpet Generator in the Poincaré Disk
Draws requested generation(s) of a concave-edged Sierpinski Carpet adapted to the Poincaré model.

"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from math import log

# ==== INTERACTIVE INPUTS WITH DEFAULTS ====
def get_input(prompt, default, cast_func=str):
    user_input = input(f"{prompt} [default: {default}]: ").strip()
    return cast_func(user_input) if user_input else default

line_width   = get_input("Boundary line width (e.g., 2.0)", 2.0, float)
max_gen      = get_input("Enter maximum generations", 4, int)
arc_strength = get_input("Arc curvature strength (0.0–0.5)", 0.05, float)
color_mode   = get_input("Color mode ('binary' or 'generation')", "binary", str).lower()
save_format  = get_input("Save format ('png' or 'svg')", "png", str).lower()
save_all     = get_input("Save all generations or only final? ('all'/'final')", "all", str).lower()

# Resolution in pixels (square)
resolution_px = get_input("Output resolution (256, 512, 1024, 2048, 4096)", 1024, int)
figsize_inch = 6  # constant
dpi = resolution_px // figsize_inch

# Prepare output folder
folder = f"hyperbolic_sierpinski_inverse_{save_format}"
os.makedirs(folder, exist_ok=True)

# ==== ARC CALCULATION ====
def arc_points(p1, p2, inward=False, strength=0.05, n=100):
    # inward=False => arc bulges outward
    p1, p2 = np.array(p1), np.array(p2)
    mid = (p1 + p2) / 2
    v = p2 - p1
    perp = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    control = mid + (-1 if inward else 1) * strength * perp
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t)**2 * p1 + 2*(1 - t)*t*control + t**2 * p2

# ==== DRAWING FUNCTIONS ====
def draw_concave_square(ax, center, size, strength, color='black'):
    cx, cy = center
    half = size / 2
    corners = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half)
    ]
    pts = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        # inward=False => inverted curvature
        arc = arc_points(p1, p2, inward=False, strength=strength, n=100)
        pts.append(arc)
    poly = np.vstack(pts)
    ax.fill(poly[:, 0], poly[:, 1], color=color)

def sierpinski_poincare(ax, center, size, depth, strength, level=0):
    if depth == 0:
        if color_mode == 'generation':
            col = plt.cm.viridis(level / max_gen)
        else:
            col = 'black'
        draw_concave_square(ax, center, size, strength, color=col)
    else:
        new_size = size / 3
        offsets = [-new_size, 0, new_size]
        for dx in offsets:
            for dy in offsets:
                if dx == 0 and dy == 0:
                    continue
                sierpinski_poincare(ax,
                                    (center[0] + dx, center[1] + dy),
                                    new_size, depth - 1, strength, level + 1)

# ==== GENERATIONS TO RENDER ====
generations = list(range(0, max_gen + 1)) if save_all == 'all' else [max_gen, 'final']

# ==== MAIN LOOP ====
for g in generations:
    fig, ax = plt.subplots(figsize=(figsize_inch, figsize_inch), dpi=dpi)
    ax.set_aspect('equal')
    ax.axis('off')

    # draw Poincaré boundary (unit circle)
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=line_width)
    ax.add_artist(circle)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    if g == 'final':
        for lvl in range(0, max_gen + 1):
            sierpinski_poincare(ax, (0, 0), 1.2, lvl, arc_strength)
        out_name = f"hyperbolic_inverse_final_all.{save_format}"
    else:
        sierpinski_poincare(ax, (0, 0), 1.2, g, arc_strength)
        out_name = f"hyperbolic_inverse_gen_{g}.{save_format}"

    out_path = os.path.join(folder, out_name)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved {'combined final' if g=='final' else 'generation '+str(g)} to {out_path}")

# ==== FRACTAL DIMENSION ====
formula = "log(8)/log(3)"
value = log(8) / log(3)
fd_txt = os.path.join(folder, "fractal_dimension.txt")
with open(fd_txt, 'w') as f:
    f.write(f"Fractal Dimension (D) = {formula} ≈ {value:.4f}\n")
print(f"FD saved to {fd_txt}")

# ==== ZIP FOLDER ====
shutil.make_archive(folder, 'zip', folder)
print(f"Archive created: {folder}.zip")
