#!/usr/bin/env python3
"""
Standardized Interactive Sierpinski Carpet Generator
Draws requested generation(s) and combined final of the Sierpinski Carpet.
Supports: line width, max generations, color mode (generation/binary),
save format (png/svg), save all/final, resolution options, FD output.
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from math import log

# ==== INTERACTIVE INPUTS ====
line_width = float(input("Line width (e.g., 1.0): ").strip())
max_gen = int(input("Enter maximum generations (e.g., 5): ").strip())
color_mode = input("Color mode (generation/binary): ").strip().lower()
save_format = input("Save format (png/svg): ").strip().lower()
save_all = input("Save all generations or only final? (all/final): ").strip().lower()

# Resolution choices for 2D
print("\nChoose resolution:")
print("1 - 128×128")
print("2 - 256×256")
print("3 - 512×512")
print("4 - 1024×1024")
print("5 - 2048×2048")
print("6 - 4096×4096")
opt = input("Option (1–6): ").strip()
res_map = {'1':128, '2':256, '3':512, '4':1024, '5':2048, '6':4096}
resolution = res_map.get(opt, 512)
figsize = (6,6)
dpi = resolution

# Prepare output folder
folder = f"sierpinski_carpet_{save_format}"
os.makedirs(folder, exist_ok=True)

# Recursive function to draw Sierpinski Carpet at a target level
def draw_carpet(ax, x, y, size, level, target):
    if level == target:
        # draw filled square
        color = 'black' if color_mode=='binary' else plt.cm.viridis(level/max_gen)
        rect = plt.Rectangle((x, y), size, size, facecolor=color, edgecolor=None)
        ax.add_patch(rect)
        return
    if level > target:
        return
    new_size = size / 3
    # iterate 3x3 grid, skip center
    for i in range(3):
        for j in range(3):
            if i==1 and j==1:
                # skip middle square
                continue
            nx = x + i*new_size
            ny = y + j*new_size
            draw_carpet(ax, nx, ny, new_size, level+1, target)

# Determine which generations to render
if save_all=='all':
    gens = list(range(0, max_gen+1))
else:
    gens = [0, max_gen, 'final']

# Main loop: render each generation or combined final
for g in gens:
    if g=='final':
        # overlay all levels
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        for lvl in range(0, max_gen+1):
            draw_carpet(ax, 0, 0, 1, 0, lvl)
        out_name = f"sierpinski_carpet_final_all.{save_format}"
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        draw_carpet(ax, 0, 0, 1, 0, g)
        out_name = f"sierpinski_carpet_gen_{g}.{save_format}"
    out_path = os.path.join(folder, out_name)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved {'combined final' if g=='final' else 'generation '+str(g)} to {out_path}")

# Fractal Dimension output
formula = "log(8)/log(3)"
value = log(8)/log(3)
print(f"\nFractal Dimension (D) = {formula} ≈ {value:.4f}")
# Save FD to text file
fd_txt = os.path.join(folder, "fractal_dimension.txt")
with open(fd_txt,'w') as f:
    f.write(f"Fractal Dimension (D) = {formula} ≈ {value:.4f}\n")
print(f"FD saved to {fd_txt}")

# Archive results
shutil.make_archive(folder, 'zip', folder)
print(f"Archive created: {folder}.zip")
