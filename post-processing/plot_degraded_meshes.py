# =============================================================================
# STANDALONE POST-PROCESSING SCRIPT:degraded_mesh.py
# Reads 'mesh_snapshots_data.npz' and renders a 300 DPI side-by-side comparison
# of the Pristine Mesh (t=0, d=1) vs. Degraded & Sheared Mesh (d -> 0).
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

script_dir = os.path.dirname(os.path.abspath(__file__))
npz_path = os.path.join(script_dir, "mesh_snapshots_data.npz")

data    = np.load(npz_path)
coords  = data["coords"]
cells   = data["cells"]
d_prist = data["d_prist"]
d_deg   = data["d_deg"]
u1_deg  = data["u1_deg"]
u2_deg  = data["u2_deg"]
t_final = float(data["t_final"])

x_prist, y_prist = coords[:, 0], coords[:, 1]
tri_prist = mtri.Triangulation(x_prist, y_prist, cells)

warp_scale = 3.5   # Visual magnification factor for tangential micro-slip
x_deg = x_prist + warp_scale * u1_deg
y_deg = y_prist + warp_scale * u2_deg
tri_deg = mtri.Triangulation(x_deg, y_deg, cells)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.2))

tpc1 = ax1.tripcolor(tri_prist, d_prist, shading="gouraud", cmap="RdYlGn", vmin=0.0, vmax=1.0)
ax1.triplot(tri_prist, color="k", lw=0.45, alpha=0.55)
ax1.set_title(r"(a) Pristine Imperfect Interface Mesh ($t = 0.00\text{ s}$, Undamaged State $d \approx 1.0$)",
              fontweight="bold", fontsize=11)
ax1.set_ylabel(r"Width $x_2$ (mm)", fontweight="bold", fontsize=10)
ax1.set_aspect("equal")

tpc2 = ax2.tripcolor(tri_deg, d_deg, shading="gouraud", cmap="RdYlGn", vmin=0.0, vmax=1.0)
ax2.triplot(tri_deg, color="k", lw=0.45, alpha=0.65)
ax2.set_title(rf"(b) Degraded & Sheared Interface Mesh ($t = {t_final:.2f}\text{{ s}}$, Warp Factor = ${warp_scale}\times$)",
              fontweight="bold", fontsize=11)
ax2.set_xlabel(r"Length $x_1$ (mm)", fontweight="bold", fontsize=10)
ax2.set_ylabel(r"Width $x_2$ (mm)", fontweight="bold", fontsize=10)
ax2.set_aspect("equal")

cbar = fig.colorbar(tpc2, ax=[ax1, ax2], orientation="vertical", fraction=0.025, pad=0.03)
cbar.set_label(r"Expected Interface Integrity $\mathbb{E}[d]$ ($1=$ Pristine, $0=$ Debonded)",
               fontweight="bold", fontsize=10)

fig_out = os.path.join(script_dir, "fig6_pristine_vs_degraded_mesh.png")
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
print(f"Pristine vs. Degraded mesh figure saved to: {fig_out}")
plt.show()
