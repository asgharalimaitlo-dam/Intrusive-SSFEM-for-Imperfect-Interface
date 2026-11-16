# =============================================================================
# STANDALONE POST-PROCESSING SCRIPT: plot_p_refinement.py
# Reads 'sfem_p_refinement.csv' and renders the 300 DPI spectral p-refinement
# convergence and computational cost scaling figure ('sfem_p_refinement.png').
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sfem_p_refinement.csv")

res_mat = np.loadtxt(csv_path, delimiter=",", skiprows=1)
p_orders, num_modes, err_mean, err_std, err_m3, wall_times = res_mat.T

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

# Left Panel: Semi-log spectral convergence across p = 1..4
ax1.semilogy(p_orders, err_mean, "o-", color="#1f77b4", lw=2.2, ms=7, label=r"Mean $\mathbb{E}[d]$ Rel. Error (%)")
ax1.semilogy(p_orders, err_std, "s--", color="#2ca02c", lw=2.2, ms=7, label=r"Std. Dev. $\mathrm{Std}[d]$ Rel. Error (%)")
ax1.semilogy(p_orders, err_m3, "^:", color="#d62728", lw=2.2, ms=7, label=r"3rd Central Moment $\mu_3[d]$ Error (%)")
ax1.set_xticks(p_orders)
ax1.set_xlabel(r"Hermite Polynomial Chaos Order $p$", fontweight="bold", fontsize=11)
ax1.set_ylabel(r"Relative $L^2$ Trajectory Error (%)", fontweight="bold", fontsize=11)
ax1.set_title(r"Spectral $p$-Refinement Convergence ($p \in \{1, 2, 3, 4\}$)", fontweight="bold", fontsize=11.5)
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

# Right Panel: Wall-clock runtime scaling across p = 1..4
bars = ax2.bar(p_orders, wall_times, width=0.45, color="#9467bd", alpha=0.85, edgecolor="k")
for bar, t_val in zip(bars, wall_times):
    ax2.text(bar.get_x() + bar.get_width() / 2.0, t_val + 0.3, f"{t_val:.2f} s",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_xticks(p_orders)
ax2.set_ylim(0, max(wall_times) * 1.18)
ax2.set_xlabel(r"Hermite Polynomial Chaos Order $p$", fontweight="bold", fontsize=11)
ax2.set_ylabel(r"Single-Pass Wall-Clock Time (s)", fontweight="bold", fontsize=11)
ax2.set_title(r"Computational Cost Scaling vs. Polynomial Order $p$", fontweight="bold", fontsize=11.5)
ax2.grid(True, axis="y", ls="--", alpha=0.5)

plt.tight_layout()
fig_out = os.path.join(script_dir, "sfem_p_refinement.png")
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
print(f"Spectral p-refinement figure saved to: {fig_out}")
plt.show()
