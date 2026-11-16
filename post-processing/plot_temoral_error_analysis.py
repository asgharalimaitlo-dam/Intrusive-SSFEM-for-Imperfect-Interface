# =============================================================================
# STANDALONE TEMPORAL POST-PROCESSING SCRIPT: plot_temporal_error_analysis.py
# Reads 'sfem_temporal_error.csv', computes empirical log-log convergence orders,
# prints the summary table, and exports 'sfem_temporal_convergence.png' at 300 DPI.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Load temporal convergence data from local CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sfem_temporal_error.csv")

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
dt_arr  = data[:, 0]
u_arr   = data[:, 1]
chi_arr = data[:, 2]

# 2. Compute empirical convergence orders via bracket-free tuple unpacking
slope_u, _   = np.polyfit(np.log(dt_arr), np.log(u_arr), 1)
slope_chi, _ = np.polyfit(np.log(dt_arr), np.log(chi_arr), 1)

print("\n=== TEMPORAL CONVERGENCE SUMMARY ===")
for dt_val, eu, ec in zip(dt_arr, u_arr, chi_arr):
    print(f"dt = {dt_val:.4f} s | L2_Error_U = {eu:.6e} | L2_Error_Chi = {ec:.6e}")
print(f"Empirical Temporal Convergence Order (Displacement Jump U): {slope_u:.3f}")
print(f"Empirical Temporal Convergence Order (Interface Damage d):  {slope_chi:.3f}\n")

# 3. Render publication-grade log-log convergence plot
plt.figure(figsize=(7.5, 5.2))

plt.loglog(
    dt_arr, u_arr, 'o-', color='#1f77b4', lw=2.2, ms=7,
    label=rf'Displacement Jump Error $\Vert \mathbf{{u}} - \mathbf{{u}}_{{\mathrm{{ref}}}} \Vert_{{L^2(\mathcal{{S}})}}$ (Slope = {slope_u:.2f})'
)
plt.loglog(
    dt_arr, chi_arr, 's--', color='#d62728', lw=2.2, ms=7,
    label=rf'Interface Damage Error $\Vert d - d_{{\mathrm{{ref}}}} \Vert_{{L^2(\mathcal{{S}})}}$ (Slope = {slope_chi:.2f})'
)

# Theoretical first-order implicit backward Euler asymptote O(dt^1.0)
ref_o1 = np.min(u_arr) * (dt_arr / np.min(dt_arr))**1.0
plt.loglog(dt_arr, ref_o1, 'k:', lw=1.6, label=r'Theoretical Asymptote $\mathcal{O}(\Delta t^{1.0})$')

plt.xlabel(r'Time Step Size $\Delta t$ (s)', fontweight='bold', fontsize=11, labelpad=8)
plt.ylabel(r'Global $L^2(\mathcal{S})$ Temporal Error Norm', fontweight='bold', fontsize=11, labelpad=8)
plt.title('Implicit Backward Euler Temporal Convergence Profile', fontweight='bold', fontsize=12, pad=12)

plt.grid(True, which="both", ls='--', alpha=0.5)
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#ccc', fontsize=9.5)
plt.tight_layout()

# 4. Save 300 DPI publication figure
fig_out = os.path.join(script_dir, "sfem_temporal_convergence.png")
plt.savefig(fig_out, dpi=300, bbox_inches='tight')
print(f"Publication plot saved to: {fig_out}")
plt.show()
