# file plot_special_error_analysis.py used to plot special error.
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Load spatial convergence data from local CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sfem_spatial_error.csv")

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
h_arr   = data[:, 0]
u_arr   = data[:, 1]
chi_arr = data[:, 2]

# 2. Compute empirical convergence rates via log-log linear regression
slope_u, _   = np.polyfit(np.log(h_arr), np.log(u_arr), 1)
slope_chi, _ = np.polyfit(np.log(h_arr), np.log(chi_arr), 1)


print("=== SPATIAL CONVERGENCE SUMMARY ===")
for h_val, eu, ec in zip(h_arr, u_arr, chi_arr):
    print(f"h = {h_val:5.2f} mm | L2_Error_U = {eu:.6e} | L2_Error_Chi = {ec:.6e}")
print(f"Empirical Convergence Order (Displacement Jump U): {slope_u:.3f}")
print(f"Empirical Convergence Order (Interface Damage d):  {slope_chi:.3f}")

# 3. Render publication-grade log-log convergence plot
plt.figure(figsize=(7.5, 5.2))

plt.loglog(
    h_arr, u_arr, 'o-', color='#1f77b4', lw=2.2, ms=7,
    label=rf'Displacement Jump Error $\Vert \mathbf{{u}} - \mathbf{{u}}_{{\mathrm{{ref}}}} \Vert_{{L^2(\mathcal{{S}})}}$ (Slope = {slope_u:.2f})'
)
plt.loglog(
    h_arr, chi_arr, 's--', color='#d62728', lw=2.2, ms=7,
    label=rf'Interface Damage Error $\Vert d - d_{{\mathrm{{ref}}}} \Vert_{{L^2(\mathcal{{S}})}}$ (Slope = {slope_chi:.2f})'
)

# Theoretical second-order quadratic reference line O(h^2) anchored at finest step
ref_o2 = np.min(u_arr) * (h_arr / np.min(h_arr))**2
plt.loglog(h_arr, ref_o2, 'k:', lw=1.6, label=r'Theoretical Asymptote $\mathcal{O}(h^2)$')

plt.xlabel(r'Characteristic Mesh Element Size $h$ (mm)', fontweight='bold', fontsize=11, labelpad=8)
plt.ylabel(r'Global $L^2(\mathcal{S})$ Spatial Error Norm', fontweight='bold', fontsize=11, labelpad=8)
plt.title('Multi-Mesh Spatial Convergence Profile: Displacement Jump & Damage', fontweight='bold', fontsize=12, pad=12)

plt.grid(True, which="both", ls='--', alpha=0.5)
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#ccc', fontsize=9.5)
plt.tight_layout()

# 4. Save 300 DPI vector-quality figure
fig_out = os.path.join(script_dir, "sfem_spatial_convergence.png")
plt.savefig(fig_out, dpi=300, bbox_inches='tight')
print(f"Publication plot saved to: {fig_out}")
plt.show()
