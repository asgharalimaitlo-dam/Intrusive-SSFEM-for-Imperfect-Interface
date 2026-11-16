# =============================================================================
# STANDALONE VALIDATION POST-PROCESSING: plot_ssfem_vs_mcs.py
# Overlays Intrusive SSFEM-Uzawa (1 coupled pass) against 500-Sample MCS
# Ensemble (Mean +/- 3*Std) from 'ssfem_vs_mcs_500.csv'.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "ssfem_vs_mcs_500.csv")

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
time, ssfem_mean, ssfem_std, mcs_mean, mcs_std = data.T

# Compute exact relative L2 error metrics across the full 0.80 s horizon
rel_l2_mean = np.linalg.norm(ssfem_mean - mcs_mean) / np.linalg.norm(mcs_mean) * 100.0
max_abs_diff = np.max(np.abs(ssfem_mean - mcs_mean))

print("\n=== INTRUSIVE SSFEM vs. 500-SAMPLE MCS ACCURACY REPORT ===")
print(f"Relative L2 Trajectory Error (Mean E[d]): {rel_l2_mean:.3f}%")
print(f"Maximum Pointwise Difference:            {max_abs_diff:.5f}\n")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True)

# Top Panel: Expected Integrity E[d] + +/-3*sigma Envelopes
ax1.plot(time, mcs_mean, color='#2ca02c', lw=2.6, alpha=0.85,
         label=r'500-Sample MCS Reference ($\mu_{\mathrm{MCS}}$)')
ax1.fill_between(time, np.clip(mcs_mean - 3.0 * mcs_std, 0.0, 1.0), np.clip(mcs_mean + 3.0 * mcs_std, 0.0, 1.0),
                 color='#2ca02c', alpha=0.18, label=r'500-Sample MCS $\pm 3\,Std[d]$ Envelope')

ax1.plot(time, ssfem_mean, color='#d62728', ls='--', lw=2.2,
         label=rf'Intrusive SSFEM--Uzawa ($\mathbb{{E}}[d]$, Rel. $L^2$ Err = {rel_l2_mean:.2f}%)')
ax1.plot(time, np.clip(ssfem_mean - 3.0 * ssfem_std, 0.0, 1.0), color='#d62728', ls=':', lw=1.4,
         label=r'Intrusive SSFEM $\pm 3\,Std[d]$ Bounds')
ax1.plot(time, np.clip(ssfem_mean + 3.0 * ssfem_std, 0.0, 1.0), color='#d62728', ls=':', lw=1.4)

ax1.set_ylabel(r'Interface Integrity Variable $d$', fontweight='bold', fontsize=11)
ax1.set_ylim(-0.05, 1.05)
ax1.set_title('Validation of Intrusive SSFEM--Uzawa Against 500-Sample Monte Carlo Simulation',
              fontweight='bold', fontsize=11.5, pad=10)
ax1.grid(True, ls='--', alpha=0.5)
ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#ccc', fontsize=9.2)

# Bottom Panel: Direct Comparison of Stochastic Standard Deviation Std[d]
ax2.plot(time, mcs_std, color='#2ca02c', lw=2.4, alpha=0.85, label=r'500-Sample MCS Standard Deviation ($\sigma_{\mathrm{MCS}}$)')
ax2.plot(time, ssfem_std, color='#d62728', ls='--', lw=2.1, label=r'Intrusive SSFEM--Uzawa Volatility ($\mathrm{Std}[d]$)')

ax2.set_xlabel(r'Simulation Time $t$ (s)', fontweight='bold', fontsize=11)
ax2.set_ylabel(r'Damage Volatility $\mathrm{Std}[d]$', fontweight='bold', fontsize=11)
ax2.grid(True, ls='--', alpha=0.5)
ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#ccc', fontsize=9.5)

plt.tight_layout()
fig_out = os.path.join(script_dir, "ssfem_vs_mcs_validation.png")
plt.savefig(fig_out, dpi=300, bbox_inches='tight')
print(f"Validation overlay figure saved to: {fig_out}")
plt.show()
