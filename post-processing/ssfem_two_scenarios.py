# =============================================================================
# STANDALONE POST-PROCESSING SCRIPT: plot_two scenarios.py
# Compares Damage Evolution & Uncertainty Bands for Scenario A (g_1 = 0) vs.
# Scenario B (g_1 != 0) from 'sfem_scenarios_A_vs_B.csv'.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the 5-column comparison dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sfem_scenarios_A_vs_B.csv")

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
time, d_mean_A, d_std_A, d_mean_B, d_std_B = data.T

# Print failure timing metrics (first time E[d] <= 0.01) for manuscript reporting
fail_idx_A = np.where(d_mean_A <= 0.01)[0]
fail_idx_B = np.where(d_mean_B <= 0.01)[0]
t_fail_A = time[fail_idx_A[0]] if len(fail_idx_A) > 0 else time[-1]
t_fail_B = time[fail_idx_B[0]] if len(fail_idx_B) > 0 else time[-1]

print("=== SCENARIO A vs. SCENARIO B SUMMARY ===")
print(f"Scenario A (g_1 = 0)  Complete Debonding Time: t = {t_fail_A:.4f} s | Max Std[d] = {np.max(d_std_A):.5f}")
print(f"Scenario B (g_1 != 0) Complete Debonding Time: t = {t_fail_B:.4f} s | Max Std[d] = {np.max(d_std_B):.5f}")

# 2. Configure 2-panel publication figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.6), sharex=True)

# Top Panel: Expected Integrity E[d] + +/-3*sigma Envelopes
ax1.plot(time, d_mean_A, color='#1f77b4', lw=2.2,
         label=r'Scenario A ($\mathbf{g}_1 = \mathbf{0}$): Expected Integrity $\mathbb{E}[d]$')
ax1.fill_between(time, np.clip(d_mean_A - 3.0 * d_std_A, 0.0, 1.0), np.clip(d_mean_A + 3.0 * d_std_A, 0.0, 1.0),
                 color='#1f77b4', alpha=0.18, label=r'Scenario A $\pm 3\mathrm{Std}[d]$ Volatility Envelope')

ax1.plot(time, d_mean_B, color='#d62728', ls='--', lw=2.2,
         label=r'Scenario B ($\mathbf{g}_1 \neq \mathbf{0}$): Expected Integrity $\mathbb{E}[d]$')
ax1.fill_between(time, np.clip(d_mean_B - 3.0 * d_std_B, 0.0, 1.0), np.clip(d_mean_B + 3.0 * d_std_B, 0.0, 1.0),
                 color='#d62728', alpha=0.22, label=r'Scenario B $\pm 3\mathrm{Std}[d]$ Volatility Envelope')

ax1.set_ylabel(r'Interface Integrity Variable $d$', fontweight='bold', fontsize=11)
ax1.set_ylim(-0.05, 1.05)
ax1.set_title(r'Sensitivity of Stochastic Damage Evolution to Boundary Traction Volatility ($\mathbf{g}_1$)',
              fontweight='bold', fontsize=11.5, pad=10)
ax1.grid(True, ls='--', alpha=0.5)
ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#ccc', fontsize=9.5)

# Bottom Panel: Direct Comparison of Damage Volatility Std[d]
ax2.plot(time, d_std_A, color='#1f77b4', lw=2.0,
         label=r'Scenario A ($\mathbf{g}_1 = \mathbf{0}$): Internal Itô Noise Only')
ax2.plot(time, d_std_B, color='#d62728', ls='--', lw=2.0,
         label=r'Scenario B ($\mathbf{g}_1 \neq \mathbf{0}$): Internal Noise + Boundary Traction Volatility')

ax2.set_xlabel(r'Simulation Time $t$ (s)', fontweight='bold', fontsize=11)
ax2.set_ylabel(r'Damage Volatility $\mathrm{Std}[d]$', fontweight='bold', fontsize=11)
ax2.grid(True, ls='--', alpha=0.5)
ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#ccc', fontsize=9.5)

plt.tight_layout()

# 3. Export 300 DPI publication figure
fig_out = os.path.join(script_dir, "sfem_scenarios_A_vs_B.png")
plt.savefig(fig_out, dpi=300, bbox_inches='tight')
print(f"Comparison graph saved to: {fig_out}")
plt.show()
