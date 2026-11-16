#plot_main_sffem_interface.py is used for plots of figs1 to figs5
import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "fem_simulation_data1.csv")

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
time     = data[:, 0]
d_mean   = data[:, 1]
d_std    = data[:, 2]
u1_shear = data[:, 3]
u3_norm  = data[:, 4]
p0       = data[:, 5]

# Reconstruct applied cyclic traction wave for hysteresis loop (Plot 5)
wave = np.sin(2.0 * np.pi * 2.5 * time)
g1_traction = 35.0 * wave

# -----------------------------------------------------------------------------
# PLOT 1: Damage Evolution with Stochastic Volatility Band
# -----------------------------------------------------------------------------
plt.figure(figsize=(7.5, 4.8))
plt.plot(time, d_mean, color='#d62728', lw=2.2, label=r'Expected Integrity $\mathbb{E}[d]$')
plt.fill_between(time, np.clip(d_mean - 3*d_std, 0, 1), np.clip(d_mean + 3*d_std, 0, 1),
                 color='#d62728', alpha=0.2, label=r'$\pm 3\mathrm{Std}[d]$ Volatility Envelope')
plt.xlabel(r'Time $t$ (s)', fontweight='bold')
plt.ylabel(r'Interface Integrity $d$', fontweight='bold')
plt.title('Stochastic Interface Damage Evolution', fontweight='bold')
plt.ylim(-0.05, 1.05)
plt.grid(True, ls='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "fig1_damage_evolution.png"), dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# PLOT 2: Isolated Stochastic Volatility Trajectory
# -----------------------------------------------------------------------------
plt.figure(figsize=(7.5, 4.8))
plt.plot(time, d_std, color='#ff7f0e', lw=2.0, label=r'Damage Standard Deviation $\mathrm{Std}[d]$')
plt.xlabel(r'Time $t$ (s)', fontweight='bold')
plt.ylabel(r'Volatility $\mathrm{Std}[d]$', fontweight='bold')
plt.title('Multiplicative Itô Damage Volatility Dynamics', fontweight='bold')
plt.grid(True, ls='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "fig2_damage_volatility.png"), dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# PLOT 3: Dual-Axis Shear Jump Amplification vs. Damage
# -----------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
l1 = ax1.plot(time, u1_shear, color='#1f77b4', lw=2.0, label=r'Shear Jump $\mathbb{E}[\lbrack\!\lbrack u_1 \rbrack\!\rbrack]$')
ax1.set_xlabel(r'Time $t$ (s)', fontweight='bold')
ax1.set_ylabel(r'Tangential Shear Jump (mm)', color='#1f77b4', fontweight='bold')
ax1.grid(True, ls='--', alpha=0.5)

ax2 = ax1.twinx()
l2 = ax2.plot(time, d_mean, color='#d62728', ls='--', lw=2.0, label=r'Integrity $\mathbb{E}[d]$')
ax2.set_ylabel(r'Interface Integrity $\mathbb{E}[d]$', color='#d62728', fontweight='bold')
ax2.set_ylim(-0.05, 1.05)

lines = l1 + l2
ax1.legend(lines, [l.get_label() for l in lines], loc='lower left')
plt.title('Shear Sliding Amplification Under Stiffness Degradation', fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(script_dir, "fig3_shear_vs_damage.png"), dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# PLOT 4: Signorini Normal Jump vs. Lagrange Multiplier Contact Pressure
# -----------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
l1 = ax1.plot(time, u3_norm, color='#2ca02c', lw=2.0, label=r'Normal Jump $\mathbb{E}[\lbrack\!\lbrack u_3 \rbrack\!\rbrack]$')
ax1.set_xlabel(r'Time $t$ (s)', fontweight='bold')
ax1.set_ylabel(r'Normal Opening Jump (mm)', color='#2ca02c', fontweight='bold')
ax1.grid(True, ls='--', alpha=0.5)

ax2 = ax1.twinx()
l2 = ax2.plot(time, p0, color='#9467bd', ls='--', lw=2.0, label=r'Lagrange Multiplier $\mathbb{E}[p^0]$')
ax2.set_ylabel(r'Contact Pressure $p^0$ (MPa)', color='#9467bd', fontweight='bold')

lines = l1 + l2
ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
plt.title('Unilateral Signorini Contact & Lagrange Multiplier Recovery', fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(script_dir, "fig4_signorini_contact.png"), dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# PLOT 5: Cyclic Cohesive Shear Traction-Jump Hysteresis Loop
# -----------------------------------------------------------------------------
plt.figure(figsize=(7.0, 5.2))
plt.plot(u1_shear, g1_traction, color='#17becf', lw=1.8, label='Cyclic Cohesive Response')
plt.axhline(0, color='k', lw=0.8, alpha=0.5)
plt.axvline(0, color='k', lw=0.8, alpha=0.5)
plt.xlabel(r'Tangential Shear Jump $\mathbb{E}[\lbrack\!\lbrack u_1 \rbrack\!\rbrack]$ (mm)', fontweight='bold')
plt.ylabel(r'Applied Shear Traction $g_1(t)$ (MPa)', fontweight='bold')
plt.title('Cohesive Shear Traction-Slip Softening Hysteresis', fontweight='bold')
plt.grid(True, ls='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "fig5_cohesive_hysteresis.png"), dpi=300)
plt.close()
plt.show
print("All 5 individual publication figures saved cleanly at 300 DPI!")
