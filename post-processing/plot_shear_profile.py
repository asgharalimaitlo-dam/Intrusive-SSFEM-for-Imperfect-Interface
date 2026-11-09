import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. LOAD AND SANITIZE STRIPPED DATA FROM COMPLEX CSV RAW TEXTS
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

plt.figure(figsize=(7.5, 5))

stages = [1, 2, 3]
colors = ['#2ca02c', '#ff7f0e', '#d62728']  # Emerald Green, Safety Orange, Deep Crimson

print("Parsing complex step profile arrays from workspace folder...")

for idx, step in enumerate(stages):
    filename = f"shear_jump_slice_step_{step:02d}.csv"
    filepath = os.path.join(script_dir, filename)
    
    if os.path.exists(filepath):
        cleaned_rows = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or 'x2_coordinate' in line or not line.strip():
                    continue
                
                sanitized_line = line.replace('(', '').replace(')', '').strip()
                tokens = [t.strip() for t in sanitized_line.split(',') if t.strip()]
                
                row_values = []
                for val in tokens:
                    if 'j' not in val:
                        val += 'j'
                    row_values.append(float(complex(val).real))
                
                if len(row_values) >= 2:
                    cleaned_rows.append(row_values[:2])
        
        data = np.array(cleaned_rows)
        x2_coords = data[:, 0]
        
        # AMPLIFICATION FIX: Multiply by 1000 to convert from mm to micrometers (um)
        # This makes the non-linear curvature highly visible on the canvas!
        u2_jumps  = data[:, 1] * 1000.0
        
        plt.plot(x2_coords, u2_jumps, '-', color=colors[idx], linewidth=2.5,
                 label=r'Load Step {:02d} ($t = {:.2f}$s)'.format(step, step*0.01))
    else:
        print(f"Warning: File {filename} not found. Skipping plot line profile.")

# =============================================================================
# 2. RESOLUTION GRAPH FORMATTING RULES
# =============================================================================
plt.xlabel(r'Spatial Width Coordinate $x_2$ (mm) along Cross-section $x_1 = L/2$', fontweight='bold', fontsize=11)

# FIXED: Updated label units to Micrometers (um)
plt.ylabel(r'Tangential Shear Sliding Jump $\lbrack\!\lbrack u_2 \rbrack\!\rbrack$ ($\mu$m)', fontweight='bold', fontsize=11)
plt.title('Spatial Distribution of Interface Shear Displacement Jump', fontweight='bold', fontsize=12, pad=15)

# AUTO-SCALE FIX: Let matplotlib automatically fit tight bounds around the micro-values
plt.autoscale(enable=True, axis='y', tight=False)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()

# Save publication-grade vector graphics artifact directly to project directory
output_fig = 'interface_shear_profile.png'
plt.savefig(os.path.join(script_dir, output_fig), dpi=300, bbox_inches='tight')
print(f"Profile curve rendered cleanly and exported to  active workspace as '{output_fig}'.")

plt.show()
