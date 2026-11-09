# An Intrusive Parallel SSFEM Framework for Asymmetric Imperfect Interfaces

This repository hosts the official open-source parallel computing pipeline and finite element solver developed for the manuscript: **"Stochastic Formulation and Intrusive Galerkin Solution of Asymmetric Imperfect Interfaces using Displacement Jump Operators."**

The software framework is engineered in Python utilizing the **DOLFINx (FEniCSx)** computing platform paired with the **PETSc** linear algebra backend to solve non-linear multi-axis fatigue degradation paths across zero-thickness manifolds.

## 🔬 Core Numerical Innovations
* **Intrusive Spectral Projections:** Eliminates traditional sampling loops by mapping random state-dependent structures onto a deterministic third-order algebraic chaos coupling tensor \(C_{ijk}\).
* **Asymmetric Unilateral Constraints:** Enforces strict Signorini non-penetration compliance (\(\lbrack\!\lbrack u_3 \rbrack\!\rbrack \ge 0\)) under cyclic loading via a high-magnitude linear contact penalty (\(\kappa_{\text{penalty}} = 10^6\) MPa/mm).
* **Multiplicative Itô Fatigue Dynamics:** Couples random microscale material faults with structural health using state-dependent noise integrals that satisfy thermodynamic irreversibility bounds (\(\dot{d} \le 0\)).

## 💻 Audited Workstation Execution Environment
All computational benchmarks and performance timelines were audited and recorded using the following localized workstation node architecture:
* **Hardware System:** HP ZBook G9 Mobile Workstation
* **Processor:** 12th Gen Intel(R) Core(TM) i7-12800H H-Series (14 Cores, 20 Threads, 2.40 GHz base)
* **Memory:** 16.0 GB dual-channel DDR5 RAM (15.6 GB addressable by Linux kernel)
* **Graphics:** NVIDIA RTX Workstation Laptop GPU (4 GB GDDR6 dedicated)
* **OS Environment:** Ubuntu Linux 22.04 LTS (x64 Subsystem)

---

## 📂 Repository Architecture & Core Code Components

The codebase is organized into a modular pipeline separating the parallel execution engines, computational error parsing operations, and automated publication-grade plotting utilities:

```text
intrusive-sfem-interface/
├── LICENSE                                 # Official MIT License file
├── README.md                               # Operational execution and build guide
├── requirements.txt                        # Python environment dependencies
│
├── src/                                    # CORE EXECUTION ENGINES
│   ├── ssfem_interface.py                  # Parallel FEniCSx simulation engine
│   ├── error_analysis.py                   # Multi-mesh h-refinement text parser
│   └── shear_jump.py                       # Transverse field extraction routine
│
└── post_process/                           # MANUSCRIPT PLOTTING PIPELINES
    ├── plot_interface_deformed_mesh.py     # Renders the warped element grid contours
    ├── plot_intrusive_non_intrusive_conv.py # Compares SSFEM modal curves vs. MCS reference
    ├── plot_interface_kinematics.py        # Maps dual-axis kinematic contact curves
    ├── plot_interface_damage_evolution.py  # Tracks history-dependent staircase fatigue steps
    ├── plot_shear_profile.py               # Plots uniform transverse horizontal responses
    ├── plot_stress_profile.py              # Visualizes the central cohesive softening stress troughs
    └── plot_error_analysis.py              # Generates the log-log h-convergence slope profiles
```

---

## 🚀 Getting Started & Execution Instructions

### 1. Environment Deployment via Docker (Recommended)
To eliminate local operating system dependency conflicts and secure the exact complex-valued PETSc bindings required by the backend assembler, pull and launch the official DOLFINx container container:
```bash
docker pull dolfinx/dolfinx:complex
docker run -it -v \$(pwd):/home/shared -w /home/shared dolfinx/dolfinx:complex
```

### 2. Running the Core Simulation Engine
To execute the 400-step high-resolution cyclic parametric sweep using distributed memory ranks across 4 active CPU thread streams, run via MPI:
```bash
mpirun -n 4 python3 src/ssfem_interface.py
```
*Note: This command solves the unified block-algebraic matrix equations concurrently in **~13.8 seconds***

### 3. Executing Spatial Error Parsers and Data Post-Processing
To compile the raw empirical data matrices for multi-mesh convergence tracking and sanitize complex-valued notation keys (`+0j`), run the background utility files:
```bash
# Run multi-mesh h-refinement sweeps to compile error matrices
python3 src/error_analysis.py

# Extract transverse shear jump vectors across the central cross-section
python3 src/shear_jump.py
```

### 4. Re-rendering Manuscript Figure Graphics
Once the text datasets are compiled, re-render the high-resolution vector graphics artifacts by running the localized plotting utilities:
```bash
# Render warped grids and total displacement magnitude contours
python3 post_process/plot_interface_deformed_mesh.py

# Plot spectral SSFEM convergence trajectories vs. MCS sample marks
python3 post_process/plot_intrusive_non_intrusive_convergence.py

# Map shear sliding waves against the Signorini contact floor
python3 post_process/plot_interface_kinematics.py

# Track threshold-dependent staircase damage plateaus
python3 post_process/plot_interface_damage_evolution.py

# Plot horizontal shear responses along the transverse width
python3 post_process/plot_shear_profile.py

# Visualize the stress relaxation damage localization troughs
python3 post_process/plot_stress_profile.py

# Generate log-log spatial error maps validating O(h²) slopes
python3 post_process/plot_error_analysis.py
```

---

## 📜 Academic Attribution & Citation
If you utilize this framework, its node-pair displacement discontinuity jump operators, or its intrusive stochastic temporal integration loop in your research, please provide formal attribution to our journal publication:

```bibtex
@article{asghar2026stochastic,
  author    = { Maitlo Asghar, Bauzet Caroline and Lebon Fr{\'e}deric},
  title     = {Stochastic Formulation and Intrusive Galerkin Solution of Asymmetric Imperfect Interfaces using Displacement Jump Operators},
  journal   = {yet to decide},
  volume    = {Accepted / In Press},
  year      = {2026},
  publisher = {},
  doi       = {}
}
```
