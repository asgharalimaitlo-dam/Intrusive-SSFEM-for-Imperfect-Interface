# Asymptotic and Spectral Stochastic Finite Element Modeling of Imperfect Interfaces via Uzawa Augmented Lagrangian

[![FEniCSx](https://shields.io)](https://fenicsproject.org)
[![PETSc / MUMPS](https://shields.io)](https://petsc.org)
[![MPI Parallel](https://shields.io)](https://readthedocs.io)

Parallel `FEniCSx` / `PETSc` implementation of a **non-linear Spectral Stochastic Finite Element Method (SSFEM)** coupled with an **Uzawa Augmented Lagrangian contact solver** for zero-thickness imperfect interfaces derived via vanishing-thickness asymptotic expansion ($\varepsilon \to 0$).

The framework resolves mixed-mode cohesive degradation, unilateral Signorini contact ($\lbrack\!\lbrack u_3 \rbrack\!\rbrack \geq 0$), and thermodynamically irreversible ($\dot{d} \le 0$) multiplicative Itô stochastic damage evolution in a single coupled pass using orthonormalized Hermite Generalized Polynomial Chaos (gPC) bases. Every simulation module is cleanly separated into a parallel MPI solver script and a standalone post-processing script (`plot_*.py`).

---

## Key Highlights & Verified Benchmarks

* **Asymptotic–Stochastic Formulation:** Couples a zero-thickness cohesive–contact law ($\varepsilon \to 0$) with multiplicative stochastic damage evolution via pre-computed third-order Galerkin contraction tensors $C_{ijk} = \mathbb{E}[\hat{\Psi}_i \hat{\Psi}_j \hat{\Psi}_k]$.
* **Optimal Numerical Convergence:** Achieves quadratic spatial convergence ($\mathcal{O}(h^2)$, slope $2.14$) and first-order temporal accuracy ($\mathcal{O}(\Delta t)$, slopes $1.09\text{--}1.11$).
* **$305.6\times$ Monte Carlo Speedup:** Completes a $400$-step cyclic simulation in **$9.88\text{ s}$** on $4$ MPI ranks versus **$3{,}019.84\text{ s}$** ($50.33\text{ min}$) for a $500$-sample non-intrusive Monte Carlo ensemble ($6.541\%$ relative $L^2$ discrepancy at $p=1$).
* **Stochastic $p$-Refinement ($p \in \{1, 2, 3, 4\}$):** Demonstrates systematic spectral error decay across higher-order moments, reducing third-central skewness error from $100.00\%$ ($p=1$) to $30.47\%$ ($p=4$).

---

## Repository Structure

| # | Solver Script | Post-Processing Script | Description |
| :- | :--- | :--- | :--- |
| **1** | `main_ssfem_interface.py` | `plot_main_ssfem_interface.py` | Transient cyclic debonding, $\pm 3\sigma$ bounds, shear jump, Signorini contact & hysteresis |
| **2** | `degraded_meshes.py` | `plot_degraded_meshes.py` | Full-field spatial damage localization and magnified interfacial distortion |
| **3** | `ssfem_two_scenarios.py` | `plot_ssfem_two_scenarios.py` | Parametric comparison of intrinsic stochasticity vs. fluctuating boundary tractions |
| **4** | `special_error_analysis.py` | `plot_special_error_analysis.py` | Spatial $h$-refinement verification across nested triangulations ($\mathcal{O}(h^2)$) |
| **5** | `temporal_error_analysis.py` | `plot_temporal_error_analysis.py` | Backward Euler–Maruyama temporal step-size convergence ($\mathcal{O}(\Delta t)$) |
| **6** | `ssfem_mcs_validation.py` | `plot_ssfem_mcs_validation.py` | $500$-sample Monte Carlo wall-clock speedup ($305.6\times$) & trajectory benchmark |
| **7** | `p_refinement.py` | `plot_p_refinement.py` | Spectral $p$-refinement across Hermite orders $p \in \{1, 2, 3, 4\}$ ($2\text{--}5$ modes) |

---

## Installation & Dependencies

This codebase requires **DOLFINx (`v0.8+`)** compiled with **PETSc** and **MUMPS** direct sparse LU solver support:

```bash
conda create -n fenicsx-ssfem -c conda-forge fenics-dolfinx petsc4py mpi4py numpy scipy matplotlib
conda activate fenicsx-ssfem
```

---

## Reproducing All Manuscript Results

Clone the repository and execute each solver + post-processing pair from the project root directory:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 1. Transient Cyclic Debonding & Signorini Contact
```bash
mpirun -n 4 python3 main_ssfem_interface.py
python3 plot_main_ssfem_interface.py
```

### 2. Full-Field Spatial Degradation & Mesh Distortion
```bash
mpirun -n 4 python3 degraded_meshes.py
python3 plot_degraded_meshes.py
```

### 3. Stochastic Boundary Excitation Sensitivity
```bash
mpirun -n 4 python3 ssfem_two_scenarios.py
python3 plot_ssfem_two_scenarios.py
```

### 4. Spatial & Temporal Convergence Studies
```bash
# Spatial h-refinement study
mpirun -n 4 python3 special_error_analysis.py
python3 plot_special_error_analysis.py

# Temporal dt-refinement study
mpirun -n 4 python3 temporal_error_analysis.py
python3 plot_temporal_error_analysis.py
```

### 5. 500-Sample Monte Carlo Benchmark
* Set `TEST_MODE = True` at line 15 of `ssfem_mcs_validation.py` for a **~15-second** $20$-sample verification run.
* Set `TEST_MODE = False` for the full **$500$-sample $\times$ $400$-step** manuscript benchmark (~50 minutes):
```bash
mpirun -n 4 python3 ssfem_mcs_validation.py
python3 plot_ssfem_mcs_validation.py
```

### 6. Stochastic $p$-Refinement Study
Evaluates orthonormalized Hermite expansions for $p \in \{1, 2, 3, 4\}$ against a $25$-point Gauss–Hermite quadrature reference (~35 seconds):
```bash
python3 p_refinement.py
python3 plot_p_refinement.py
```

---

## Citation

If you use this codebase in your research, please cite the accompanying manuscript:

```bibtex
@article{Asghar_SSFEM_Uzawa_2026,
  title   = {Asymptotic and Spectral Stochastic Finite Element Modeling of Imperfect Interfaces via Uzawa Augmented Lagrangian},
  author  = {Asghar Ali Maitlo, Frédéric Lebon and Caroline Bauzet},
  journal = {Submitted},
  year    = {2026}
}
```
