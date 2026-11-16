# =============================================================================
# SPATIAL CONVERGENCE ERROR ANALYSIS: spatial_error_analysis.py
# Computes exact L2(S) spatial errors for Displacement Jump (U) & Damage (d)
# across multi-mesh refinements h = L / nx using DOLFINx non-matching mesh interpolation.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem
import dolfinx.fem.petsc
import basix.ufl
import ufl
from ufl import inner, grad, dx

# =============================================================================
# 1. SHARED PHYSICAL & NUMERICAL PARAMETERS
# =============================================================================
L, W = 100.0, 20.0
P_modes = 2

eta = 50.0
omega = 0.05
alpha = 0.1
dt = 0.002
eval_steps = 50       # Evaluate spatial error at t = 50 * 0.002 = 0.10 s (active damage regime)
penalty_val = 1e6

mu_0 = 50.0
lam_0 = 70.0

C_ijk = np.zeros((P_modes, P_modes, P_modes))
C_ijk[0, 0, 0] = 1.0
C_ijk[0, 1, 1] = 1.0
C_ijk[1, 0, 1] = 1.0
C_ijk[1, 1, 0] = 1.0


def run_ssfem_on_mesh(nx, ny):
    """Executes the intrusive SSFEM-Uzawa solver on an (nx x ny) mesh and returns Mode-0 U and Chi."""
    domain = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=[np.array([0.0, 0.0]), np.array([L, W])],
        n=[nx, ny],
        cell_type=mesh.CellType.triangle
    )

    el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    el_pce_u = basix.ufl.mixed_element([el_u] * P_modes)
    V_jump = fem.functionspace(domain, el_pce_u)

    el_chi = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    el_pce_chi = basix.ufl.mixed_element([el_chi] * P_modes)
    V_chi = fem.functionspace(domain, el_pce_chi)
    V_p   = fem.functionspace(domain, el_pce_chi)

    U_modes   = fem.Function(V_jump)
    Chi_modes = fem.Function(V_chi)
    Chi_old   = fem.Function(V_chi)
    p_mult    = fem.Function(V_p)

    # Spatial heterogeneity in initial micro-defect triggers smooth localized damage gradients
    Chi_modes.sub(0).interpolate(lambda x: 1.0 - 0.05 * np.exp(-((x[0] - L/2.0)**2) / (2.0 * 15.0**2)))
    Chi_modes.sub(1).interpolate(lambda x: 0.01 * np.ones(x.shape[1]))

    Chi_modes.x.scatter_forward()

    Chi_old.x.array[:] = Chi_modes.x.array[:]
    Chi_old.x.scatter_forward()

    _, dofs_u0_z = V_jump.sub(0).sub(2).collapse()
    _, dofs_u1_z = V_jump.sub(1).sub(2).collapse()
    _, dofs_chi0 = V_chi.sub(0).collapse()
    _, dofs_chi1 = V_chi.sub(1).collapse()
    _, dofs_p0   = V_p.sub(0).collapse()
    _, dofs_p1   = V_p.sub(1).collapse()

    v_test_modes   = ufl.TestFunctions(V_jump)
    u_trial_modes  = ufl.split(U_modes)
    chi_curr_modes = ufl.split(Chi_modes)
    p_curr_modes   = ufl.split(p_mult)

    def get_k_star(chi_val):
        k_normal = (lam_0 + 2.0 * mu_0) * (chi_val**2)
        k_shear  = mu_0 * (chi_val**2)
        return ufl.as_matrix([[k_shear, 0, 0], [0, k_shear, 0], [0, 0, k_normal]])

    def get_k_star_deriv(chi_val):
        k_normal_d = 2.0 * (lam_0 + 2.0 * mu_0) * chi_val
        k_shear_d  = 2.0 * mu_0 * chi_val
        return ufl.as_matrix([[k_shear_d, 0, 0], [0, k_shear_d, 0], [0, 0, k_normal_d]])

    g_stochastic = [
        fem.Constant(domain, PETSc.ScalarType((35.0, 0.0, 45.0))),
        fem.Constant(domain, PETSc.ScalarType((4.0, 0.0, 5.0)))
    ]
    e_3 = ufl.as_vector([0.0, 0.0, 1.0])

    F_equilibrium = 0.0
    for k in range(P_modes):
        v_k = v_test_modes[k]
        F_equilibrium -= ufl.inner(g_stochastic[k], v_k) * dx

        aug_p_k = ufl.conditional(
            ufl.real(p_curr_modes[k] + penalty_val * ufl.dot(u_trial_modes[k], e_3)) < 0.0,
            p_curr_modes[k] + penalty_val * ufl.dot(u_trial_modes[k], e_3),
            0.0
        )


        F_equilibrium += aug_p_k * ufl.inner(e_3, v_k) * dx

        for i in range(P_modes):
            u_i = u_trial_modes[i]
            for j in range(P_modes):
                if C_ijk[i, j, k] > 0.0:
                    F_equilibrium += C_ijk[i, j, k] * ufl.inner(get_k_star(chi_curr_modes[j]) * u_i, v_k) * dx

    u_problem = dolfinx.fem.petsc.NonlinearProblem(
        F_equilibrium, U_modes,
        petsc_options={
            "snes_type": "newtonls", "snes_linesearch_type": "none",
            "snes_atol": 1e-8, "snes_rtol": 1e-8,
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"
        },
        petsc_options_prefix=f"spatial_err_{nx}_"
    )

    chi_trial = ufl.TrialFunctions(V_chi)
    chi_test  = ufl.TestFunctions(V_chi)

    e_1 = ufl.as_vector([1.0, 0.0, 0.0])
    e_2 = ufl.as_vector([0.0, 1.0, 0.0])
    e_3 = ufl.as_vector([0.0, 0.0, 1.0])

    u0_vec = u_trial_modes[0]
    u0_1 = ufl.dot(u0_vec, e_1)
    u0_2 = ufl.dot(u0_vec, e_2)
    u0_3 = ufl.dot(u0_vec, e_3)

    u_0_pos = ufl.as_vector([
        u0_1,
        u0_2,
        ufl.conditional(ufl.real(u0_3) < 0.0, 0.0, u0_3)
    ])
    energy_driving_force = ufl.real(0.5 * ufl.inner(get_k_star_deriv(chi_curr_modes[0]) * u_0_pos, u_0_pos))


    M_chi_form, R_damage_form = 0.0, 0.0
    for j in range(P_modes):
        M_chi_form += (eta / dt) * ufl.inner(chi_trial[j], chi_test[j]) * dx
        R_damage_form += (eta / dt) * ufl.inner(Chi_old.sub(j), chi_test[j]) * dx
        R_damage_form -= alpha * ufl.inner(grad(chi_curr_modes[j]), grad(chi_test[j])) * dx

        if j == 0:
            damage_source = ufl.conditional(omega - energy_driving_force < 0.0, omega - energy_driving_force, 0.0)
        else:
            damage_source = ufl.conditional(omega - energy_driving_force < 0.0, -0.25 * energy_driving_force * chi_curr_modes[1], 0.0)

        R_damage_form += ufl.inner(damage_source, chi_test[j]) * dx

    a_chi = fem.form(M_chi_form)
    L_chi = fem.form(R_damage_form)
    A_matrix = fem.petsc.create_matrix(a_chi)
    b_vector = fem.petsc.create_vector(V_chi)

    solver_chi = PETSc.KSP().create(domain.comm)
    solver_chi.setType(PETSc.KSP.Type.PREONLY)
    solver_chi.getPC().setType(PETSc.PC.Type.LU)
    solver_chi.getPC().setFactorSolverType("mumps")

    # Deterministic zero-noise seed for clean spatial discretization error isolation
    for step in range(1, eval_steps + 1):
        current_time = step * dt
        wave = np.sin(2.0 * np.pi * 2.5 * current_time)
        g_stochastic[0].value = np.array([35.0 * wave, 0.0, 45.0 * wave])
        g_stochastic[1].value = np.array([4.0 * wave, 0.0, 5.0 * wave])


        p_mult.x.array[:] = 0.0
        for _ in range(5):
            u_problem.solve()
            U_modes.x.scatter_forward()
            p_mult.x.array[dofs_p0] = np.minimum(0.0, np.real(p_mult.x.array[dofs_p0]) + penalty_val * np.real(U_modes.x.array[dofs_u0_z]))
            p_mult.x.array[dofs_p1] = np.real(p_mult.x.array[dofs_p1]) + penalty_val * np.real(U_modes.x.array[dofs_u1_z])
            p_mult.x.array[dofs_p1][np.real(p_mult.x.array[dofs_p0]) >= 0.0] = 0.0
            p_mult.x.scatter_forward()

        A_matrix.zeroEntries()
        fem.petsc.assemble_matrix(A_matrix, a_chi)
        A_matrix.assemble()

        b_vector.set(0.0)
        fem.petsc.assemble_vector(b_vector, L_chi)
        b_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        solver_chi.setOperators(A_matrix)
        solver_chi.solve(b_vector, Chi_modes.x.petsc_vec)
        Chi_modes.x.scatter_forward()

        chi0_vals = np.real(Chi_old.x.array[dofs_chi0])
        chi0_new = np.minimum(np.clip(np.real(Chi_modes.x.array[dofs_chi0]), 0.0, 1.0), chi0_vals)
        Chi_modes.x.array[dofs_chi0] = chi0_new
        Chi_modes.x.array[dofs_chi1] = np.real(Chi_modes.x.array[dofs_chi1]) * chi0_new
        Chi_modes.x.scatter_forward()

        Chi_old.x.array[:] = Chi_modes.x.array[:]
        Chi_old.x.scatter_forward()

    solver_chi.destroy()
    A_matrix.destroy()
    b_vector.destroy()

    return U_modes.sub(0).collapse(), Chi_modes.sub(0).collapse()

# =============================================================================
# 2. MULTI-MESH REFINEMENT LOOP & EXACT L2(S) ERROR NORM INTEGRATION
# =============================================================================
if MPI.COMM_WORLD.rank == 0:
    print("Computing high-resolution reference solution (nx=80, ny=40)...")

u_ref, chi_ref = run_ssfem_on_mesh(80, 40)
V_u_ref   = u_ref.function_space
V_chi_ref = chi_ref.function_space
domain_ref = V_u_ref.mesh

coarse_grids = [(10, 5), (20, 10), (40, 20)]
h_list, err_u_list, err_chi_list = [], [], []

for (nx, ny) in coarse_grids:
    h_val = L / nx
    if MPI.COMM_WORLD.rank == 0:
        print(f"Evaluating coarse grid nx={nx}, ny={ny} (h = {h_val:.2f} mm)...")

    u_coarse, chi_coarse = run_ssfem_on_mesh(nx, ny)

       # Interpolate coarse solutions onto fine reference mesh (DOLFINx 0.9+ API)
    cells_ref = np.arange(domain_ref.topology.index_map(domain_ref.topology.dim).size_local, dtype=np.int32)

    u_interp = fem.Function(V_u_ref)
    nmm_u = fem.create_interpolation_data(V_u_ref, u_coarse.function_space, cells_ref, padding=1e-6)
    u_interp.interpolate_nonmatching(u_coarse, cells_ref, nmm_u)

    chi_interp = fem.Function(V_chi_ref)
    nmm_chi = fem.create_interpolation_data(V_chi_ref, chi_coarse.function_space, cells_ref, padding=1e-6)
    chi_interp.interpolate_nonmatching(chi_coarse, cells_ref, nmm_chi)



    # Assemble exact global L2(S) error norms across MPI partitions
    diff_u = u_interp - u_ref
    l2_u_local = fem.assemble_scalar(fem.form(ufl.real(ufl.inner(diff_u, diff_u)) * dx))
    l2_u = np.sqrt(domain_ref.comm.allreduce(l2_u_local, op=MPI.SUM))

    diff_chi = chi_interp - chi_ref
    l2_chi_local = fem.assemble_scalar(fem.form(ufl.real(ufl.inner(diff_chi, diff_chi)) * dx))
    l2_chi = np.sqrt(domain_ref.comm.allreduce(l2_chi_local, op=MPI.SUM))

    h_list.append(h_val)
    err_u_list.append(float(np.real(l2_u)))
    err_chi_list.append(float(np.real(l2_chi)))

    if MPI.COMM_WORLD.rank == 0:
        print(f"  -> h = {h_val:.2f} mm | L2_Error_U = {l2_u:.6e} | L2_Error_Chi = {l2_chi:.6e}")

# =============================================================================
# 3. SAVE CLEAN REAL-VALUED SPATIAL ERROR CSV ON RANK 0 (NO PLOTTING)
# =============================================================================
if MPI.COMM_WORLD.rank == 0:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_matrix = np.column_stack((h_list, err_u_list, err_chi_list))
    csv_out = os.path.join(script_dir, "sfem_spatial_error.csv")
    np.savetxt(
        csv_out,
        results_matrix,
        delimiter=",",
        header="h_size,L2_Error_U,L2_Error_Chi",
        comments=""
    )
    print(f"\nSpatial error analysis complete. Clean dataset saved to: {csv_out}")

