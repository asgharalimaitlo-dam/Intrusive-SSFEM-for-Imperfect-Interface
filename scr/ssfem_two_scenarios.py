# =============================================================================
# SCENARIO SENSITIVITY SOLVER: ssfem_two_scenarios.py
# Compares Scenario A (g_1 = 0: Deterministic Load) vs. Scenario B (g_1 != 0:
# Stochastic Boundary Traction) under Intrusive SSFEM-Uzawa Contact Dynamics.
# =============================================================================

import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem
import dolfinx.fem.petsc
import basix.ufl
import ufl
from ufl import inner, grad, dx

# =============================================================================
# 1. SHARED PHYSICAL PARAMETERS & SHARED MESH
# =============================================================================
L, W = 100.0, 20.0
P_modes = 2

eta = 50
omega = 0.05
alpha = 0.1
dt = 0.002
total_steps = 400     # 0.80 s total cyclic horizon
penalty_val = 1e6

mu_0 = 50.0
lam_0 = 70.0

C_ijk = np.zeros((P_modes, P_modes, P_modes))
C_ijk[0, 0, 0] = 1.0
C_ijk[0, 1, 1] = 1.0
C_ijk[1, 0, 1] = 1.0
C_ijk[1, 1, 0] = 1.0


domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[np.array([0.0, 0.0]), np.array([L, W])],
    n=[50,20],
    cell_type=mesh.CellType.triangle
)

def run_scenario(g1_active):
    """Runs the intrusive SSFEM-Uzawa solver with g_1=0 (False) or g_1!=0 (True)."""
    tag = "B" if g1_active else "A"
    if MPI.COMM_WORLD.rank == 0:
        print(f"\n--- Running Scenario {tag} (g_1 active = {g1_active}) ---")

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

    # Identical initial micro-defect seed for fair comparison
    Chi_modes.sub(0).interpolate(lambda x: np.ones(x.shape[1]))
    Chi_modes.sub(1).interpolate(lambda x: 0.01 * np.ones(x.shape[1]))

    Chi_modes.x.scatter_forward()

    Chi_old.x.array[:] = Chi_modes.x.array[:]
    Chi_old.x.scatter_forward()

    _, dofs_u0_z = V_jump.sub(0).sub(2).collapse()
    _, dofs_u1_z = V_jump.sub(1).sub(2).collapse()
    V_chi_single, dofs_chi0 = V_chi.sub(0).collapse()
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

    g0_const = fem.Constant(domain, PETSc.ScalarType((35.0, 0.0, 45.0)))
    g1_const = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0)))
    g_stochastic = [g0_const, g1_const]

    e_1 = ufl.as_vector([1.0, 0.0, 0.0])
    e_2 = ufl.as_vector([0.0, 1.0, 0.0])
    e_3 = ufl.as_vector([0.0, 0.0, 1.0])

    F_equilibrium = 0.0
    for k in range(P_modes):
        v_k = v_test_modes[k]
        F_equilibrium -= ufl.inner(g_stochastic[k], v_k) * dx

        u_k_3 = ufl.dot(u_trial_modes[k], e_3)
        aug_p_k = ufl.conditional(
            ufl.real(p_curr_modes[k] + penalty_val * u_k_3) < 0.0,
            p_curr_modes[k] + penalty_val * u_k_3,
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
        petsc_options_prefix=f"scen_{tag}_"
    )

    chi_trial = ufl.TrialFunctions(V_chi)
    chi_test  = ufl.TestFunctions(V_chi)

    u0_vec, u1_vec = u_trial_modes
    u0_pos = ufl.as_vector([
        ufl.dot(u0_vec, e_1),
        ufl.dot(u0_vec, e_2),
        ufl.conditional(ufl.real(ufl.dot(u0_vec, e_3)) < 0.0, 0.0, ufl.dot(u0_vec, e_3))
    ])
    u1_pos = ufl.as_vector([
        ufl.dot(u1_vec, e_1),
        ufl.dot(u1_vec, e_2),
        ufl.conditional(ufl.real(ufl.dot(u1_vec, e_3)) < 0.0, 0.0, ufl.dot(u1_vec, e_3))
    ])

    chi0_curr, chi1_curr = chi_curr_modes
    # Mode 0 driving energy includes C_000*(u0*u0) + C_110*(u1*u1) cross-coupling
    energy_driving_force = ufl.real(
        0.5 * ufl.inner(get_k_star_deriv(chi0_curr) * u0_pos, u0_pos) +
        0.5 * ufl.inner(get_k_star_deriv(chi0_curr) * u1_pos, u1_pos)
    )

    M_chi_form, R_damage_form = 0.0, 0.0
    for j in range(P_modes):
        M_chi_form += (eta / dt) * ufl.inner(chi_trial[j], chi_test[j]) * dx
        R_damage_form += (eta / dt) * ufl.inner(Chi_old.sub(j), chi_test[j]) * dx
        R_damage_form -= alpha * ufl.inner(grad(chi_curr_modes[j]), grad(chi_test[j])) * dx

        if j == 0:
            damage_source = ufl.conditional(omega - energy_driving_force < 0.0, omega - energy_driving_force, 0.0)
        else:
            damage_source = ufl.conditional(omega - energy_driving_force < 0.0, -0.25 * energy_driving_force * chi1_curr, 0.0)
        R_damage_form += ufl.inner(damage_source, chi_test[j]) * dx

    a_chi = fem.form(M_chi_form)
    L_chi = fem.form(R_damage_form)
    A_matrix = fem.petsc.create_matrix(a_chi)
    b_vector = fem.petsc.create_vector(V_chi)

    solver_chi = PETSc.KSP().create(domain.comm)
    solver_chi.setType(PETSc.KSP.Type.PREONLY)
    solver_chi.getPC().setType(PETSc.PC.Type.LU)
    solver_chi.getPC().setFactorSolverType("mumps")

    # Identical Brownian seed across both scenarios so differences come purely from g_1
    np.random.seed(2026)
    W_increments = np.random.normal(0.0, np.sqrt(dt), size=(total_steps, P_modes))

    chi_mean_out = fem.Function(V_chi_single)
    chi_std_out  = fem.Function(V_chi_single)

    t_log, d_mean_log, d_std_log = [], [], []

    for step in range(1, total_steps + 1):
        current_time = step * dt
        wave = np.sin(2.0 * np.pi * 2.5 * current_time)

        g0_const.value = np.array([35.0 * wave, 0.0, 45.0 * wave])
        if g1_active:
            g1_const.value = np.array([12.0 * wave, 0.0, 15.0 * wave])
        else:
            g1_const.value = np.array([0.0, 0.0, 0.0])

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

        chi0_vals = np.real(Chi_old.x.array[dofs_chi0])
        _, w1 = W_increments[step - 1]
        noise_increment = 0.35 * chi0_vals * w1
        b_vector.array[dofs_chi1] += (eta / dt) * noise_increment

        solver_chi.setOperators(A_matrix)
        solver_chi.solve(b_vector, Chi_modes.x.petsc_vec)
        Chi_modes.x.scatter_forward()

        chi0_new = np.minimum(np.clip(np.real(Chi_modes.x.array[dofs_chi0]), 0.0, 1.0), chi0_vals)
        Chi_modes.x.array[dofs_chi0] = chi0_new
        Chi_modes.x.array[dofs_chi1] = np.real(Chi_modes.x.array[dofs_chi1]) * chi0_new
        Chi_modes.x.scatter_forward()

        Chi_old.x.array[:] = Chi_modes.x.array[:]
        Chi_old.x.scatter_forward()

        chi_mean_out.x.array[:] = Chi_modes.sub(0).collapse().x.array[:]
        chi_std_out.x.array[:]  = Chi_modes.sub(1).collapse().x.array[:]

        g_mean = domain.comm.allreduce(np.real(np.mean(chi_mean_out.x.array)), op=MPI.SUM) / domain.comm.size
        g_std  = domain.comm.allreduce(np.real(np.mean(np.abs(chi_std_out.x.array))), op=MPI.SUM) / domain.comm.size

        t_log.append(current_time)
        d_mean_log.append(float(g_mean))
        d_std_log.append(float(g_std))

    solver_chi.destroy()
    A_matrix.destroy()
    b_vector.destroy()

    return np.array(t_log), np.array(d_mean_log), np.array(d_std_log)

# =============================================================================
# 2. EXECUTE SCENARIO A (g_1 = 0) & SCENARIO B (g_1 != 0) AND EXPORT CSV
# =============================================================================
time_arr, d_mean_A, d_std_A = run_scenario(g1_active=False)
_,        d_mean_B, d_std_B = run_scenario(g1_active=True)

if MPI.COMM_WORLD.rank == 0:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_matrix = np.column_stack((time_arr, d_mean_A, d_std_A, d_mean_B, d_std_B))
    csv_path = os.path.join(script_dir, "sfem_scenarios_A_vs_B.csv")
    np.savetxt(
        csv_path,
        out_matrix,
        delimiter=",",
        header="Time,Damage_Mean_g1_zero,Damage_Std_g1_zero,Damage_Mean_g1_nonzero,Damage_Std_g1_nonzero",
        comments=""
    )
    print(f"\nScenario comparison complete! Clean 5-column CSV saved to: {csv_path}")