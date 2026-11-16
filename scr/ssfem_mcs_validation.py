# fine name ssfem_mcs_validation.py is used for validation of ssfem against 500 sample mcs 
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem
import dolfinx.fem.petsc
import basix.ufl
import ufl
from ufl import grad

# =============================================================================
# 1. CONFIGURATION SWITCH: SET True FOR 15-SEC CHECK, False FOR FULL 500-MCS
# =============================================================================
TEST_MODE = False

if TEST_MODE:
    N_mcs = 20
    dt = 0.004
    total_steps = 100    
else:
    N_mcs = 500
    dt = 0.002
    total_steps = 400     

L, W = 100.0, 20.0
eta, omega, alpha = 50.0, 0.05, 0.1
penalty_val = 1e6
mu_0, lam_0 = 50.0, 70.0

C_ijk = np.zeros((2, 2, 2))
np.put(C_ijk, (0, 3, 5, 6), 1.0)

domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[np.array([0.0, 0.0]), np.array([L, W])],
    n=(20, 10),
    cell_type=mesh.CellType.triangle
)
dx = ufl.Measure("dx", domain=domain)

e_1 = ufl.as_vector([1.0, 0.0, 0.0])
e_2 = ufl.as_vector([0.0, 1.0, 0.0])
e_3 = ufl.as_vector([0.0, 0.0, 1.0])

def get_k_star(chi_val):
    d_eff = chi_val**2 + 1e-4
    return ufl.as_matrix([[mu_0 * d_eff, 0, 0], [0, mu_0 * d_eff, 0], [0, 0, (lam_0 + 2.0 * mu_0) * d_eff]])

def get_k_star_deriv(chi_val):
    return ufl.as_matrix([[2.0 * mu_0 * chi_val, 0, 0], [0, 2.0 * mu_0 * chi_val, 0], [0, 0, 2.0 * (lam_0 + 2.0 * mu_0) * chi_val]])

fast_opts = {
    "snes_type": "newtonls", "snes_linesearch_type": "none", "snes_max_it": 8,
    "snes_atol": 1e-7, "snes_rtol": 1e-7,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"
}

if MPI.COMM_WORLD.rank == 0:
    mode_label = "QUICK SANITY CHECK" if TEST_MODE else "FULL 500-SAMPLE PRODUCTION RUN"
    print(f"\nStarting {mode_label}: N_mcs={N_mcs}, total_steps={total_steps}, dt={dt}")

# =============================================================================
# 2. PART A: INTRUSIVE SSFEM-UZAWA SOLVE (SINGLE COUPLED PASS)
# =============================================================================
el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
el_chi = basix.ufl.element("Lagrange", domain.basix_cell(), 1)

V_jump = fem.functionspace(domain, basix.ufl.mixed_element([el_u, el_u]))
V_chi  = fem.functionspace(domain, basix.ufl.mixed_element([el_chi, el_chi]))
V_p    = fem.functionspace(domain, basix.ufl.mixed_element([el_chi, el_chi]))

U_modes, Chi_modes, Chi_old, p_mult = fem.Function(V_jump), fem.Function(V_chi), fem.Function(V_chi), fem.Function(V_p)
Chi_modes.sub(0).interpolate(lambda x: np.ones(len(x.T)))
Chi_modes.sub(1).interpolate(lambda x: 0.01 * np.ones(len(x.T)))
Chi_modes.x.scatter_forward()
Chi_old.x.array[:] = Chi_modes.x.array[:]; Chi_old.x.scatter_forward()

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

g0_c = fem.Constant(domain, PETSc.ScalarType((35.0, 0.0, 45.0)))
g1_c = fem.Constant(domain, PETSc.ScalarType((4.0, 0.0, 5.0)))
g_stoch = [g0_c, g1_c]

F_eq = 0.0
for k in range(2):
    v_k = v_test_modes[k]
    F_eq -= ufl.inner(g_stoch[k], v_k) * dx
    uk3 = ufl.dot(u_trial_modes[k], e_3)
    aug_p = ufl.conditional(ufl.real(p_curr_modes[k] + penalty_val * uk3) < 0.0, p_curr_modes[k] + penalty_val * uk3, 0.0)
    F_eq += aug_p * ufl.inner(e_3, v_k) * dx
    for i in range(2):
        for j in range(2):
            if C_ijk[i, j, k] > 0.0:
                F_eq += C_ijk[i, j, k] * ufl.inner(get_k_star(chi_curr_modes[j]) * u_trial_modes[i], v_k) * dx

u_prob = dolfinx.fem.petsc.NonlinearProblem(F_eq, U_modes, petsc_options=fast_opts, petsc_options_prefix="ssfem_fin_")

chi_tr0, chi_tr1 = ufl.TrialFunctions(V_chi)
chi_te0, chi_te1 = ufl.TestFunctions(V_chi)
chi0_old_s, chi1_old_s = ufl.split(Chi_old)
chi0_c, chi1_c = chi_curr_modes

u0_v, u1_v = u_trial_modes
u0_p = ufl.as_vector([ufl.dot(u0_v, e_1), ufl.dot(u0_v, e_2), ufl.conditional(ufl.real(ufl.dot(u0_v, e_3)) < 0.0, 0.0, ufl.dot(u0_v, e_3))])
u1_p = ufl.as_vector([ufl.dot(u1_v, e_1), ufl.dot(u1_v, e_2), ufl.conditional(ufl.real(ufl.dot(u1_v, e_3)) < 0.0, 0.0, ufl.dot(u1_v, e_3))])

Y0 = ufl.real(0.5 * ufl.inner(get_k_star_deriv(chi0_c) * u0_p, u0_p) + 0.5 * ufl.inner(get_k_star_deriv(chi0_c) * u1_p, u1_p))
Y1 = ufl.real(ufl.inner(get_k_star_deriv(chi0_c) * u0_p, u1_p) + 0.5 * ufl.inner(get_k_star_deriv(chi1_c) * u0_p, u0_p))

thresh_cond = omega - (Y0 + 0.35 * ufl.real(ufl.sqrt(Y1**2 + 1e-12))) < 0.0
src0 = ufl.conditional(thresh_cond, omega - Y0, 0.0)
src1 = ufl.conditional(thresh_cond, -Y1, 0.0)



a_c = fem.form((eta / dt) * (ufl.inner(chi_tr0, chi_te0) + ufl.inner(chi_tr1, chi_te1)) * dx)
L_c = fem.form(
    (eta / dt) * (ufl.inner(chi0_old_s, chi_te0) + ufl.inner(chi1_old_s, chi_te1)) * dx
    - alpha * (ufl.inner(grad(chi0_c), grad(chi_te0)) + ufl.inner(grad(chi1_c), grad(chi_te1))) * dx
    + (ufl.inner(src0, chi_te0) + ufl.inner(src1, chi_te1)) * dx
)

A_m = fem.petsc.assemble_matrix(a_c); A_m.assemble()
b_v = fem.petsc.create_vector(V_chi)
ksp_c = PETSc.KSP().create(domain.comm)
ksp_c.setType("preonly"); ksp_c.getPC().setType("lu"); ksp_c.getPC().setFactorSolverType("mumps")
ksp_c.setOperators(A_m)

t_ssfem_start = MPI.Wtime()
ssfem_time, ssfem_mean, ssfem_std = [], [], []

for step in range(1, total_steps + 1):
    t_curr = step * dt
    w = np.sin(2.0 * np.pi * 2.5 * t_curr)
    g0_c.value = np.array([35.0 * w, 0.0, 45.0 * w])
    g1_c.value = np.array([4.0 * w, 0.0, 5.0 * w])

    p_mult.x.array[:] = 0.0
    for _ in range(3):
        u_prob.solve(); U_modes.x.scatter_forward()
        p_mult.x.array[dofs_p0] = np.minimum(0.0, np.real(p_mult.x.array[dofs_p0]) + penalty_val * np.real(U_modes.x.array[dofs_u0_z]))
        p_mult.x.array[dofs_p1] = np.real(p_mult.x.array[dofs_p1]) + penalty_val * np.real(U_modes.x.array[dofs_u1_z])
        p_mult.x.array[dofs_p1][np.real(p_mult.x.array[dofs_p0]) >= 0.0] = 0.0
        p_mult.x.scatter_forward()

    b_v.set(0.0); fem.petsc.assemble_vector(b_v, L_c); b_v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    ksp_c.solve(b_v, Chi_modes.x.petsc_vec); Chi_modes.x.scatter_forward()

    c0_old = np.real(Chi_old.x.array[dofs_chi0])
    c0_new = np.minimum(np.clip(np.real(Chi_modes.x.array[dofs_chi0]), 1e-5, 1.0), c0_old)
    Chi_modes.x.array[dofs_chi0] = c0_new
    max_std_bound = np.minimum(c0_new, 1.0 - c0_new + 0.01)
    Chi_modes.x.array[dofs_chi1] = np.clip(np.real(Chi_modes.x.array[dofs_chi1]), -max_std_bound, max_std_bound)
    Chi_modes.x.scatter_forward()

    Chi_old.x.array[:] = Chi_modes.x.array[:]; Chi_old.x.scatter_forward()

    ssfem_time.append(t_curr)
    ssfem_mean.append(domain.comm.allreduce(np.real(np.mean(Chi_modes.sub(0).collapse().x.array)), op=MPI.SUM) / domain.comm.size)
    ssfem_std.append(domain.comm.allreduce(np.real(np.mean(np.abs(Chi_modes.sub(1).collapse().x.array))), op=MPI.SUM) / domain.comm.size)

t_ssfem_elapsed = MPI.Wtime() - t_ssfem_start
ksp_c.destroy(); A_m.destroy(); b_v.destroy()

if MPI.COMM_WORLD.rank == 0:
    print(f"Intrusive SSFEM Solve Completed in {t_ssfem_elapsed:.3f} s")

# =============================================================================
# 3. PART B: NON-INTRUSIVE MONTE CARLO BENCHMARK (MASS-CONSISTENT ENSEMBLE)
# =============================================================================
V_u_det = fem.functionspace(domain, el_u)
V_c_det = fem.functionspace(domain, el_chi)
u_det, c_det, c_old_det, p_det = fem.Function(V_u_det), fem.Function(V_c_det), fem.Function(V_c_det), fem.Function(V_c_det)
v_det, ct_tr, ct_te = ufl.TestFunction(V_u_det), ufl.TrialFunction(V_c_det), ufl.TestFunction(V_c_det)
g_det = fem.Constant(domain, PETSc.ScalarType((35.0, 0.0, 45.0)))

u3_d = ufl.dot(u_det, e_3)
aug_pd = ufl.conditional(ufl.real(p_det + penalty_val * u3_d) < 0.0, p_det + penalty_val * u3_d, 0.0)
F_det = ufl.inner(get_k_star(c_det) * u_det, v_det) * dx + aug_pd * ufl.inner(e_3, v_det) * dx - ufl.inner(g_det, v_det) * dx
u_prob_det = dolfinx.fem.petsc.NonlinearProblem(F_det, u_det, petsc_options=fast_opts, petsc_options_prefix="mcs_fin_")

up_d = ufl.as_vector([ufl.dot(u_det, e_1), ufl.dot(u_det, e_2), ufl.conditional(ufl.real(u3_d) < 0.0, 0.0, u3_d)])
Yd = ufl.real(0.5 * ufl.inner(get_k_star_deriv(c_det) * up_d, up_d))

a_d = fem.form((eta / dt) * ufl.inner(ct_tr, ct_te) * dx)
L_d = fem.form(
    ufl.inner((eta / dt) * c_old_det + ufl.conditional(omega - Yd < 0.0, omega - Yd, 0.0), ct_te) * dx
    - alpha * ufl.inner(grad(c_det), grad(ct_te)) * dx
)

A_d = fem.petsc.assemble_matrix(a_d); A_d.assemble()
b_d = fem.petsc.create_vector(V_c_det)
ksp_d = PETSc.KSP().create(domain.comm)
ksp_d.setType("preonly"); ksp_d.getPC().setType("lu"); ksp_d.getPC().setFactorSolverType("mumps")
ksp_d.setOperators(A_d)

np.random.seed(42)
xi_samples = np.random.normal(0.0, 1.0, size=N_mcs)
mcs_trajectories = np.zeros((N_mcs, total_steps))
report_interval = 5 if TEST_MODE else 50

t_mcs_start = MPI.Wtime()
for m in range(N_mcs):
    xi_m = xi_samples[m]
    c_init = np.clip(1.0 + 0.01 * xi_m, 1e-5, 1.0)
    u_det.x.array[:] = 0.0; u_det.x.scatter_forward()
    c_det.x.array[:] = c_init; c_det.x.scatter_forward()
    c_old_det.x.array[:] = c_init; c_old_det.x.scatter_forward()

    for step in range(1, total_steps + 1):
        t_curr = step * dt
        w = np.sin(2.0 * np.pi * 2.5 * t_curr)
        g_det.value = np.array([(35.0 + 4.0 * xi_m) * w, 0.0, (45.0 + 5.0 * xi_m) * w])

        p_det.x.array[:] = 0.0
        for _ in range(3):
            u_prob_det.solve(); u_det.x.scatter_forward()
            u3_vals = np.real(u_det.sub(2).collapse().x.array)
            p_det.x.array[:] = np.minimum(0.0, np.real(p_det.x.array) + penalty_val * u3_vals)
            p_det.x.scatter_forward()

        b_d.set(0.0); fem.petsc.assemble_vector(b_d, L_d); b_d.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        ksp_d.solve(b_d, c_det.x.petsc_vec); c_det.x.scatter_forward()

        c_old_arr = np.real(c_old_det.x.array)
        c_new = np.minimum(np.clip(np.real(c_det.x.array), 1e-5, 1.0), c_old_arr)
        c_det.x.array[:] = c_new; c_det.x.scatter_forward()
        c_old_det.x.array[:] = c_new; c_old_det.x.scatter_forward()

        mcs_trajectories[m, step - 1] = domain.comm.allreduce(np.mean(c_new), op=MPI.SUM) / domain.comm.size

    if (m + 1) % report_interval == 0 and MPI.COMM_WORLD.rank == 0:
        elapsed = MPI.Wtime() - t_mcs_start
        est_rem = (elapsed / (m + 1)) * (N_mcs - (m + 1))
        print(f"MCS Progress: {m + 1}/{N_mcs} completed ({elapsed:.1f} s elapsed, ~{est_rem:.1f} s remaining)")

t_mcs_elapsed = MPI.Wtime() - t_mcs_start
ksp_d.destroy(); A_d.destroy(); b_d.destroy()

if MPI.COMM_WORLD.rank == 0:
    mcs_mean = np.mean(mcs_trajectories, axis=0)
    mcs_std  = np.std(mcs_trajectories, axis=0)
    ssfem_mean = np.array(ssfem_mean)
    ssfem_std  = np.array(ssfem_std)

    rel_l2_err = np.linalg.norm(ssfem_mean - mcs_mean) / np.linalg.norm(mcs_mean) * 100.0
    speedup = t_mcs_elapsed / t_ssfem_elapsed

    print(f"\n=== INTRUSIVE SSFEM vs. {N_mcs}-SAMPLE MCS VALIDATION ===")
    print(f"Intrusive SSFEM Solve Time: {t_ssfem_elapsed:.3f} s")
    print(f"{N_mcs}-Sample MCS Solve Time:  {t_mcs_elapsed:.3f} s")
    print(f"Computational Speedup:      {speedup:.1f}x faster")
    print(f"Relative L2 Mean Error:     {rel_l2_err:.3f}%")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_mat = np.column_stack((ssfem_time, ssfem_mean, ssfem_std, mcs_mean, mcs_std))
    csv_out = os.path.join(script_dir, "ssfem_vs_mcs_500.csv")
    np.savetxt(
        csv_out,
        out_mat,
        delimiter=",",
        header="Time,SSFEM_Mean,SSFEM_Std,MCS_Mean,MCS_Std",
        comments=""
    )
    print(f"Saved validation CSV to: {csv_out}")
