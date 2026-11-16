# p_refinement.py is used for refinement of hermite polynomials for p=1,2,3,4
import os
import numpy as np
from scipy.special import eval_hermitenorm, factorial
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem
import dolfinx.fem.petsc
import basix.ufl
import ufl
from ufl import grad

# =============================================================================
# 1. ORTHONORMAL HERMITE BASIS & AUTOMATIC C_ijk TENSOR GENERATOR
# =============================================================================
def eval_orthonormal_hermite(k, xi):
    return eval_hermitenorm(k, xi) / np.sqrt(factorial(k))

def compute_cijk_orthonormal(num_modes):
    gh_x, gh_w = np.polynomial.hermite.hermgauss(40)
    xi_q = np.sqrt(2.0) * gh_x
    w_q = gh_w / np.sqrt(np.pi)
    psi_vals = np.array([eval_orthonormal_hermite(k, xi_q) for k in range(num_modes)])
    C = np.einsum("q,iq,jq,kq->ijk", w_q, psi_vals, psi_vals, psi_vals)
    C[np.abs(C) < 1e-12] = 0.0
    return C

# =============================================================================
# 2. SHARED PROBLEM PARAMETERS & MESH
# =============================================================================
L, W = 100.0, 20.0
eta, omega, alpha = 50.0, 0.05, 0.1
dt = 0.004
total_steps = 100
penalty_val = 1e6
mu_0, lam_0 = 50.0, 70.0

domain = mesh.create_rectangle(
    comm=MPI.COMM_SELF,
    points=(np.array((0.0, 0.0)), np.array((L, W))),
    n=(20, 10),
    cell_type=mesh.CellType.triangle
)
dx = ufl.Measure("dx", domain=domain)

e_1 = ufl.as_vector((1.0, 0.0, 0.0))
e_2 = ufl.as_vector((0.0, 1.0, 0.0))
e_3 = ufl.as_vector((0.0, 0.0, 1.0))

def get_k_star(chi_val):
    d_eff = chi_val**2 + 1e-4
    return ufl.as_matrix(((mu_0 * d_eff, 0, 0), (0, mu_0 * d_eff, 0), (0, 0, (lam_0 + 2.0 * mu_0) * d_eff)))

def get_k_star_deriv(chi_val):
    return ufl.as_matrix(((2.0 * mu_0 * chi_val, 0, 0), (0, 2.0 * mu_0 * chi_val, 0), (0, 0, 2.0 * (lam_0 + 2.0 * mu_0) * chi_val)))

fast_opts = {
    "snes_type": "newtonls", "snes_linesearch_type": "none", "snes_max_it": 8,
    "snes_atol": 1e-8, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"
}

el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
el_chi = basix.ufl.element("Lagrange", domain.basix_cell(), 1)

# =============================================================================
# 3. HIGH-ACCURACY STOCHASTIC REFERENCE (25-PT GAUSS-HERMITE QUADRATURE)
# =============================================================================
print("Computing high-accuracy stochastic reference (25-pt Gauss-Hermite)...")

V_u_det = fem.functionspace(domain, el_u)
V_c_det = fem.functionspace(domain, el_chi)
u_det, c_det, c_old_det, p_det = fem.Function(V_u_det), fem.Function(V_c_det), fem.Function(V_c_det), fem.Function(V_c_det)
v_det, ct_tr, ct_te = ufl.TestFunction(V_u_det), ufl.TrialFunction(V_c_det), ufl.TestFunction(V_c_det)
g_det = fem.Constant(domain, PETSc.ScalarType((35.0, 0.0, 45.0)))

u3_d = ufl.dot(u_det, e_3)
aug_pd = ufl.conditional(ufl.real(p_det + penalty_val * u3_d) < 0.0, p_det + penalty_val * u3_d, 0.0)
F_det = ufl.inner(get_k_star(c_det) * u_det, v_det) * dx + aug_pd * ufl.inner(e_3, v_det) * dx - ufl.inner(g_det, v_det) * dx
u_prob_det = dolfinx.fem.petsc.NonlinearProblem(F_det, u_det, petsc_options=fast_opts, petsc_options_prefix="ref_gh_")

up_d = ufl.as_vector((ufl.dot(u_det, e_1), ufl.dot(u_det, e_2), ufl.conditional(ufl.real(u3_d) < 0.0, 0.0, u3_d)))
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

gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(25)
xi_pts = np.sqrt(2.0) * gh_nodes
prob_wts = gh_weights / np.sqrt(np.pi)
ref_trajectories = np.zeros((25, total_steps))

for q in range(25):
    xi_q = float(xi_pts[q])
    c_init = float(np.clip(0.95 + 0.02 * xi_q))
    u_det.x.array[:] = 0.0
    c_det.x.array[:] = c_init
    c_old_det.x.array[:] = c_init

    for step in range(1, total_steps + 1):
        t_curr = step * dt
        w = np.sin(2.0 * np.pi * 2.5 * t_curr)
        g_det.value = np.array(((35.0 + 5.0 * xi_q) * w, 0.0, (45.0 + 6.0 * xi_q) * w))

        p_det.x.array[:] = 0.0
        for _ in range(4):
            u_prob_det.solve()
            u3_vals = np.real(u_det.sub(2).collapse().x.array)
            p_det.x.array[:] = np.minimum(0.0, np.real(p_det.x.array) + penalty_val * u3_vals)

        b_d.set(0.0); fem.petsc.assemble_vector(b_d, L_d)
        ksp_d.solve(b_d, c_det.x.petsc_vec)

        c_old_arr = np.real(c_old_det.x.array).copy()
        c_new = np.minimum(np.clip(np.real(c_det.x.array), 1e-5, 1.0), c_old_arr)
        c_det.x.array[:] = c_new
        c_old_det.x.array[:] = c_new

        ref_trajectories[(q, step - 1)] = float(np.mean(c_new))

ksp_d.destroy(); A_d.destroy(); b_d.destroy()

ref_mean = np.sum(prob_wts[:, None] * ref_trajectories, axis=0)
ref_var  = np.sum(prob_wts[:, None] * (ref_trajectories - ref_mean[None, :])**2, axis=0)
ref_std  = np.sqrt(ref_var)
ref_m3   = np.sum(prob_wts[:, None] * (ref_trajectories - ref_mean[None, :])**3, axis=0)

# =============================================================================
# 4. STOCHASTIC p-REFINEMENT LOOP: p in {1, 2, 3, 4} (P_modes in {2, 3, 4, 5})
# =============================================================================
results_summary = []

gh_proj_x, gh_proj_w = np.polynomial.hermite.hermgauss(16)
xi_proj = np.sqrt(2.0) * gh_proj_x
w_proj = gh_proj_w / np.sqrt(np.pi)

for p_deg in (1, 2, 3, 4):
    M_modes = int(p_deg) + 1
    C_ijk = compute_cijk_orthonormal(M_modes)
    psi_proj = np.array([eval_orthonormal_hermite(m, xi_proj) for m in range(M_modes)])

    V_jump = fem.functionspace(domain, basix.ufl.mixed_element([el_u] * M_modes))
    V_chi  = fem.functionspace(domain, basix.ufl.mixed_element([el_chi] * M_modes))
    V_p    = fem.functionspace(domain, basix.ufl.mixed_element([el_chi] * M_modes))

    U_modes, Chi_modes, Chi_old, p_mult = fem.Function(V_jump), fem.Function(V_chi), fem.Function(V_chi), fem.Function(V_p)

    # Extract exact scalar DOF index arrays for each stochastic mode m = 0..p
    dofs_chi_parent = tuple(np.asarray(list(V_chi.sub(m).collapse()).pop(), dtype=np.int32).ravel() for m in range(M_modes))
    dofs_p_parent   = tuple(np.asarray(list(V_p.sub(m).collapse()).pop(), dtype=np.int32).ravel() for m in range(M_modes))

    # Initialize Mode 0 = 0.95 and Mode 1 = 0.02 cleanly via iterator
    chi_init_iter = iter(dofs_chi_parent)
    Chi_modes.x.array[:] = 0.0
    Chi_modes.x.array[next(chi_init_iter)] = 0.90
    Chi_modes.x.array[next(chi_init_iter)] = 0.03
    Chi_old.x.array[:] = Chi_modes.x.array[:]

    v_test_modes   = ufl.TestFunctions(V_jump)
    u_trial_modes  = ufl.split(U_modes)
    chi_curr_modes = ufl.split(Chi_modes)
    p_curr_modes   = ufl.split(p_mult)

    g0_c = fem.Constant(domain, PETSc.ScalarType((35.0, 0.0, 45.0)))
    g1_c = fem.Constant(domain, PETSc.ScalarType((5.0, 0.0, 6.0)))
    g_zero = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0)))
    g_stoch = (g0_c, g1_c) + (g_zero,) * (M_modes - 2)

        # Project exact quadratic stiffness factor d(xi)^2 + 1e-4 onto Hermite modes k = 0..p
    d2_funcs = tuple(fem.Function(V_c_det) for _ in range(M_modes))

    u0_3 = ufl.dot(next(iter(u_trial_modes)), e_3)
    p0_c = next(iter(p_curr_modes))
    contact_active_ufl = ufl.real(p0_c + penalty_val * u0_3) < 0.0

    F_eq = 0.0
    for k, (v_k, g_k, u_k, p_k) in enumerate(zip(v_test_modes, g_stoch, u_trial_modes, p_curr_modes)):
        F_eq -= ufl.inner(g_k, v_k) * dx
        uk3 = ufl.dot(u_k, e_3)
        aug_p_k = ufl.conditional(contact_active_ufl, p_k + penalty_val * uk3, 0.0)
        F_eq += aug_p_k * ufl.inner(e_3, v_k) * dx
        for i, u_i in enumerate(u_trial_modes):
            for j, d2_j in enumerate(d2_funcs):
                c_val = float(C_ijk[i, j, k])
                if c_val != 0.0:
                    K_mat_j = ufl.as_matrix(((mu_0 * d2_j, 0, 0), (0, mu_0 * d2_j, 0), (0, 0, (lam_0 + 2.0 * mu_0) * d2_j)))
                    F_eq += c_val * ufl.inner(K_mat_j * u_i, v_k) * dx

    u_prob = dolfinx.fem.petsc.NonlinearProblem(
        F_eq, U_modes, petsc_options=fast_opts, petsc_options_prefix=f"ssfem_p{p_deg}_"
    )


    chi_tr = ufl.TrialFunctions(V_chi)
    chi_te = ufl.TestFunctions(V_chi)
    chi_old_s = ufl.split(Chi_old)

    src_funcs = tuple(fem.Function(V_c_det) for _ in range(M_modes))

    a_form_expr = sum((eta / dt) * ufl.inner(tr_m, te_m) * dx for tr_m, te_m in zip(chi_tr, chi_te))
    L_form_expr = sum(
        (eta / dt) * ufl.inner(old_m, te_m) * dx
        - alpha * ufl.inner(grad(curr_m), grad(te_m)) * dx
        + ufl.inner(src_m, te_m) * dx
        for old_m, curr_m, te_m, src_m in zip(chi_old_s, chi_curr_modes, chi_te, src_funcs)
    )

    a_c = fem.form(a_form_expr)
    L_c = fem.form(L_form_expr)
    A_m = fem.petsc.assemble_matrix(a_c); A_m.assemble()
    b_v = fem.petsc.create_vector(V_chi)
    ksp_c = PETSc.KSP().create(domain.comm)
    ksp_c.setType("preonly"); ksp_c.getPC().setType("lu"); ksp_c.getPC().setFactorSolverType("mumps")
    ksp_c.setOperators(A_m)

    t_p_start = MPI.Wtime()
    p_mean_traj, p_std_traj, p_m3_traj = [], [], []

    for step in range(1, total_steps + 1):
        t_curr = step * dt
        w = np.sin(2.0 * np.pi * 2.5 * t_curr)
        g0_c.value = np.array((35.0 * w, 0.0, 45.0 * w))
        g1_c.value = np.array((5.0 * w, 0.0, 6.0 * w))
        chi_curr_nodal = np.array([np.real(Chi_modes.x.array[idx]) for idx in dofs_chi_parent])
        d2_q = np.clip(np.dot(chi_curr_nodal.T, psi_proj), 0.0, 1.0)**2 + 1e-4
        d2_proj_modes = np.einsum("nq,mq,q->mn", d2_q, psi_proj, w_proj)
        for d2_fn, d2_vals in zip(d2_funcs, d2_proj_modes):
            d2_fn.x.array[:] = d2_vals

        p_mult.x.array[:] = 0.0
        for _ in range(5):
            u_prob.solve()
            u3_nodal = np.array([np.real(U_modes.sub(m).sub(2).collapse().x.array) for m in range(M_modes)])
            p_nodal  = np.array([np.real(p_mult.x.array[idx]) for idx in dofs_p_parent])
            p_quad = np.dot(p_nodal.T, psi_proj) + penalty_val * np.dot(u3_nodal.T, psi_proj)
            p_clamped = np.minimum(0.0, p_quad)
            p_proj_modes = np.einsum("nq,mq,q->mn", p_clamped, psi_proj, w_proj)
            for parent_idx, p_m_vals in zip(dofs_p_parent, p_proj_modes):
                p_mult.x.array[parent_idx] = p_m_vals

        u1_nodal = np.array([np.real(U_modes.sub(m).sub(0).collapse().x.array) for m in range(M_modes)])
        u2_nodal = np.array([np.real(U_modes.sub(m).sub(1).collapse().x.array) for m in range(M_modes)])
        u3_nodal = np.array([np.real(U_modes.sub(m).sub(2).collapse().x.array) for m in range(M_modes)])
        chi_nodal_all = np.array([np.real(Chi_modes.x.array[idx]) for idx in dofs_chi_parent])

        u1_q = np.dot(u1_nodal.T, psi_proj)
        u2_q = np.dot(u2_nodal.T, psi_proj)
        u3_q = np.maximum(0.0, np.dot(u3_nodal.T, psi_proj))
        chi_q = np.clip(np.dot(chi_nodal_all.T, psi_proj), 0.0, 1.0)

        Y_q = chi_q * (mu_0 * (u1_q**2 + u2_q**2) + (lam_0 + 2.0 * mu_0) * (u3_q**2))
        src_q = np.minimum(0.0, omega - Y_q)
        src_proj_modes = np.einsum("nq,mq,q->mn", src_q, psi_proj, w_proj)
        for src_fn, s_vals in zip(src_funcs, src_proj_modes):
            src_fn.x.array[:] = s_vals

        b_v.set(0.0); fem.petsc.assemble_vector(b_v, L_c)
        ksp_c.solve(b_v, Chi_modes.x.petsc_vec)

        chi_new_nodal = np.array([np.real(Chi_modes.x.array[idx]) for idx in dofs_chi_parent])
        chi_old_nodal = np.array([np.real(Chi_old.x.array[idx]) for idx in dofs_chi_parent])
        d_new_q = np.minimum(np.clip(np.dot(chi_new_nodal.T, psi_proj), 1e-5, 1.0), np.dot(chi_old_nodal.T, psi_proj))
        # Mild 4th-order spectral filter (Hou-Li) suppresses Gibbs ringing across non-smooth irreversibility corner
        modal_filter = np.array([np.exp(-0.15 * (m / max(1, p_deg))**4) for m in range(M_modes)])
        d_proj_modes = np.einsum("nq,mq,q->mn", d_new_q, psi_proj, w_proj) * modal_filter[:, None]
 
        for parent_idx, d_m_vals in zip(dofs_chi_parent, d_proj_modes):
            Chi_modes.x.array[parent_idx] = d_m_vals
        Chi_old.x.array[:] = Chi_modes.x.array[:]

        d_spatial_mean_q = np.mean(d_new_q, axis=0)
        mu_step = float(np.sum(w_proj * d_spatial_mean_q))
        var_step = float(np.sum(w_proj * (d_spatial_mean_q - mu_step)**2))
        m3_step = float(np.sum(w_proj * (d_spatial_mean_q - mu_step)**3))

        p_mean_traj.append(mu_step)
        p_std_traj.append(np.sqrt(max(0.0, var_step)))
        p_m3_traj.append(m3_step)

    t_p_elapsed = MPI.Wtime() - t_p_start
    ksp_c.destroy(); A_m.destroy(); b_v.destroy()

    p_mean_arr = np.array(p_mean_traj)
    p_std_arr  = np.array(p_std_traj)
    p_m3_arr   = np.array(p_m3_traj)

    err_mean = np.linalg.norm(p_mean_arr - ref_mean) / np.linalg.norm(ref_mean) * 100.0
    err_std  = np.linalg.norm(p_std_arr - ref_std) / np.linalg.norm(ref_std) * 100.0
    err_m3   = np.linalg.norm(p_m3_arr - ref_m3) / (np.linalg.norm(ref_m3) + 1e-12) * 100.0

    results_summary.append((p_deg, M_modes, err_mean, err_std, err_m3, t_p_elapsed))
    print(f"p = {p_deg} ({M_modes} modes) | Mean Err: {err_mean:.3f}% | Std Err: {err_std:.3f}% | 3rd-Moment Skew Err: {err_m3:.2f}% | Time: {t_p_elapsed:.2f} s")

# =============================================================================
# 5. EXPORT CSV DATASET ONLY
# =============================================================================
res_mat = np.array(results_summary)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_out = os.path.join(script_dir, "sfem_p_refinement.csv")
np.savetxt(
    csv_out, res_mat, delimiter=",",
    header="p_order,num_modes,Mean_Rel_Err_Pct,Std_Rel_Err_Pct,Skew_M3_Rel_Err_Pct,Wall_Time_s",
    comments=""
)
print(f"\nSaved p-refinement dataset to: {csv_out}")
