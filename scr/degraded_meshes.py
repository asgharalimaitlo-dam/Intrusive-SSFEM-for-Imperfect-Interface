# =============================================================================
# PRISTINE vs. DEGRADED MESH SOLVER:degraded_meshes.py
# Runs the intrusive SSFEM-Uzawa solver and exports Pristine (t=0) and
# Degraded (t=tf) nodal coordinates, triangulation, U, and d to '.npz' & '.bp'.
# =============================================================================

import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem, io
import dolfinx.fem.petsc
import basix.ufl
import ufl
from ufl import grad

# =============================================================================
# 1. PARAMETERS & MESH INITIALIZATION
# =============================================================================
L, W = 100.0, 20.0
eta, omega, alpha = 50.0, 0.05, 0.1
dt = 0.002
target_steps = 200    # Active partial degradation state (t = 0.24 s)
penalty_val = 1e6
mu_0, lam_0 = 50.0, 70.0

C_ijk = np.zeros((2, 2, 2))
np.put(C_ijk, (0, 3, 5, 6), 1.0)

domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[np.array([0.0, 0.0]), np.array([L, W])],
    n=(50, 20),
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

el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
el_chi = basix.ufl.element("Lagrange", domain.basix_cell(), 1)

V_jump = fem.functionspace(domain, basix.ufl.mixed_element([el_u, el_u]))
V_chi  = fem.functionspace(domain, basix.ufl.mixed_element([el_chi, el_chi]))
V_p    = fem.functionspace(domain, basix.ufl.mixed_element([el_chi, el_chi]))

U_modes, Chi_modes, Chi_old, p_mult = fem.Function(V_jump), fem.Function(V_chi), fem.Function(V_chi), fem.Function(V_p)

# Pristine initial state with subtle central micro-flaw heterogeneity
Chi_modes.sub(0).interpolate(lambda x: 1.0 - 0.08 * np.exp(-((next(iter(x)) - L / 2.0)**2) / (2.0 * 14.0**2)))
Chi_modes.sub(1).interpolate(lambda x: 0.01 * np.ones(len(x.T)))
Chi_modes.x.scatter_forward()
Chi_old.x.array[:] = Chi_modes.x.array[:]; Chi_old.x.scatter_forward()

# Capture Pristine State (t = 0)
u_pristine = U_modes.sub(0).collapse()
u_pristine.name = "Displacement_Jump_Mean"
d_pristine = Chi_modes.sub(0).collapse()
d_pristine.name = "Interface_Integrity_d"

script_dir = os.path.dirname(os.path.abspath(__file__))
with io.VTXWriter(domain.comm, os.path.join(script_dir, "mesh_pristine_t0.bp"), [u_pristine, d_pristine], engine="BP4") as vtx0:
    vtx0.write(0.0)

d_vals_prist = np.real(d_pristine.x.array).copy()

# =============================================================================
# 2. EVOLVE TO DEGRADED INTERFACE STATE
# =============================================================================
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

u_prob = dolfinx.fem.petsc.NonlinearProblem(
    F_eq,
    U_modes,
    petsc_options={
        "snes_type": "newtonls", "snes_linesearch_type": "none", "snes_max_it": 10,
        "snes_atol": 1e-8, "snes_rtol": 1e-8,
        "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"
    },
    petsc_options_prefix="mesh_dist_"
)

chi_tr0, chi_tr1 = ufl.TrialFunctions(V_chi)
chi_te0, chi_te1 = ufl.TestFunctions(V_chi)
chi0_old_s, chi1_old_s = ufl.split(Chi_old)
chi0_c, chi1_c = chi_curr_modes

u0_v, u1_v = u_trial_modes
u0_p = ufl.as_vector([ufl.dot(u0_v, e_1), ufl.dot(u0_v, e_2), ufl.conditional(ufl.real(ufl.dot(u0_v, e_3)) < 0.0, 0.0, ufl.dot(u0_v, e_3))])
u1_p = ufl.as_vector([ufl.dot(u1_v, e_1), ufl.dot(u1_v, e_2), ufl.conditional(ufl.real(ufl.dot(u1_v, e_3)) < 0.0, 0.0, ufl.dot(u1_v, e_3))])
Y0 = ufl.real(0.5 * ufl.inner(get_k_star_deriv(chi0_c) * u0_p, u0_p) + 0.5 * ufl.inner(get_k_star_deriv(chi0_c) * u1_p, u1_p))

src0 = ufl.conditional(omega - Y0 < 0.0, omega - Y0, 0.0)
src1 = ufl.conditional(omega - Y0 < 0.0, -0.25 * Y0 * chi1_c, 0.0)

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

np.random.seed(2026)
for step in range(1, target_steps + 1):
    t_curr = step * dt
    w = np.sin(2.0 * np.pi * 2.5 * t_curr)
    g0_c.value = np.array([35.0 * w, 0.0, 45.0 * w])
    g1_c.value = np.array([4.0 * w, 0.0, 5.0 * w])

    p_mult.x.array[:] = 0.0
    for _ in range(4):
        u_prob.solve(); U_modes.x.scatter_forward()
        p_mult.x.array[dofs_p0] = np.minimum(0.0, np.real(p_mult.x.array[dofs_p0]) + penalty_val * np.real(U_modes.x.array[dofs_u0_z]))
        p_mult.x.array[dofs_p1] = np.real(p_mult.x.array[dofs_p1]) + penalty_val * np.real(U_modes.x.array[dofs_u1_z])
        p_mult.x.array[dofs_p1][np.real(p_mult.x.array[dofs_p0]) >= 0.0] = 0.0
        p_mult.x.scatter_forward()

    b_v.set(0.0); fem.petsc.assemble_vector(b_v, L_c); b_v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    c0_old = np.real(Chi_old.x.array[dofs_chi0])
    b_v.array[dofs_chi1] += (eta / dt) * (0.35 * c0_old * np.random.normal(0.0, np.sqrt(dt)))
    ksp_c.solve(b_v, Chi_modes.x.petsc_vec); Chi_modes.x.scatter_forward()

    c0_new = np.minimum(np.clip(np.real(Chi_modes.x.array[dofs_chi0]), 0.0, 1.0), c0_old)
    Chi_modes.x.array[dofs_chi0] = c0_new
    Chi_modes.x.array[dofs_chi1] = np.real(Chi_modes.x.array[dofs_chi1]) * c0_new
    Chi_modes.x.scatter_forward(); Chi_old.x.array[:] = Chi_modes.x.array[:]; Chi_old.x.scatter_forward()

# Capture Degraded State (t = target_steps * dt)
u_degraded = U_modes.sub(0).collapse()
u_degraded.name = "Displacement_Jump_Mean"
d_degraded = Chi_modes.sub(0).collapse()
d_degraded.name = "Interface_Integrity_d"

with io.VTXWriter(domain.comm, os.path.join(script_dir, "mesh_degraded_tf.bp"), [u_degraded, d_degraded], engine="BP4") as vtx1:
    vtx1.write(target_steps * dt)

ksp_c.destroy(); A_m.destroy(); b_v.destroy()

# =============================================================================
# 3. SAVE MESH TOPOLOGY & FIELD ARRAYS TO '.npz' FOR STANDALONE PLOTTING
# =============================================================================
if MPI.COMM_WORLD.rank == 0:
    coords = domain.geometry.x
    cells = domain.topology.connectivity(2, 0).array.reshape(-1, 3)
    u1_vals = np.real(u_degraded.sub(0).collapse().x.array)
    u2_vals = np.real(u_degraded.sub(1).collapse().x.array)
    u3_vals = np.real(u_degraded.sub(2).collapse().x.array)
    d_vals_deg = np.real(d_degraded.x.array)

    npz_out = os.path.join(script_dir, "mesh_snapshots_data.npz")
    np.savez(
        npz_out,
        coords=coords,
        cells=cells,
        d_prist=d_vals_prist,
        d_deg=d_vals_deg,
        u1_deg=u1_vals,
        u2_deg=u2_vals,
        u3_deg=u3_vals,
        t_final=target_steps * dt
    )
    print(f"Mesh snapshot arrays saved to: {npz_out}")
    print("ParaView ADIOS2 files saved: mesh_pristine_t0.bp and mesh_degraded_tf.bp")
