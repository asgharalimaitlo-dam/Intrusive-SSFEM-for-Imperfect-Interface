# =============================================================================
# MASTER SIMULATION ENGINE: main_ssfem_interface.py
# TITLE: Asymptotic and Spectral Stochastic Finite Element Modeling of Imperfect Interfaces via Uzawa Augmented Lagrangian
# AUTHORS: Asghar Maitlo, Frédéric Lebon, and Caroline Bauzet
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
from ufl import inner, grad, dx

# =============================================================================
# 1. PROBLEM PARAMETERS & STOCHASTIC PC CONFIGURATION
# =============================================================================
global_start_time = MPI.Wtime()

L, W = 100.0, 20.0   # Length and width of interface (mm)
P_modes = 2          # Mode 0: Expected Mean, Mode 1: Stochastic Volatility

E = 70000.0          # Bulk Young's modulus (MPa)
nu = 0.3             # Poisson's ratio
eta = 50          # Interfacial damage viscosity (MPa*s*mm)
omega = 0.05         # Specific adhesion fracture threshold (MPa*mm)
alpha = 0.1          # Non-local damage gradient regularization (MPa*mm^3)
dt = 0.002           # Time step size (s) -> 400 * 0.002 = 0.8s
total_steps = 400    # Total transient steps
penalty_val = 1e6    # Augmented Lagrangian penalty parameter (MPa/mm)

# Third-order Hermite Polynomial Chaos coupling tensor C_ijk = E[Psi_i Psi_j Psi_k]
C_ijk = np.zeros((P_modes, P_modes, P_modes))
C_ijk[0, 0, 0] = 1.0
C_ijk[0, 1, 1] = 1.0
C_ijk[1, 0, 1] = 1.0
C_ijk[1, 1, 0] = 1.0

# =============================================================================
# 2. MESH GENERATION & INTRUSIVE SSFEM FUNCTION SPACES
# =============================================================================
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, 
    points=[np.array([0.0, 0.0]), np.array([L, W])],
    n=[50, 20], 
    cell_type=mesh.CellType.triangle
)

# Displacement jump space V_jump = [P1]^3 \otimes V_P
el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
el_pce_u = basix.ufl.mixed_element([el_u] * P_modes)
V_jump = fem.functionspace(domain, el_pce_u)

# Scalar damage space V_chi and Lagrange multiplier space V_p = P1 \otimes V_P
el_chi = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
el_pce_chi = basix.ufl.mixed_element([el_chi] * P_modes)
V_chi = fem.functionspace(domain, el_pce_chi)
V_p   = fem.functionspace(domain, el_pce_chi)

U_modes   = fem.Function(V_jump)      
Chi_modes = fem.Function(V_chi)    
Chi_old   = fem.Function(V_chi)      
p_mult    = fem.Function(V_p)    # Exact SSFEM Lagrange multiplier modes [p_0, p_1]

# Initialize Mode 0 (Mean = 1.0) and Mode 1 (Initial micro-defect volatility = 0.01)
Chi_modes.sub(0).interpolate(lambda x: np.ones(x.shape[1]))
Chi_modes.sub(1).interpolate(lambda x: 0.01 * np.ones(x.shape[1]))
Chi_modes.x.scatter_forward()

Chi_old.x.array[:] = Chi_modes.x.array[:]
Chi_old.x.scatter_forward()

# Extract exact interleaved DOF maps for parallel subspaces
V_u_single, _     = V_jump.sub(0).collapse()
_, dofs_u0_z      = V_jump.sub(0).sub(2).collapse()
_, dofs_u1_z      = V_jump.sub(1).sub(2).collapse()
V_chi_single, dofs_chi0 = V_chi.sub(0).collapse()
_, dofs_chi1      = V_chi.sub(1).collapse()
_, dofs_p0        = V_p.sub(0).collapse()
_, dofs_p1        = V_p.sub(1).collapse()

# =============================================================================
# 3. WEAK FORMULATION WITH AUGMENTED LAGRANGIAN SIGNORINI CONTACT
# =============================================================================
v_test_modes   = ufl.TestFunctions(V_jump)
u_trial_modes  = ufl.split(U_modes)
chi_curr_modes = ufl.split(Chi_modes)
p_curr_modes   = ufl.split(p_mult)

F_equilibrium = 0.0

# Compliant adhesive layer reference stiffnesses (MPa/mm)
mu_0 = 50.0  
lam_0 = 70.0 

def get_k_star(chi_val):
    k_normal = (lam_0 + 2.0 * mu_0) * (chi_val**2)
    k_shear = mu_0 * (chi_val**2)
    return ufl.as_matrix([[k_shear, 0, 0], [0, k_shear, 0], [0, 0, k_normal]])

def get_k_star_deriv(chi_val):
    k_normal_d = 2.0 * (lam_0 + 2.0 * mu_0) * chi_val
    k_shear_d = 2.0 * mu_0 * chi_val
    return ufl.as_matrix([[k_shear_d, 0, 0], [0, k_shear_d, 0], [0, 0, k_normal_d]])

# Mixed cyclic shear (x1) and normal (x3) boundary traction constants
g_stochastic = [
    fem.Constant(domain, PETSc.ScalarType((35.0, 0.0, 45.0))), 
    fem.Constant(domain, PETSc.ScalarType((4.0, 0.0, 5.0)))   
]

e_3 = ufl.as_vector([0.0, 0.0, 1.0])

for k in range(P_modes):
    v_k = v_test_modes[k]
    F_equilibrium -= ufl.inner(g_stochastic[k], v_k) * dx
    
    # Intrusive SSFEM Augmented Lagrangian Signorini contact term for mode k
    # Complex-safe: u_trial_modes[k][2] is real-valued, and v_k is the 2nd arg of ufl.inner
        # Intrusive SSFEM Augmented Lagrangian Signorini contact term for mode k
    # Wrap ONLY the '< 0.0' condition in ufl.real() so complex comparison succeeds
    # while keeping the value branch differentiable!
    aug_p_k = ufl.conditional(
        ufl.real(p_curr_modes[k] + penalty_val * u_trial_modes[k][2]) < 0.0,
        p_curr_modes[k] + penalty_val * u_trial_modes[k][2], 
        0.0
    )
    F_equilibrium += aug_p_k * ufl.inner(e_3, v_k) * dx
    
    for i in range(P_modes):
        u_i = u_trial_modes[i]
        for j in range(P_modes):
            if C_ijk[i, j, k] > 0.0:
                K_star_j = get_k_star(chi_curr_modes[j])
                F_equilibrium += C_ijk[i, j, k] * ufl.inner(K_star_j * u_i, v_k) * dx

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

u_problem = dolfinx.fem.petsc.NonlinearProblem(
    F_equilibrium, 
    U_modes, 
    petsc_options=petsc_options, 
    petsc_options_prefix="sfem_interface_"
)

# =============================================================================
# 4. DAMAGE EVOLUTION SOLVER COMPILATION (OUTSIDE TIME LOOP)
# =============================================================================
chi_trial = ufl.TrialFunctions(V_chi)
chi_test  = ufl.TestFunctions(V_chi)

M_chi_form = 0.0
R_damage_form = 0.0

np.random.seed(2026)
W_increments = np.random.normal(0.0, np.sqrt(dt), size=(total_steps, P_modes))

# Positive jump vector [[u]]_+ clips normal jump under compression
u_0_pos = ufl.as_vector([
    u_trial_modes[0][0], 
    u_trial_modes[0][1], 
    ufl.conditional(ufl.real(u_trial_modes[0][2]) < 0.0, 0.0, u_trial_modes[0][2])
])
energy_driving_force = ufl.real(0.5 * ufl.inner(get_k_star_deriv(chi_curr_modes[0]) * u_0_pos, u_0_pos))

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

# Pre-allocate persistent PETSc solver and matrices for maximum speed
A_matrix = fem.petsc.create_matrix(a_chi)
b_vector = fem.petsc.create_vector(V_chi)
solver_chi = PETSc.KSP().create(domain.comm)
solver_chi.setType(PETSc.KSP.Type.PREONLY)
solver_chi.getPC().setType(PETSc.PC.Type.LU)
solver_chi.getPC().setFactorSolverType("mumps")

# =============================================================================
# 5. TIME INTEGRATION PROCESS LOOP WITH UZAWA LAGRANGE MULTIPLIER RECOVERY
# =============================================================================
u_mean_out   = fem.Function(V_u_single, name="U_Jump_Mean")
u_std_out    = fem.Function(V_u_single, name="U_Jump_Volatility")
chi_mean_out = fem.Function(V_chi_single, name="Damage_Mean")
chi_std_out  = fem.Function(V_chi_single, name="Damage_Volatility")
p_mean_out   = fem.Function(V_chi_single, name="Contact_Pressure_p0")

vtf_u_mean   = io.VTXWriter(domain.comm, "out_u_mean.bp", [u_mean_out], engine="BP4")
vtf_u_std    = io.VTXWriter(domain.comm, "out_u_std.bp", [u_std_out], engine="BP4")
vtf_chi_mean = io.VTXWriter(domain.comm, "out_chi_mean.bp", [chi_mean_out], engine="BP4")
vtf_chi_std  = io.VTXWriter(domain.comm, "out_chi_std.bp", [chi_std_out], engine="BP4")

extracted_time = []
extracted_damage_mean = []
extracted_damage_volatility = []
extracted_shear_jump = []
extracted_normal_jump = []
extracted_contact_pressure = []

if MPI.COMM_WORLD.rank == 0:
    print("Starting Intrusive SSFEM Incremental Simulation Pipeline...")

def write_vtf_step(t_val):
    u_mean_out.x.array[:]   = U_modes.sub(0).collapse().x.array[:]
    u_std_out.x.array[:]    = U_modes.sub(1).collapse().x.array[:]
    chi_mean_out.x.array[:] = Chi_modes.sub(0).collapse().x.array[:]
    chi_std_out.x.array[:]  = Chi_modes.sub(1).collapse().x.array[:]
    p_mean_out.x.array[:]   = p_mult.sub(0).collapse().x.array[:]
    
    vtf_u_mean.write(t_val)
    vtf_u_std.write(t_val)
    vtf_chi_mean.write(t_val)
    vtf_chi_std.write(t_val)

write_vtf_step(0.0)
loop_start_time = MPI.Wtime()

for step in range(1, total_steps + 1):
    current_time = step * dt
    
    # Mixed cyclic shear (x1) and normal traction/compression (x3) loading
    wave = np.sin(2.0 * np.pi * 2.5 * current_time)
    g_stochastic[0].value = np.array([35.0 * wave, 0.0, 45.0 * wave])
    g_stochastic[1].value = np.array([4.0 * wave, 0.0, 5.0 * wave])
    
    # Uzawa Augmented Lagrangian loop: recovers exact SSFEM Lagrange multipliers [p_0, p_1]
    p_mult.x.array[:] = 0.0
    for uzawa_iter in range(5):
        u_problem.solve()
        U_modes.x.scatter_forward()
        
        p_mult.x.array[dofs_p0] = np.minimum(
            0.0, 
            np.real(p_mult.x.array[dofs_p0]) + penalty_val * np.real(U_modes.x.array[dofs_u0_z])
        )
        p_mult.x.array[dofs_p1] = (
            np.real(p_mult.x.array[dofs_p1]) + penalty_val * np.real(U_modes.x.array[dofs_u1_z])
        )
        # Zero out Mode 1 pressure volatility wherever Mode 0 contact is open (p_0 == 0)
        inactive_mask = (np.real(p_mult.x.array[dofs_p0]) >= 0.0)
        p_mult.x.array[dofs_p1][inactive_mask] = 0.0
        p_mult.x.scatter_forward()
    
    # Assemble and solve implicit backward Euler-Maruyama damage step
    A_matrix.zeroEntries()
    fem.petsc.assemble_matrix(A_matrix, a_chi)
    A_matrix.assemble()

    
    b_vector.set(0.0)
    fem.petsc.assemble_vector(b_vector, L_chi)
    b_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # Inject multiplicative Itô noise into interleaved Mode 1 DOFs
    chi0_vals = np.real(Chi_old.x.array[dofs_chi0])
    noise_increment = 0.35 * chi0_vals * W_increments[step - 1, 1]
    b_vector.array[dofs_chi1] += (eta / dt) * noise_increment
        
    solver_chi.setOperators(A_matrix)
    solver_chi.solve(b_vector, Chi_modes.x.petsc_vec)
    Chi_modes.x.scatter_forward()
    
    # Enforce thermodynamic bounds and irreversibility on Mode 0, scale Mode 1 multiplicatively
    chi0_new = np.clip(np.real(Chi_modes.x.array[dofs_chi0]), 0.0, 1.0)
    chi0_new = np.minimum(chi0_new, chi0_vals)
    Chi_modes.x.array[dofs_chi0] = chi0_new
    Chi_modes.x.array[dofs_chi1] = np.real(Chi_modes.x.array[dofs_chi1]) * chi0_new
    Chi_modes.x.scatter_forward()
    
    Chi_old.x.array[:] = Chi_modes.x.array[:]
    Chi_old.x.scatter_forward()
    
    write_vtf_step(current_time)
    
    # Compute global spatial averages across MPI ranks
    u_array_reshaped  = np.real(u_mean_out.x.array.reshape(-1, 3))
    local_shear_mean  = np.mean(u_array_reshaped[:, 0])  
    local_normal_mean = np.mean(u_array_reshaped[:, 2])  
    local_damage_mean = np.real(np.mean(chi_mean_out.x.array))
    local_damage_std  = np.real(np.mean(np.abs(chi_std_out.x.array)))
    local_p0_mean     = np.real(np.mean(p_mean_out.x.array))
    
    global_u1  = domain.comm.allreduce(local_shear_mean, op=MPI.SUM) / domain.comm.size
    global_u3  = domain.comm.allreduce(local_normal_mean, op=MPI.SUM) / domain.comm.size
    global_chi = domain.comm.allreduce(local_damage_mean, op=MPI.SUM) / domain.comm.size
    global_std = domain.comm.allreduce(local_damage_std, op=MPI.SUM) / domain.comm.size
    global_p0  = domain.comm.allreduce(local_p0_mean, op=MPI.SUM) / domain.comm.size
    
    extracted_time.append(current_time)
    extracted_damage_mean.append(global_chi)
    extracted_damage_volatility.append(global_std)
    extracted_shear_jump.append(global_u1)
    extracted_normal_jump.append(global_u3)
    extracted_contact_pressure.append(global_p0)
    
    if step % 100 == 0 and MPI.COMM_WORLD.rank == 0:
        print(f"Step {step}/{total_steps} | t={current_time:.2f}s | E[d]={global_chi:.4f} | Std[d]={global_std:.4f} | u1={global_u1:.4f} | u3={global_u3:.5f} | p0={global_p0:.2f}")

# =============================================================================
# 6. CLEANUP & SAVE 6-COLUMN CSV DATASET
# =============================================================================
loop_end_time = MPI.Wtime()

vtf_u_mean.close()
vtf_u_std.close()
vtf_chi_mean.close()
vtf_chi_std.close()

solver_chi.destroy()
A_matrix.destroy()
b_vector.destroy()

if MPI.COMM_WORLD.rank == 0:
    output_matrix = np.column_stack((
        extracted_time,
        extracted_damage_mean,
        extracted_damage_volatility,
        extracted_shear_jump,
        extracted_normal_jump,
        extracted_contact_pressure
    ))
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssfem_simulation_data.csv")
    np.savetxt(
        csv_file,
        output_matrix,
        delimiter=",",
        header="Time,Expected_Damage_Mean,Damage_Volatility,Shear_Jump_u1,Normal_Jump_u3,Contact_Pressure_p0",
        comments=""
    )
    print(f"\nSimulation Completed Successfully in {loop_end_time - loop_start_time:.2f} seconds!")
    print(f"6-column data matrix (including exact Lagrange multiplier p0) saved to: {csv_file}")

