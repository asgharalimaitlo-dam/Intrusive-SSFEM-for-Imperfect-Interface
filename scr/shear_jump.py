import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem, io
import dolfinx.fem.petsc  
import dolfinx.geometry
import basix.ufl
import ufl
from ufl import TrialFunction, TestFunction, inner, grad, dx, ds, div
import os

# =============================================================================
# 0. STOCHASTIC RUN CONFIGURATION
# =============================================================================
USE_MULTIPLICATIVE_NOISE = True  

# =============================================================================
# 1. PROBLEM PARAMETERS & STOCHASTIC PC CONFIGURATION
# =============================================================================
L, W = 100.0, 20.0  
P_modes = 2  

E = 70000.0       
nu = 0.3          
eta = 10.0        
omega = 0.05      
alpha = 0.1       
dt = 0.01         
total_steps = 10
penalty_val = 1e6 

C_ijk = np.zeros((P_modes, P_modes, P_modes))
C_ijk[0, 0, 0] = 1.0
C_ijk[0, 1, 1] = 1.0
C_ijk[1, 0, 1] = 1.0
C_ijk[1, 1, 0] = 1.0

# =============================================================================
# 2. MESH GENERATION & FUNCTION SPACES
# =============================================================================
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, 
    points=[np.array([0.0, 0.0]), np.array([L, W])],
    n=[50, 10], 
    cell_type=mesh.CellType.triangle
)

el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
el_pce_u = basix.ufl.mixed_element([el_u] * P_modes)
V_jump = fem.functionspace(domain, el_pce_u)

el_chi = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
el_pce_chi = basix.ufl.mixed_element([el_chi] * P_modes)
V_chi = fem.functionspace(domain, el_pce_chi)

U_modes = fem.Function(V_jump)      
Chi_modes = fem.Function(V_chi)    
Chi_old = fem.Function(V_chi)      

Chi_modes.sub(0).interpolate(lambda x: np.ones(x.shape[1]))
Chi_old.sub(0).interpolate(lambda x: np.ones(x.shape[1]))

# =============================================================================
# 3. WEAK FORMULATION WITH STOCHASTIC BOUNDARY CONDITIONS
# =============================================================================
v_test_modes = ufl.TestFunctions(V_jump)
u_trial_modes = ufl.split(U_modes)
chi_curr_modes = ufl.split(Chi_modes)

F_equilibrium = 0.0

def get_k_star(chi_val):
    # COMPLIANCE FIX: Reduced baseline stiffness parameters to allow visible deformations
    mu_0, lam_0 = 50.0, 70.0 
    k_normal = (lam_0 + 2.0 * mu_0) * (chi_val**2)
    k_shear = mu_0 * (chi_val**2)
    return ufl.as_matrix([[k_shear, 0, 0], [0, k_shear, 0], [0, 0, k_normal]])

g_stochastic = [
    fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0))), 
    fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0)))  
]

for k in range(P_modes):
    v_k = v_test_modes[k]
    F_equilibrium -= ufl.inner(g_stochastic[k], v_k) * dx
    
    for i in range(P_modes):
        u_i = u_trial_modes[i]
        u_3_expected = ufl.real(u_i[2]) 
        penalty_factor = ufl.conditional(u_3_expected < 0.0, penalty_val, 0.0)
        F_equilibrium += penalty_factor * ufl.inner(u_i, v_k) * dx
        
        for j in range(P_modes):
            if C_ijk[i, j, k] > 0.0:
                K_star_j = get_k_star(chi_curr_modes[j])
                F_equilibrium += C_ijk[i, j, k] * ufl.inner(K_star_j * u_i, v_k) * dx

petsc_options = {
    "snes_type": "newtonls", 
    "snes_linesearch_type": "none",
    "snes_atol": 1e-6, 
    "snes_rtol": 1e-6, 
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
# 4. DAMAGE EVOLUTION SOLVER EXECUTION
# =============================================================================
chi_trial = ufl.TrialFunctions(V_chi)
chi_test = ufl.TestFunctions(V_chi)

M_chi_form = 0.0
R_damage_form = 0.0

np.random.seed(2026)
W_increments = np.random.normal(0.0, np.sqrt(dt), size=(total_steps, P_modes))

for j in range(P_modes):
    M_chi_form += (eta / dt) * ufl.inner(chi_trial[j], chi_test[j]) * dx
    R_damage_form += (eta / dt) * ufl.inner(Chi_old.sub(j), chi_test[j]) * dx
    R_damage_form -= alpha * ufl.inner(grad(chi_curr_modes[j]), grad(chi_test[j])) * dx
    
    energy_driving_force = ufl.real(0.5 * ufl.inner(U_modes, U_modes)) 
    damage_source = ufl.conditional(omega - energy_driving_force < 0.0, omega - energy_driving_force, 0.0)
    R_damage_form += ufl.inner(damage_source, chi_test[j]) * dx

a_chi = fem.form(M_chi_form)
L_chi = fem.form(R_damage_form)

# =============================================================================
# 5. TIME INTEGRATION PROCESS LOOP WITH SLICE EXTRACTION
# =============================================================================
V_u_single, _ = V_jump.sub(0).collapse()
V_chi_single, _ = V_chi.sub(0).collapse()

u_mean_out = fem.Function(V_u_single, name="U_Jump_Mean")
u_std_out  = fem.Function(V_u_single, name="U_Jump_Volatility")
chi_mean_out = fem.Function(V_chi_single, name="Damage_Mean")
chi_std_out  = fem.Function(V_chi_single, name="Damage_Volatility")

project_dir = os.path.dirname(os.path.abspath(__file__))

vtf_u_mean   = io.VTXWriter(domain.comm, os.path.join(project_dir, "out_u_mean.bp"), [u_mean_out], engine="BP4")
vtf_u_std    = io.VTXWriter(domain.comm, os.path.join(project_dir, "out_u_std.bp"), [u_std_out], engine="BP4")
vtf_chi_mean = io.VTXWriter(domain.comm, os.path.join(project_dir, "out_chi_mean.bp"), [chi_mean_out], engine="BP4")
vtf_chi_std  = io.VTXWriter(domain.comm, os.path.join(project_dir, "out_chi_std.bp"), [chi_std_out], engine="BP4")

extracted_time = []
extracted_damage_mean = []
extracted_damage_volatility = []
extracted_shear_jump = []
extracted_normal_jump = []

print("Starting Solver Engine with Amplified Traction...")
u_mean_out.x.array[:] = U_modes.sub(0).collapse().x.array[:]
chi_mean_out.x.array[:] = Chi_modes.sub(0).collapse().x.array[:]
vtf_u_mean.write(0.0)
vtf_chi_mean.write(0.0)

for step in range(1, total_steps + 1):
    current_time = step * dt
    
    # AMPLIFICATION FIX: Increased traction force magnitudes to induce structural discontinuities
    g_stochastic[0].value = np.array([45.0 * np.sin(2.0 * np.pi * current_time), 45.0 * np.sin(2.0 * np.pi * current_time), 0.0])
    g_stochastic[1].value = np.array([25.0 * np.sin(2.0 * np.pi * current_time), 25.0 * np.sin(2.0 * np.pi * current_time), 0.0])
    
    num_its, converged, *extra = u_problem.solve()
    U_modes.x.scatter_forward()
    
    A_matrix = fem.petsc.assemble_matrix(a_chi)
    A_matrix.assemble()
    b_vector = fem.petsc.assemble_vector(L_chi)
    b_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    for j in range(P_modes):
        chi_old_collapsed = Chi_old.sub(j).collapse()
        if USE_MULTIPLICATIVE_NOISE:
            noise_increment = 0.15 * chi_old_collapsed.x.array * W_increments[step-1, j]
        else:
            noise_increment = 0.15 * np.ones_like(chi_old_collapsed.x.array) * W_increments[step-1, j]
        
        if j == 0:
            mode1_dofs = np.arange(len(noise_increment)) + len(noise_increment)
            b_vector.array[mode1_dofs] += noise_increment
        else:
            b_vector.array[0:len(noise_increment)] += noise_increment
        
    solver_chi = PETSc.KSP().create(domain.comm)
    solver_chi.setOperators(A_matrix)
    solver_chi.setType(PETSc.KSP.Type.PREONLY)
    solver_chi.getPC().setType(PETSc.PC.Type.LU)
    solver_chi.solve(b_vector, Chi_modes.x.petsc_vec)
    Chi_modes.x.scatter_forward()
    
    Chi_modes.x.array[:] = np.clip(Chi_modes.x.array, 0.0, 1.0)
    Chi_old.x.array[:] = Chi_modes.x.array[:]
    
    u_mean_out.x.array[:] = U_modes.sub(0).collapse().x.array[:]
    u_std_out.x.array[:]  = U_modes.sub(1).collapse().x.array[:]
    chi_mean_out.x.array[:] = Chi_modes.sub(0).collapse().x.array[:]
    chi_std_out.x.array[:]  = Chi_modes.sub(1).collapse().x.array[:]
    
    vtf_u_mean.write(current_time)
    vtf_u_std.write(current_time)
    vtf_chi_mean.write(current_time)
    vtf_chi_std.write(current_time)
    
    u_array_reshaped = np.real(u_mean_out.x.array.reshape(-1, 3))
    local_shear_mean  = np.mean(u_array_reshaped[:, 1])  # Extract component u_2
    local_normal_mean = np.mean(u_array_reshaped[:, 2])  
    local_damage_mean = np.real(np.mean(chi_mean_out.x.array))
    local_damage_std  = np.real(np.mean(chi_std_out.x.array))
    
    global_u1  = domain.comm.allreduce(local_shear_mean, op=MPI.SUM) / domain.comm.size
    global_u3  = domain.comm.allreduce(local_normal_mean, op=MPI.SUM) / domain.comm.size
    global_chi = domain.comm.allreduce(local_damage_mean, op=MPI.SUM) / domain.comm.size
    global_std = domain.comm.allreduce(local_damage_std, op=MPI.SUM) / domain.comm.size
    
    extracted_time.append(current_time)
    extracted_damage_mean.append(global_chi)
    extracted_damage_volatility.append(global_std)
    extracted_shear_jump.append(global_u1)
    extracted_normal_jump.append(global_u3)
    
    if MPI.COMM_WORLD.rank == 0:
        x2_points = np.linspace(0.0, W, 50)
        slice_coords = np.zeros((50, 3))
        slice_coords[:, 0] = L / 2.0  
        slice_coords[:, 1] = x2_points 
        
        bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, slice_coords)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, slice_coords)
        
        cells = []
        for i in range(len(slice_coords)):
            links = colliding_cells.links(i)
            cells.append(links[0] if len(links) > 0 else 0)   
            
        u_mean_collapsed = U_modes.sub(0).collapse()
        shear_jump_values = u_mean_collapsed.eval(slice_coords, np.array(cells, dtype=np.int32))
        u2_slice_data = shear_jump_values[:, 1]
        slice_matrix = np.column_stack((x2_points, u2_slice_data))
        out_slice_name = f"shear_jump_slice_step_{step:02d}.csv"
        np.savetxt(os.path.join(project_dir, out_slice_name), slice_matrix, delimiter=",",
header="x2_coordinate,Shear_Jump_u2", comments="")
    if MPI.COMM_WORLD.rank == 0:
       print(f"Step {step:02d} | Time: {current_time:.2f}s | Expected Volatility (Std): {global_std:.6f}")
vtf_u_mean.close()
vtf_u_std.close()
vtf_chi_mean.close()
vtf_chi_std.close()
if MPI.COMM_WORLD.rank == 0:
   csv_absolute_path = os.path.join(project_dir, "sfem_simulation_data.csv")
   data_matrix = np.column_stack((
extracted_time,
extracted_damage_mean,
extracted_damage_volatility,
extracted_shear_jump,
extracted_normal_jump
))
   np.savetxt(csv_absolute_path, data_matrix, delimiter=",",
              header="Time,Expected_Damage_Mean,Damage_Volatility,Shear_Jump_u1,Normal_Jump_u3", comments="")
print(f"Simulation Matrix Saved successfully to: {csv_absolute_path}")