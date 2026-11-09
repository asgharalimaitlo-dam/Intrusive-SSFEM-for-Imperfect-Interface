import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem, io
import dolfinx.fem.petsc  
import dolfinx.geometry
import basix.ufl
import ufl
from ufl import inner, grad, dx

# =============================================================================
# 1. TUNED PHYSICS PARAMETERS & HIGH-RESOLUTION TIME HORIZON CONFIGURATION
# =============================================================================
L, W = 100.0, 20.0  
P_modes = 2  # Mode 0: Expected Mean, Mode 1: Stochastic Volatility

E = 70000.0       
nu = 0.3          
eta = 10.0        
alpha = 0.1       

# HIGH-RESOLUTION TIME STEP SETTINGS: (t_0 = 0 to t_f = 400 * 0.002 = 0.8 seconds)
dt = 0.002        # Finer steps capture sharp microscale fatigue steps
total_steps = 400 # 400 incremental steps trace multi-cycle fatigue trends
penalty_val = 1e6 

# NON-LINEARITY FIX: Lowered fracture threshold ensures load cycles trigger damage growth
omega = 0.01

# Analytical Chaos Coupling Weight Tensor C_ijk = E[Psi_i * Psi_j * Psi_k]
C_ijk = np.zeros((P_modes, P_modes, P_modes))
C_ijk[0, 0, 0] = 1.0
C_ijk[0, 1, 1] = 1.0
C_ijk[1, 0, 1] = 1.0
C_ijk[1, 1, 0] = 1.0

# Generate 2D physical mesh tracking boundary coordinates
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, points=[np.array([0.0, 0.0]), np.array([L, W])],
    n=[50, 10], cell_type=mesh.CellType.triangle
)

# Interleave vector and scalar function spaces via mixed gPC expansion structures
el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
V_jump = fem.functionspace(domain, basix.ufl.mixed_element([el_u] * P_modes))
el_d = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
V_d = fem.functionspace(domain, basix.ufl.mixed_element([el_d] * P_modes))

U_modes = fem.Function(V_jump)      
d_modes = fem.Function(V_d)    
d_old = fem.Function(V_d)      

# SOFTENING FIX: Lowered baseline stiffness ensures rapid non-linear amplification
k_base_constant = fem.Constant(domain, PETSc.ScalarType(5.0))

def get_k_star(d_val):
    # DYNAMIC INTERFACE COUPLING: Vanishes quadratically as local integrity d -> 0.0
    mu_0 = k_base_constant
    lam_0 = 1.4 * k_base_constant
    k_normal = (lam_0 + 2.0 * mu_0) * (d_val**2)
    k_shear = mu_0 * (d_val**2)
    return ufl.as_matrix([[k_shear, 0, 0], [0, k_shear, 0], [0, 0, k_normal]])

# Dynamic boundary tracking fields compiled as continuous constants
g_cyclic_baseline = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0)))
g_stochastic_fluctuation = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0)))

# Native UFL scenario tracking toggle (0.0 = Scenario A, 1.0 = Scenario B)
scenario_toggle = fem.Constant(domain, PETSc.ScalarType(0.0))

g_stochastic = [
    g_cyclic_baseline,
    scenario_toggle * g_stochastic_fluctuation
]

v_test_modes = ufl.TestFunctions(V_jump)
u_trial_modes = ufl.split(U_modes)
d_curr_modes = ufl.split(d_modes)

# Compile foundational variational statement using explicit scalar projection channels
F_equilibrium = 0.0
for k in range(P_modes):
    v_k = v_test_modes[k]
    F_equilibrium -= ufl.inner(g_stochastic[k], v_k) * dx
    for i in range(P_modes):
        u_i = u_trial_modes[i]
        
        # Spatial normal unit vector matching perpendicular direction 3
        e_3 = ufl.as_vector([0.0, 0.0, 1.0])
        u_3_expected = ufl.real(ufl.dot(u_i, e_3))
        v_3_test = ufl.dot(v_k, e_3)
        
        # Unilateral Contact Work applied strictly across the matching scalar normal channels
        penalty_factor = ufl.conditional(u_3_expected < 0.0, penalty_val, 0.0)
        F_equilibrium += penalty_factor * u_3_expected * v_3_test * dx
        
        for j in range(P_modes):
            if C_ijk[i, j, k] > 0.0:
                K_star_j = get_k_star(d_curr_modes[j])
                F_equilibrium += C_ijk[i, j, k] * ufl.inner(K_star_j * u_i, v_k) * dx

d_trial, d_test = ufl.TrialFunctions(V_d), ufl.TestFunctions(V_d)
np.random.seed(2026)
W_increments = np.random.normal(0.0, np.sqrt(dt), size=(total_steps, P_modes))

project_dir = os.path.dirname(os.path.abspath(__file__))
scenarios = ["A", "B"]

# =============================================================================
# 2. RUN COUPLING SOLVER ENGINE SCENARIOS (400-INCREMENT LIFE-CYCLE SENSITIVITY)
# =============================================================================
for scen in scenarios:
    if MPI.COMM_WORLD.rank == 0:
        print(f"\n--- EXECUTING HIGH-RES CYCLIC RUN FOR SCENARIO {scen} ---")
        
    # Rebuild independent problem context handlers to drop old backend caches
    u_problem = dolfinx.fem.petsc.NonlinearProblem(F_equilibrium, U_modes, petsc_options={
        "snes_type": "newtonls", "snes_linesearch_type": "none", "snes_atol": 1e-6, "snes_rtol": 1e-6, "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"
    }, petsc_options_prefix=f"sfem_interface_{scen}_")
        
    # Flush memory arrays completely across cross-core boundaries
    U_modes.x.array[:] = 0.0
    U_modes.x.petsc_vec.set(0.0)
    U_modes.x.scatter_forward()
    
    # ENFORCE SYSTEM BOUNDS: Reset Mean integrity d_0 to 1.0 (Pristine Undamaged Joint)
    num_dofs_sub = len(d_modes.sub(0).collapse().x.array)
    d_modes.x.array[0:num_dofs_sub] = 1.0  
    d_modes.x.array[num_dofs_sub:]  = 0.0  
    d_modes.x.petsc_vec.assemble()
    d_modes.x.scatter_forward()
    
    d_old.x.array[:] = d_modes.x.array[:]
    d_old.x.scatter_forward()
    
    if scen == "A":
        scenario_toggle.value = 0.0
    else:
        scenario_toggle.value = 1.0
        
    # Compile subdifferential thermodynamic fatigue evolution statements
    M_d_form, R_damage_form = 0.0, 0.0
    for j in range(P_modes):
        M_d_form += (eta / dt) * ufl.inner(d_trial[j], d_test[j]) * dx
        R_damage_form += (eta / dt) * ufl.inner(d_old.sub(j), d_test[j]) * dx
        R_damage_form -= alpha * ufl.inner(grad(d_curr_modes[j]), grad(d_test[j])) * dx
        energy_driving_force = ufl.real(0.5 * ufl.inner(U_modes, U_modes)) 
        damage_source = ufl.conditional(omega - energy_driving_force < 0.0, omega - energy_driving_force, 0.0)
        R_damage_form += ufl.inner(damage_source, d_test[j]) * dx

    a_chi, L_chi = fem.form(M_d_form), fem.form(R_damage_form)
    
    time_log, d_det_log, d_stoc_log, u_shear_log, u_normal_log = [], [], [], [], []
    
    #    # =========================================================================
    # 5. INNER TIME-INTEGRATION INCREMENT LOOP WITH DYNAMIC SOLVER RE-REGISTRATION
    # =============================================================================
    for step in range(1, total_steps + 1):
        current_time = step * dt
        
        # FIXED: Re-instantiate the NonlinearProblem context directly INSIDE the time loop!
        # This forces the C++ backend to read the degraded d_modes array coefficients at each step,
        # triggering massive, non-linear amplification spikes in the displacement jumps!
        u_problem = dolfinx.fem.petsc.NonlinearProblem(F_equilibrium, U_modes, petsc_options={
            "snes_type": "newtonls", "snes_linesearch_type": "none", 
            "snes_atol": 1e-6, "snes_rtol": 1e-6, 
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"
        }, petsc_options_prefix=f"sfem_dynamic_{scen}_{step}_")
        
        cyclic_force = 280.0 * np.sin(2.0 * np.pi * current_time)
                # FIXED: Variable names matched perfectly to the declarations on Line 53
        g_cyclic_baseline.value[:] = np.array([cyclic_force, cyclic_force, 0.0])
        g_stochastic_fluctuation.value[:] = np.array([90.0 * np.sin(4.0 * np.pi * current_time), 
                                                      90.0 * np.sin(4.0 * np.pi * current_time), 0.0])

            
        u_problem.solve()
        U_modes.x.scatter_forward()

        
        A_matrix = fem.petsc.assemble_matrix(a_chi)
        A_matrix.assemble()
        b_vector = fem.petsc.assemble_vector(L_chi)
        b_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        for j in range(P_modes):
            d_old_collapsed = d_old.sub(j).collapse()
            if scen == "A" and j == 1:
                noise_increment = np.zeros_like(d_old_collapsed.x.array)
            else:
                noise_increment = 0.20 * d_old_collapsed.x.array * W_increments[step-1, j]
                
            if j == 0:
                b_vector.array[len(noise_increment):] += noise_increment
            else:
                b_vector.array[0:len(noise_increment)] += noise_increment
                
        solver_d = PETSc.KSP().create(domain.comm)
        solver_d.setOperators(A_matrix)
        solver_d.setType(PETSc.KSP.Type.PREONLY)
        solver_d.getPC().setType(PETSc.PC.Type.LU)
        solver_d.solve(b_vector, d_modes.x.petsc_vec)
        d_modes.x.scatter_forward()
        
        d_modes.x.array[:] = np.clip(d_modes.x.array, 0.0, 1.0)
        d_old.x.array[:] = d_modes.x.array[:]
        
        V_u_collapsed = V_jump.sub(0).collapse()
        u_mean_space = fem.Function(V_u_collapsed)
        u_mean_space.x.array[:] = U_modes.sub(0).collapse().x.array[:]
        u_array = np.real(u_mean_space.x.array.reshape(-1, 3))
        
        local_u_shear  = np.mean(u_array[:, 1])  # Component u2 (Shear)
        local_u_normal = np.mean(u_array[:, 2])  # Component u3 (Normal)
        local_d_det    = np.mean(d_modes.sub(0).collapse().x.array)
        local_d_stoc   = np.mean(d_modes.sub(1).collapse().x.array)
        
        global_u_shear  = domain.comm.allreduce(local_u_shear, op=MPI.SUM) / domain.comm.size
        global_u_normal = domain.comm.allreduce(local_u_normal, op=MPI.SUM) / domain.comm.size
        global_d_det    = domain.comm.allreduce(local_d_det, op=MPI.SUM) / domain.comm.size
        global_d_stoc   = domain.comm.allreduce(local_d_stoc, op=MPI.SUM) / domain.comm.size
        
        time_log.append(current_time)
        d_det_log.append(global_d_det)
        d_stoc_log.append(global_d_stoc)
        u_shear_log.append(global_u_shear)
        u_normal_log.append(global_u_normal)
        
        if MPI.COMM_WORLD.rank == 0 and step % 80 == 0:
            print(f"Scenario {scen} | Progress: {int(step/total_steps*100)}% | Integrity E[d]: {global_d_det:.4f}")
        
    if MPI.COMM_WORLD.rank == 0:
        # Write scenario specific comparative arrays
        scen_out_name = f"stochastic_comparison_scenario_{scen}.csv"
        matrix_scen_out = np.column_stack((time_log, d_det_log, d_stoc_log, u_shear_log))
        np.savetxt(os.path.join(project_dir, scen_out_name), matrix_scen_out, delimiter=",",
                   header="Time,Damage_Mean,Damage_Volatility,Shear_Jump_u2", comments="")
        print(f"--> Saved scenario history matrix: {scen_out_name}")
        # FIXED RESTORATION MANDATE: Explicitly generate the global tracking database for history visualizers
        # We save Scenario B data to match your paper's mixed tracking anchors
    if scen == "B":
       global_csv_path = os.path.join(project_dir, "sfem_simulation_data.csv")
       matrix_global_out = np.column_stack((time_log, d_det_log, d_stoc_log, u_shear_log, u_normal_log))
       np.savetxt(global_csv_path, matrix_global_out, delimiter=",",
                 header="Time,Expected_Damage_Mean,Damage_Volatility,Shear_Jump_u2,Normal_Jump_u3", comments="")
       print(f"--> SUCCESS! Global kinematics file created cleanly at: {global_csv_path}")
       # =============================================================================
       # OPTIMIZED EXTRES PROFILE SLICE EXTRACTION AT STEP 400 (FINAL LIFECYCLE)
       # =============================================================================
    if MPI.COMM_WORLD.rank == 0 and scen == "B":
        x1_points = np.linspace(0.0, L, 100)
        stress_coords = np.zeros((100, 3))
        stress_coords[:, 0] = x1_points 
        stress_coords[:, 1] = W / 2.0  # Midline slice profile coordinate
    
        bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, stress_coords)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, stress_coords)
    
        cells = []
        for i in range(len(stress_coords)):
           links = colliding_cells.links(i)
           cells.append(links if len(links) > 0 else 0)
        
        # Evaluate displacement jumps and integrity fields at identical points
        u_mean_collapsed = U_modes.sub(0).collapse()
        d_mean_collapsed = d_modes.sub(0).collapse()
    
        u_eval = u_mean_collapsed.eval(stress_coords, np.array(cells, dtype=np.int32))
        d_eval = np.real(d_mean_collapsed.eval(stress_coords, np.array(cells, dtype=np.int32)))
    
        u2_shear_jump = np.real(u_eval[:, 1])
    
        # Calculate constitutive shear stress: sigma_12 = k_base * d^2 * [[u2]]
        # Using k_base = 5.0 MPa/mm matching your soft baseline state
        k_shear_base = 5.0
        computed_shear_stress = k_shear_base * (d_eval**2) * u2_shear_jump
    
        # Export stress profile matrix to your active workspace folder
        stress_matrix = np.column_stack((x1_points, computed_shear_stress))
        out_stress_name = "interface_stress_profile.csv"
        np.savetxt(os.path.join(project_dir, out_stress_name), stress_matrix, delimiter=",", 
                   h_eader="x1_coordinate,Shear_Stress_Sigma12", comments="")
        print(f"--> SUCCESS! Final stress line profile exported cleanly to: {out_stress_name}")

    
