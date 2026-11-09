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
# 1. CORE SOLVER SUBROUTINE FUNCTION FOR SINGLE MESH AT t = dt
# =============================================================================
def solve_sfem_mesh(n_cells):
    """Solves the stochastic interface problem for a specific mesh density."""
    L, W = 100.0, 20.0
    P_modes = 2
    eta, omega, alpha, dt, penalty_val = 10.0, 0.05, 0.1, 0.01, 1e6
    C_ijk = np.ones((P_modes, P_modes, P_modes))

    # Mesh Initialization
    domain = mesh.create_rectangle(
        comm=MPI.COMM_WORLD, 
        points=[np.array([0.0, 0.0]), np.array([L, W])],
        n=[n_cells * 5, n_cells], 
        cell_type=mesh.CellType.triangle
    )

    # Function Spaces via modern Basix API
    el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    V_jump = fem.functionspace(domain, basix.ufl.mixed_element([el_u] * P_modes))
    el_chi = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V_chi = fem.functionspace(domain, basix.ufl.mixed_element([el_chi] * P_modes))

    U_modes = fem.Function(V_jump)
    Chi_modes = fem.Function(V_chi)
    Chi_old = fem.Function(V_chi)
    
    # 1D Coordinates Array View Shape Realignment Fix
    Chi_modes.sub(0).interpolate(lambda x: np.ones(x.shape[1]))
    Chi_old.sub(0).interpolate(lambda x: np.ones(x.shape[1]))

    v_test_modes = ufl.TestFunctions(V_jump)
    u_trial_modes = ufl.split(U_modes)
    chi_curr_modes = ufl.split(Chi_modes)

    # Kinematics Weak Form Residual
    def get_k_star(chi_val):
        mu_0, lam_0 = 5000.0, 7000.0
        k_normal = (lam_0 + 2.0 * mu_0) * (chi_val**2)
        k_shear = mu_0 * (chi_val**2)
        return ufl.as_matrix([[k_shear, 0, 0], [0, k_shear, 0], [0, 0, k_normal]])

    # Dynamic Random Traction Fields: Mean cyclic load + stochastic volatility
    g_stochastic = [
        fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 45.0 * np.sin(2.0 * np.pi * dt)))),
        fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 5.0 * np.sin(2.0 * np.pi * dt))))
    ]

    F_equilibrium = 0.0
    for k in range(P_modes):
        F_equilibrium -= ufl.inner(g_stochastic[k], v_test_modes[k]) * dx
        for i in range(P_modes):
            u_i = u_trial_modes[i]
            
            # Complex build compatibility handler via ufl.real()
            u_3_expected = ufl.real(u_i[2])
            penalty_factor = ufl.conditional(u_3_expected < 0.0, penalty_val, 0.0)
            F_equilibrium += penalty_factor * ufl.inner(u_i, v_test_modes[k]) * dx
            
            for j in range(P_modes):
                if C_ijk[i, j, k] > 0.0:
                    F_equilibrium += C_ijk[i, j, k] * ufl.inner(get_k_star(chi_curr_modes[j]) * u_i, v_test_modes[k]) * dx

    petsc_options = {
        "snes_type": "newtonls", 
        "snes_atol": 1e-7, 
        "snes_rtol": 1e-7, 
        "ksp_type": "preonly", 
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    
    u_problem = dolfinx.fem.petsc.NonlinearProblem(
        F_equilibrium, 
        U_modes, 
        petsc_options=petsc_options,
        petsc_options_prefix="error_analysis_"
    )
    u_problem.solve()
    U_modes.x.scatter_forward()

    # Damage Evolution Weak Form
    chi_trial = ufl.TrialFunctions(V_chi)
    chi_test = ufl.TestFunctions(V_chi)
    M_chi_form = 0.0
    R_damage_form = 0.0
    
    W_inc = np.random.normal(0.0, np.sqrt(dt), size=P_modes)

    for j in range(P_modes):
        M_chi_form += (eta / dt) * ufl.inner(chi_trial[j], chi_test[j]) * dx
        R_damage_form += (eta / dt) * ufl.inner(Chi_old.sub(j), chi_test[j]) * dx
        R_damage_form -= alpha * ufl.inner(grad(chi_curr_modes[j]), grad(chi_test[j])) * dx
        
        energy_driving_force = ufl.real(0.5 * ufl.inner(U_modes, U_modes))
        damage_source = ufl.conditional(omega - energy_driving_force < 0.0, omega - energy_driving_force, 0.0)
        R_damage_form += ufl.inner(damage_source, chi_test[j]) * dx

    a_chi = fem.form(M_chi_form)
    L_chi = fem.form(R_damage_form)

    A_matrix = fem.petsc.assemble_matrix(a_chi)
    A_matrix.assemble()
    b_vector = fem.petsc.assemble_vector(L_chi)
    b_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    for j in range(P_modes):
        noise_increment = 0.15 * Chi_old.sub(j).collapse().x.array * W_inc[j]
        if j == 0:
            b_vector.array[len(noise_increment):] += noise_increment
        else:
            b_vector.array[0:len(noise_increment)] += noise_increment

    solver_chi = PETSc.KSP().create(domain.comm)
    solver_chi.setOperators(A_matrix)
    solver_chi.setType(PETSc.KSP.Type.PREONLY)
    solver_chi.getPC().setType(PETSc.PC.Type.LU)
    solver_chi.solve(b_vector, Chi_modes.x.petsc_vec)
    Chi_modes.x.scatter_forward()
    Chi_modes.x.array[:] = np.clip(Chi_modes.x.array, 0.0, 1.0)

    return domain, U_modes, Chi_modes

# =============================================================================
# VERSION-PROOF CROSS-MESH INTERPOLATION ROUTINE FOR MIXED SPACES
# =============================================================================
def interpolate_cross_mesh(src_func, dest_func):
    """Evaluates src_func (fine mesh) at dest_func's DOFs and copies memory directly."""
    src_mesh = src_func.function_space.mesh
    dest_space = dest_func.function_space
    
    # Secure geometric bounding box trees on the source fine mesh
    bb_tree = dolfinx.geometry.bb_tree(src_mesh, src_mesh.topology.dim)
    
    num_sub_spaces = dest_space.num_sub_spaces if hasattr(dest_space, "num_sub_spaces") else 1
    
    if num_sub_spaces > 1:
        # Loop through each stochastic expansion mode channel independently
        for mode in range(num_sub_spaces):
            V_sub, sub_to_mixed = dest_space.sub(mode).collapse()
            dof_coords = V_sub.tabulate_dof_coordinates()
            
            # Find intersecting cells in the source fine mesh using bounding box collisions
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, dof_coords)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(src_mesh, cell_candidates, dof_coords)
            
            cells = []
            for i in range(len(dof_coords)):
                # Read via native links interface for AdjacencyList compliance
                active_links = colliding_cells.links(i)
                if len(active_links) > 0:
                    cells.append(active_links[0])
                else:
                    cells.append(0)
            
            # Evaluate the fine function subchannel at these explicit coordinates
            vals = src_func.sub(mode).eval(dof_coords, np.array(cells, dtype=np.int32))
            
            # Write directly to the respective memory slots in the destination vector array
            dest_func.x.array[sub_to_mixed] = vals.flatten()
    else:
        # Fallback handle for single-space configurations (like the scalar Damage space)
        dof_coords = dest_space.tabulate_dof_coordinates()
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, dof_coords)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(src_mesh, cell_candidates, dof_coords)
        
        cells = []
        for i in range(len(dof_coords)):
            active_links = colliding_cells.links(i)
            if len(active_links) > 0:
                cells.append(active_links[0])
            else:
                cells.append(0)
                
        vals = src_func.eval(dof_coords, np.array(cells, dtype=np.int32))
        dest_func.x.array[:] = vals.flatten()

# =============================================================================
# 2. RUN CONVERGENCE LOOP AND COMPUTE COARSE-TO-FINE L2 ERRORS
# =============================================================================
mesh_densities = [10, 20, 40]
l2_errors_u = []
l2_errors_chi = []

if MPI.COMM_WORLD.rank == 0:
    print("Executing Optimized Reference Profile (n = 40) safely within RAM limits...")
mesh_ref, U_ref, Chi_ref = solve_sfem_mesh(40)

for n in mesh_densities:
    if MPI.COMM_WORLD.rank == 0:
        print(f"Solving fields for Grid Density n = {n}...")
    mesh_coarse, U_coarse, Chi_coarse = solve_sfem_mesh(n)
    
    # Instantiate clean destination functions on the coarse space
    U_ref_interp = fem.Function(U_coarse.function_space)
    Chi_ref_interp = fem.Function(Chi_coarse.function_space)
    
    # Execute the version-proof geometric node interpolation updates
    interpolate_cross_mesh(U_ref, U_ref_interp)
    interpolate_cross_mesh(Chi_ref, Chi_ref_interp)
    
    # Compute Spatial L2 Error Norms over coarse mesh domain
    err_u_form = fem.form(ufl.inner(U_coarse - U_ref_interp, U_coarse - U_ref_interp) * dx)
    err_chi_form = fem.form(ufl.inner(Chi_coarse - Chi_ref_interp, Chi_coarse - Chi_ref_interp) * dx)
    
    local_err_u = fem.assemble_scalar(err_u_form)
    local_err_chi = fem.assemble_scalar(err_chi_form)
    
    global_err_u = np.sqrt(mesh_coarse.comm.allreduce(local_err_u, op=MPI.SUM))
    global_err_chi = np.sqrt(mesh_coarse.comm.allreduce(local_err_chi, op=MPI.SUM))
    
    l2_errors_u.append(global_err_u)
    l2_errors_chi.append(global_err_chi)

# =============================================================================
# 3. COMPUTE SPATIAL CONVERGENCE RATES AND SAVE RESULT DATA
# =============================================================================
if MPI.COMM_WORLD.rank == 0:
    print("\n--- SPATIAL ERROR ANALYSIS RESULTS ---")
    h_sizes = 20.0 / np.array(mesh_densities) 
    
    for idx in range(len(mesh_densities)):
        print(f"Mesh n={mesh_densities[idx]:02d} (h={h_sizes[idx]:.3f}) | L2 Error [[u]]: {l2_errors_u[idx]:.4e} | L2 Error Chi: {l2_errors_chi[idx]:.4e}")
    
    rate_u = np.log(l2_errors_u[-1]/l2_errors_u[-2]) / np.log(h_sizes[-1]/h_sizes[-2])
    rate_chi = np.log(l2_errors_chi[-1]/l2_errors_chi[-2]) / np.log(h_sizes[-1]/h_sizes[-2])
    print(f"\nEmpirical Spatial Convergence Rate O(h^p) -> [[u]]: {rate_u:.2f} | Chi: {rate_chi:.2f}")

    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sfem_spatial_error.csv")
    np.savetxt(out_path, np.column_stack((h_sizes, l2_errors_u, l2_errors_chi)), delimiter=",",
               header="h_size,L2_Error_U,L2_Error_Chi", comments="")
    print(f"Spatial convergence data safely saved to: {out_path}")
