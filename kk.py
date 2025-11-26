def solve_locally(model, make_plots=True):
    print("\n--- Solving with Local MindtPy (GLPK + IPOPT) ---")
    
    # 1. PASTE YOUR EXACT PATH HERE inside the r"..."
    # Make sure to include the actual executable filenames at the end!
    # GLPK is named 'glpsol.exe', IPOPT is named 'ipopt.exe'
    
    # UPDATE THIS PART:
    bin_folder = r"C:\Users\YOUR_NAME\anaconda3\envs\camd_env\Library\bin"
    
    glpk_path = f"{bin_folder}\\glpsol.exe"
    ipopt_path = f"{bin_folder}\\ipopt.exe"
    
    # 2. Setup SolverFactory
    opt = SolverFactory('mindtpy')
    
    try:
        results = opt.solve(model, 
                            mip_solver='glpk', 
                            nlp_solver='ipopt', 
                            
                            # THIS IS THE KEY FIX:
                            mip_solver_args={'executable': glpk_path}, 
                            nlp_solver_args={'executable': ipopt_path}, 
                            
                            strategy='OA', 
                            init_strategy='FP',
                            tee=True)
                            
    except Exception as e:
        print(f"Solver Error: {e}")
        return

    # ... (Keep the rest of your original function below this line) ...
    if (results.solver.status == pyo.SolverStatus.ok and 
        results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
        
        print("\n--- Optimal Solvent Candidate Found ---\n")
        print(f"Objective Value: {pyo.value(model.Objective):.4f}")
        
        print("Molecular Structure:")
        for g in model.G:
            c = pyo.value(model.N[g])
            if c > 0.1:
                print(f"  {g}: {int(round(c))}")

        print("\nEstimated Properties:")
        print(f"  RED: {pyo.value(model.RED):.4f}")
        print(f"  Delta: {pyo.value(model.Delta):.2f}")
        print(f"  Cp_spec: {pyo.value(model.Cp_spec):.3f}")
        print(f"  Density: {pyo.value(model.Density):.1f}")
        
        if make_plots:
            groups = [g for g in model.G if pyo.value(model.N[g])>0.1]
            counts = [pyo.value(model.N[g]) for g in groups]
            plt.figure(figsize=(6,4))
            plt.bar(groups, counts, color='teal')
            plt.title('Optimal Molecule')
            plt.show()
    else:
        print("\n--- No Solution Found ---")
        print(f"Status: {results.solver.status}")

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Check GLPK
glpk_available = SolverFactory('glpk').available(exception_flag=False)
print(f"GLPK available: {glpk_available}")

# Check IPOPT
ipopt_available = SolverFactory('ipopt').available(exception_flag=False)
print(f"IPOPT available: {ipopt_available}")

import os
import sys

# Get the base directory of your current Conda environment
env_path = sys.prefix

# The standard location for executables in Conda on Windows is 'Library/bin'
candidate_path = os.path.join(env_path, 'Library', 'bin')

print(f"Checking folder: {candidate_path}")
print("-" * 30)

# Check for GLPK (the executable is named glpsol.exe)
glpk_exe = os.path.join(candidate_path, 'glpsol.exe')
if os.path.exists(glpk_exe):
    print(f"✅ FOUND GLPK: {glpk_exe}")
else:
    print(f"❌ GLPK not found in Library/bin. Checking 'Scripts'...")
    # Fallback check
    glpk_exe = os.path.join(env_path, 'Scripts', 'glpsol.exe')
    if os.path.exists(glpk_exe):
        print(f"✅ FOUND GLPK: {glpk_exe}")
    else:
        print("❌ GLPK executable completely missing.")

# Check for IPOPT (the executable is named ipopt.exe)
ipopt_exe = os.path.join(candidate_path, 'ipopt.exe')
if os.path.exists(ipopt_exe):
    print(f"✅ FOUND IPOPT: {ipopt_exe}")
else:
    print(f"❌ IPOPT not found in Library/bin. Checking 'Scripts'...")
    # Fallback check
    ipopt_exe = os.path.join(env_path, 'Scripts', 'ipopt.exe')
    if os.path.exists(ipopt_exe):
        print(f"✅ FOUND IPOPT: {ipopt_exe}")
    else:
        print("❌ IPOPT executable completely missing.")