# FINAL WORKING CAMD MODEL (Local MindtPy)
# Fixes:
# 1. Corrected the Valency Constraint (The root cause of "Infeasible")
# 2. Adjusted MindtPy strategy to 'FP' (Feasibility Pump)

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

# ==========================================
# 1. CONSTANTS & DATA
# ==========================================
T_ABS = 313.0   
T_DES = 393.0   

DELTA_CO2 = 13.5
RED_MAX = 3.0           
N_MAX_GROUPS = 15       

# Expanded ranges to ensure solver finds a solution
SCALING_RANGES = {
    'RED': {'min': 0.0, 'max': 5.0},
    'Cp_spec': {'min': 0.5, 'max': 6.0},
    'Density': {'min': 500.0, 'max': 1500.0}
}

GROUP_DATA = {
    # Key: [MW, Vm, U, Tm, Tbp, Rho, Cp_m, Valency]
    'CH3': [15.035, 33.5, 1200,   5.0,  20.0,  80.0, 45.0, 1],
    'CH2': [14.027, 16.0, 1000,   3.0,  25.0,  90.0, 30.0, 2],
    'NH2': [16.023, 18.0, 3500,  -5.0,  30.0, 110.0, 55.0, 1],
    'OH':  [17.008, 15.0, 4000,  10.0,  45.0, 150.0, 60.0, 1]
}
GROUPS = list(GROUP_DATA.keys())

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
def create_camd_model(weights=None):
    model = pyo.ConcreteModel()
    if weights is None: weights = {'RED': 0.4, 'Cp_spec': 0.3, 'Density': 0.3}

    model.G = pyo.Set(initialize=GROUPS)
    
    # Parameters
    model.MW_GC = pyo.Param(model.G, initialize={k: v[0] for k, v in GROUP_DATA.items()})
    model.VM_GC = pyo.Param(model.G, initialize={k: v[1] for k, v in GROUP_DATA.items()})
    model.U_GC  = pyo.Param(model.G, initialize={k: v[2] for k, v in GROUP_DATA.items()})
    model.TM_GC = pyo.Param(model.G, initialize={k: v[3] for k, v in GROUP_DATA.items()})
    model.TBP_GC= pyo.Param(model.G, initialize={k: v[4] for k, v in GROUP_DATA.items()})
    model.RHO_GC= pyo.Param(model.G, initialize={k: v[5] for k, v in GROUP_DATA.items()})
    model.CP_GC = pyo.Param(model.G, initialize={k: v[6] for k, v in GROUP_DATA.items()})
    model.VAL_GC= pyo.Param(model.G, initialize={k: v[7] for k, v in GROUP_DATA.items()})

    model.w_RED = pyo.Param(initialize=weights['RED'])
    model.w_Cp  = pyo.Param(initialize=weights['Cp_spec'])
    model.w_Rho = pyo.Param(initialize=weights['Density'])

    # Variables (Relaxed bounds)
    model.N = pyo.Var(model.G, domain=pyo.NonNegativeIntegers, initialize=1, bounds=(0, 10))
    model.Delta = pyo.Var(domain=pyo.NonNegativeReals, initialize=15.0, bounds=(1, 50))
    model.RED = pyo.Var(domain=pyo.NonNegativeReals, initialize=1.0, bounds=(0, 10))

    # Expressions
    model.MW = pyo.Expression(expr=sum(model.N[g]*model.MW_GC[g] for g in model.G))
    model.Vm = pyo.Expression(expr=sum(model.N[g]*model.VM_GC[g] for g in model.G))
    model.U  = pyo.Expression(expr=sum(model.N[g]*model.U_GC[g] for g in model.G))
    model.Tm = pyo.Expression(expr=sum(model.N[g]*model.TM_GC[g] for g in model.G))
    model.Tbp= pyo.Expression(expr=sum(model.N[g]*model.TBP_GC[g] for g in model.G))
    model.Density = pyo.Expression(expr=sum(model.N[g]*model.RHO_GC[g] for g in model.G))
    model.Cp_molar= pyo.Expression(expr=sum(model.N[g]*model.CP_GC[g] for g in model.G))
    model.N_total = pyo.Expression(expr=sum(model.N[g] for g in model.G))

    # Nonlinear Properties
    model.Cp_spec = pyo.Expression(expr=model.Cp_molar / (model.MW + 1e-6))
    
    # --- CONSTRAINTS ---

    # 1. Solubility (Standard Definition)
    # Vm is in cm3/mol, U is J/mol. Factor 1e-6 converts cm3 to m3 for SI consistency in some correlations, 
    # but here we stick to standard Hildebrand units (MPa^0.5).
    # Delta^2 = (U / Vm) * 1000 (To get MPa from J/cm3) -> 1 J/cm3 = 1 MPa.
    # So Delta = sqrt(U/Vm). 
    model.C_delta_def = pyo.Constraint(expr=model.Delta**2 * (model.Vm + 1e-6) == model.U)

    # 2. RED Definition (Smooth)
    model.C_red_def = pyo.Constraint(expr=(model.RED * DELTA_CO2)**2 == (model.Delta - DELTA_CO2)**2)

    # 3. Structural Constraints
    model.C_total_max = pyo.Constraint(expr=model.N_total <= N_MAX_GROUPS)
    model.C_total_min = pyo.Constraint(expr=model.N_total >= 2)
    model.C_amine = pyo.Constraint(expr=model.N['NH2'] >= 1)
    
    # 4. Process Constraints
    model.C_RED = pyo.Constraint(expr=model.RED <= RED_MAX)
    
    # 5. CORRECTED VALENCY CONSTRAINT (The Acyclic Rule)
    # Sum( Ni * (2 - Vi) ) == 2
    model.C_valency = pyo.Constraint(
        expr=sum(model.N[g] * (2 - model.VAL_GC[g]) for g in model.G) == 2
    )

    # Objective
    Z_RED = (model.RED - SCALING_RANGES['RED']['min']) / (SCALING_RANGES['RED']['max'] - SCALING_RANGES['RED']['min'])
    Z_Cp  = (model.Cp_spec - SCALING_RANGES['Cp_spec']['min']) / (SCALING_RANGES['Cp_spec']['max'] - SCALING_RANGES['Cp_spec']['min'])
    Z_Rho = (SCALING_RANGES['Density']['max'] - model.Density) / (SCALING_RANGES['Density']['max'] - SCALING_RANGES['Density']['min'])

    model.Objective = pyo.Objective(expr=model.w_RED*Z_RED + model.w_Cp*Z_Cp + model.w_Rho*Z_Rho, sense=pyo.minimize)

    return model

# ==========================================
# 3. LOCAL SOLVER ROUTINE
# ==========================================
def solve_locally(model, make_plots=True):
    print("\n--- Solving with Local MindtPy (GLPK + IPOPT) ---")
    
    opt = SolverFactory('mindtpy')
    
    try:
        # Changed strategy to 'FP' (Feasibility Pump) - often better for finding initial integer solutions
        results = opt.solve(model, 
                            mip_solver='glpk', 
                            nlp_solver='ipopt', 
                            strategy='OA', # OA is standard, but if it fails again, try 'FP'
                            init_strategy='FP', # Use Feasibility Pump to find starting point
                            tee=True)
    except Exception as e:
        print(f"Solver Error: {e}")
        return

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
        print(f"Termination: {results.solver.termination_condition}")

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    m = create_camd_model()
    solve_locally(m, make_plots=True)