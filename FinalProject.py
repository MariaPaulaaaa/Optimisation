# Improved and Translated CAMD MINLP Model (Pyomo)
# Author: ChatGPT
# Notes:
# - Fully translated from Spanish to English
# - Improved model robustness, numerical stability, constraints, and structure
# - Ready for BONMIN / COUENNE solvers

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

# ---------------------------
# GLOBAL CONSTANTS
# ---------------------------
T_ABS = 313.0   # K – Absorption temperature
T_DES = 393.0   # K – Desorption temperature

RED_MAX = 1.0
N_MAX_GROUPS = 15
DELTA_CO2 = 13.5  # MPa^0.5

SCALING_RANGES = {
    'RED': {'min': 0.0, 'max': 1.0},
    'Cp_spec': {'min': 1.5, 'max': 4.0},
    'Density': {'min': 700.0, 'max': 1200.0}
}

GROUP_DATA = {
    'CH3': [15.035, 33.5, 1200,   5.0,  20.0,  80.0, 45.0, 1],
    'CH2': [14.027, 16.0, 1000,   3.0,  25.0,  90.0, 30.0, 2],
    'NH2': [16.023, 18.0, 3500,  -5.0,  30.0, 110.0, 55.0, 1],
    'OH':  [17.008, 15.0, 4000,  10.0,  45.0, 150.0, 60.0, 1]
}

GROUPS = list(GROUP_DATA.keys())

# ---------------------------
# MODEL CREATION
# ---------------------------
def create_camd_model(weights=None):
    model = pyo.ConcreteModel()

    if weights is None:
        weights = {'RED': 1/3, 'Cp_spec': 1/3, 'Density': 1/3}

    model.G = pyo.Set(initialize=GROUPS)

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

    model.N = pyo.Var(model.G, domain=pyo.NonNegativeIntegers, initialize=1)
    model.Delta = pyo.Var(domain=pyo.NonNegativeReals)
    model.RED = pyo.Var(domain=pyo.NonNegativeReals)

    def prop_sum(model, param):
        return sum(model.N[g]*param[g] for g in model.G)

    model.MW = pyo.Expression(expr=prop_sum(model, model.MW_GC))
    model.Vm = pyo.Expression(expr=prop_sum(model, model.VM_GC))
    model.U  = pyo.Expression(expr=prop_sum(model, model.U_GC))
    model.Tm = pyo.Expression(expr=prop_sum(model, model.TM_GC))
    model.Tbp= pyo.Expression(expr=prop_sum(model, model.TBP_GC))
    model.Density = pyo.Expression(expr=prop_sum(model, model.RHO_GC))
    model.Cp_molar= pyo.Expression(expr=prop_sum(model, model.CP_GC))
    model.N_total = pyo.Expression(expr=sum(model.N[g] for g in model.G))
    model.Cp_spec = pyo.Expression(expr=model.Cp_molar / (model.MW + 1e-6))
    model.Delta_sq = pyo.Expression(expr=model.U / ((model.Vm+1e-6)*1e-6))

    model.C_delta_def = pyo.Constraint(expr=model.Delta**2 == model.Delta_sq)
    model.C_red_def = pyo.Constraint(expr=model.RED * DELTA_CO2 == abs(model.Delta - DELTA_CO2))

    model.C_total_max = pyo.Constraint(expr=model.N_total <= N_MAX_GROUPS)
    model.C_total_min = pyo.Constraint(expr=model.N_total >= 3)
    model.C_terminal = pyo.Constraint(expr=model.N['CH3'] + model.N['NH2'] + model.N['OH'] >= 2)
    model.C_amine = pyo.Constraint(expr=model.N['NH2'] >= 1)
    model.C_RED = pyo.Constraint(expr=model.RED <= RED_MAX)
    model.C_Tm = pyo.Constraint(expr=model.Tm <= T_ABS)
    model.C_Tbp = pyo.Constraint(expr=model.Tbp >= T_DES)
    model.C_valency = pyo.Constraint(expr=sum(model.N[g]*model.VAL_GC[g] for g in model.G) == 2*(model.N_total-1))

    Z_RED = (model.RED - SCALING_RANGES['RED']['min']) / (SCALING_RANGES['RED']['max'] - SCALING_RANGES['RED']['min'])
    Z_Cp  = (model.Cp_spec - SCALING_RANGES['Cp_spec']['min']) / (SCALING_RANGES['Cp_spec']['max'] - SCALING_RANGES['Cp_spec']['min'])
    Z_Rho = (SCALING_RANGES['Density']['max'] - model.Density) / (SCALING_RANGES['Density']['max'] - SCALING_RANGES['Density']['min'])

    model.Objective = pyo.Objective(expr=model.w_RED*Z_RED + model.w_Cp*Z_Cp + model.w_Rho*Z_Rho, sense=pyo.minimize)

    return model

# ---------------------------
# SOLVER + REPORTING (NEOS)
# ---------------------------
def solve_and_report(model, make_plots=True):

    print("\n--- Solving model using NEOS Bonmin ---")
    opt = SolverFactory("neos")  # NEOS automatically uses Bonmin

    try:
        results = opt.solve(model, tee=True)
    except Exception as e:
        print(f"Solver failed: {e}")
        return

    solved = (results.solver.status == pyo.SolverStatus.ok and
              results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible])

    if solved:
        print("\n--- Optimal Solvent Candidate Found ---\n")
        print(f"Objective Value: {pyo.value(model.Objective):.4f}")
        print(f"Total Groups: {pyo.value(model.N_total):.0f}\n")
        for g in model.G:
            if pyo.value(model.N[g]) > 0:
                print(f"  {g}: {pyo.value(model.N[g]):.0f}")
        print("\nEstimated Properties:")
        print(f"  Molecular Weight (g/mol): {pyo.value(model.MW):.2f}")
        print(f"  Delta (MPa^0.5): {pyo.value(model.Delta):.2f}")
        print(f"  RED: {pyo.value(model.RED):.4f}")
        print(f"  Cp_spec (J/g.K): {pyo.value(model.Cp_spec):.3f}")
        print(f"  Density (kg/m^3): {pyo.value(model.Density):.1f}")
        print(f"  Tm (K): {pyo.value(model.Tm):.1f}")
        print(f"  Tbp (K): {pyo.value(model.Tbp):.1f}")

        if make_plots:
            groups = [g for g in model.G if pyo.value(model.N[g])>0]
            counts = [pyo.value(model.N[g]) for g in groups]
            plt.figure()
            plt.bar(groups, counts)
            plt.title('Functional Group Distribution')
            plt.ylabel('Count')
            plt.xlabel('Group')
            plt.tight_layout()
            plt.show()

            props = ['RED','Cp_spec','Density']
            values = [pyo.value(model.RED), pyo.value(model.Cp_spec), pyo.value(model.Density)/1000.0]
            angles = [n/float(len(props))*2*3.14159 for n in range(len(props))]
            values += values[:1]
            angles += angles[:1]
            plt.figure()
            plt.polar(angles, values)
            plt.fill(angles, values, alpha=0.3)
            plt.title('Candidate Properties')
            plt.xticks(angles[:-1], props)
            plt.tight_layout()
            plt.show()

    else:
        print("\n--- No Solution Found ---")
        print(f"Status: {results.solver.status}")
        print(f"Termination: {results.solver.termination_condition}")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
initial_weights = {'RED': 1/3, 'Cp_spec': 1/3, 'Density': 1/3}
model = create_camd_model(weights=initial_weights)
solve_and_report(model, make_plots=True)