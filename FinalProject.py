import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# --- PROBLEM SETUP ---
# Process Constraints (Conditions e & f)
T_ABS = 313.0  # K, Absorption temperature
T_DES = 393.0  # K, Desorption temperature

# Design Constraints
RED_MAX = 1.0  # Relative Energy Difference max (Condition a)
N_MAX_GROUPS = 15  # Maximum number of total groups for model tractability

# Objective Scaling Ranges (Crucial for the weighted sum objective)
# P_min and P_max define the range over which the property is scaled to [0, 1]
# For minimisation (RED, Cp): Z_i = (P_i - P_min) / (P_max - P_min)
# For maximisation (Density): Z_i = (P_max - P_i) / (P_max - P_min)
SCALING_RANGES = {
    'RED': {'min': 0.5, 'max': 1.5},
    'Cp': {'min': 1.5, 'max': 4.0},   # J/g.K
    'Density': {'min': 700.0, 'max': 1200.0},  # kg/m^3
}

# --- GROUP CONTRIBUTION DATA (PLACEHOLDER VALUES) ---
# NOTE: These values are simplified placeholders. For a high-quality MSc project,
# the user MUST substitute these with the full, accurate group contribution
# values from the cited literature (Hukkerikar et al. [1], Rayer et al. [2]).
# The properties are: [MW (g/mol), RED_GC, Cp_GC (J/g.K), Density_GC (kg/m^3), Tm_GC (K), Tbp_GC (K)]
# Molecular Weight (MW) is added as a property for final solution evaluation.
GROUP_DATA = {
    'CH3':   [15.035,  0.05,  0.40,  80.0,  5.0,  20.0,  1],  # [..., Valency]
    'CH2':   [14.027,  0.03,  0.35,  90.0,  3.0,  25.0,  2],
    'NH2':   [16.023, -0.15,  0.25, 110.0, -5.0,  30.0,  1],  # Primary Amine (Active site)
    'OH':    [17.008, -0.20,  0.50, 150.0, 10.0,  45.0,  1],  # Hydroxyl (Polar/Regeneration aid)
    # Note: Valency (last element) is simplified for acyclic structure constraints.
}

# Define the set of groups
GROUPS = list(GROUP_DATA.keys())

def create_camd_model(weights=None):
    """
    Creates and returns the Pyomo Concrete Model for the CAMD problem.
    """
    model = pyo.ConcreteModel()

    # Default weights for Task 1 & 2 (equal weighting)
    if weights is None:
        weights = {'RED': 1/3, 'Cp': 1/3, 'Density': 1/3}

    # --- SETS and PARAMETERS ---
    model.G = pyo.Set(initialize=GROUPS, doc='Functional Groups')

    # Parameters for Group Contribution data
    model.MW_GC = pyo.Param(model.G, initialize={k: v[0] for k, v in GROUP_DATA.items()}, doc='Molecular Weight GC')
    model.RED_GC = pyo.Param(model.G, initialize={k: v[1] for k, v in GROUP_DATA.items()}, doc='RED Group Contribution')
    model.Cp_GC = pyo.Param(model.G, initialize={k: v[2] for k, v in GROUP_DATA.items()}, doc='Heat Capacity GC [J/g.K]')
    model.DENSITY_GC = pyo.Param(model.G, initialize={k: v[3] for k, v in GROUP_DATA.items()}, doc='Liquid Density GC [kg/m^3]')
    model.TM_GC = pyo.Param(model.G, initialize={k: v[4] for k, v in GROUP_DATA.items()}, doc='Melting Temp GC [K]')
    model.TBP_GC = pyo.Param(model.G, initialize={k: v[5] for k, v in GROUP_DATA.items()}, doc='Boiling Temp GC [K]')
    model.VALENCY_GC = pyo.Param(model.G, initialize={k: v[6] for k, v in GROUP_DATA.items()}, doc='Valency Contribution')

    # Weights for the Objective Function (Task 3 variation)
    model.w_RED = pyo.Param(initialize=weights['RED'], doc='Weight for RED objective')
    model.w_Cp = pyo.Param(initialize=weights['Cp'], doc='Weight for Cp objective')
    model.w_Density = pyo.Param(initialize=weights['Density'], doc='Weight for Density objective')


    # --- VARIABLES ---
    # N_k: Number of groups k. Must be a non-negative integer.
    model.N = pyo.Var(model.G, domain=pyo.NonNegativeIntegers, doc='Number of groups')


    # --- CONSTRAINTS ---

    # C1: Total number of groups must be within a reasonable limit
    model.C1_TotalGroups = pyo.Constraint(expr=sum(model.N[g] for g in model.G) <= N_MAX_GROUPS)
    model.C1_MinGroups = pyo.Constraint(expr=sum(model.N[g] for g in model.G) >= 3) # Must have at least a backbone

    # C2: Structural Feasibility (Simplified for acyclic, functionalised alkanes)
    # The sum of valencies must equal 2 * (Number of bonds). For a structure with N_g groups and N_g - 1 bonds:
    # Sum(N_k * Valency_k) = 2 * (Sum(N_k) - 1)
    # We must ensure enough chain-end groups (CH3, NH2, OH) to cap the CH2 groups.
    # We enforce that the number of capping groups is at least 2 for a linear chain.
    model.C2_Capping = pyo.Constraint(expr=model.N['CH3'] + model.N['NH2'] + model.N['OH'] >= 2)

    # C3: Amine groups must be present for a chemisorption solvent
    model.C3_AminePresent = pyo.Constraint(expr=model.N['NH2'] >= 1)

    # C4: Property Constraints (Screening)
    # Calculate properties based on group contributions
    def prop_calc(model, prop_param):
        """Calculates a property P = sum(N_k * C_k)"""
        return sum(model.N[g] * prop_param[g] for g in model.G)

    # Total Group Sums (used for specific property models)
    model.N_total = pyo.Expression(expr=sum(model.N[g] for g in model.G), doc='Total number of groups')

    # Property Expressions (using group contributions)
    model.RED = pyo.Expression(expr=prop_calc(model, model.RED_GC), doc='Relative Energy Difference')
    model.Cp = pyo.Expression(expr=prop_calc(model, model.Cp_GC), doc='Liquid Heat Capacity [J/g.K]')
    model.Density = pyo.Expression(expr=prop_calc(model, model.DENSITY_GC), doc='Liquid Density [kg/m^3]')
    model.Tm = pyo.Expression(expr=prop_calc(model, model.TM_GC), doc='Melting Temperature [K]')
    model.Tbp = pyo.Expression(expr=prop_calc(model, model.TBP_GC), doc='Boiling Temperature [K]')

    # Property Constraint C4.a: RED constraint
    model.C4a_RED = pyo.Constraint(expr=model.RED <= RED_MAX)

    # Property Constraint C4.e: Melting Temperature constraint
    model.C4e_Tm = pyo.Constraint(expr=model.Tm <= T_ABS)

    # Property Constraint C4.f: Boiling Temperature constraint
    model.C4f_Tbp = pyo.Constraint(expr=model.Tbp >= T_DES)


    # --- OBJECTIVE FUNCTION (Scaled Weighted Sum) ---
    # Calculate the scaled (normalised) objectives Z_i
    Z_RED = (model.RED - SCALING_RANGES['RED']['min']) / \
            (SCALING_RANGES['RED']['max'] - SCALING_RANGES['RED']['min'])

    Z_Cp = (model.Cp - SCALING_RANGES['Cp']['min']) / \
           (SCALING_RANGES['Cp']['max'] - SCALING_RANGES['Cp']['min'])

    # Density is Maximised, so the numerator is inverted to turn it into a minimisation objective.
    Z_Density = (SCALING_RANGES['Density']['max'] - model.Density) / \
                (SCALING_RANGES['Density']['max'] - SCALING_RANGES['Density']['min'])

    # Combined Objective: Minimise Z
    def obj_rule(model):
        return model.w_RED * Z_RED + model.w_Cp * Z_Cp + model.w_Density * Z_Density

    model.Objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize, doc='Scaled Weighted Sum Objective')

    return model

def solve_and_report(model):
    """
    Solves the model and prints the results for the optimal solvent.
    """
    # This is a Mixed-Integer Linear Program (MILP) if group contributions are linear.
    # Pyomo can use solvers like GLPK (if all linear) or Bonmin/Couenne (if nonlinear or MINLP)
    # Since our example is linear, we use GLPK for simplicity. The user may need a MINLP solver
    # if the actual Hukkerikar/Rayer models introduce nonlinear terms.

    # Recommended solver for the actual project (if non-linear):
    # opt = SolverFactory('bonmin') # Needs installation
    # Recommended solver for this simplified linear example:
    opt = SolverFactory('glpk')

    print(f"\n--- Solving Model with Weights: RED={model.w_RED.value:.3f}, Cp={model.w_Cp.value:.3f}, Density={model.w_Density.value:.3f} ---")

    try:
        results = opt.solve(model, tee=True)
    except Exception as e:
        print(f"Solver Error: {e}")
        return

    # Check for optimal solution
    if (results.solver.status == pyo.SolverStatus.ok) and \
       (results.solver.termination_condition == pyo.TerminationCondition.optimal or
        results.solver.termination_condition == pyo.TerminationCondition.feasible):

        print("\n--- OPTIMAL SOLVENT CANDIDATE FOUND ---")

        # 1. Optimal Solution Summary
        total_groups = sum(pyo.value(model.N[g]) for g in model.G)
        print(f"Objective Value (Z): {pyo.value(model.Objective):.4f}")
        print(f"Total Groups: {total_groups}")
        print("\nGroup Counts:")
        for g in model.G:
            if pyo.value(model.N[g]) > 0:
                print(f"  {g}: {pyo.value(model.N[g]):.0f}")

        # 2. Optimal Properties
        print("\nEstimated Properties:")
        mw = prop_calc(model, model.MW_GC)
        print(f"  Molecular Weight (g/mol): {pyo.value(mw):.2f}")
        print(f"  RED: {pyo.value(model.RED):.4f} (Constraint: <= {RED_MAX:.1f})")
        print(f"  Cp (J/g.K): {pyo.value(model.Cp):.3f}")
        print(f"  Density (kg/m^3): {pyo.value(model.Density):.1f}")
        print(f"  Tm (K): {pyo.value(model.Tm):.1f} (Constraint: <= {T_ABS:.1f} K)")
        print(f"  Tbp (K): {pyo.value(model.Tbp):.1f} (Constraint: >= {T_DES:.1f} K)")

    else:
        print("\n--- SOLUTION NOT FOUND ---")
        print(f"Solver Status: {results.solver.status}")
        print(f"Termination Condition: {results.solver.termination_condition}")
        print("The problem may be infeasible or requires a different solver/reformulation.")


if __name__ == '__main__':
    # Initial run for Task 2 (Equal Weights)
    initial_weights = {'RED': 1/3, 'Cp': 1/3, 'Density': 1/3}
    model = create_camd_model(weights=initial_weights)
    solve_and_report(model)

    # Example for Task 3 (You will need to run 5 different weight vectors)
    # The following shows how to set up the next run.
    # model_2 = create_camd_model(weights={'RED': 0.8, 'Cp': 0.1, 'Density': 0.1})
    # solve_and_report(model_2)