import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# --- PROBLEM SETUP AND CONSTANTS ---
# Define the process temperatures which act as physical constraints on the solvent.
T_ABS = 313.0  # K (Kelvin) - Absorption column average temperature (Condition e)
T_DES = 393.0  # K (Kelvin) - Desorption column average temperature (Condition f)

# Define the maximum acceptable Relative Energy Difference (RED) for good CO2 miscibility.
RED_MAX = 1.0  # (Dimensionless) - Maximum acceptable RED (Condition a)
N_MAX_GROUPS = 15  # Maximum total number of groups to limit search space size

# Objective Scaling Ranges:
# These ranges are used to normalize the objective properties to a [0, 1] scale.
# This ensures that equal weights assigned to the objectives (e.g., 1/3, 1/3, 1/3)
# result in proportional contributions to the overall cost function.
SCALING_RANGES = {
    # RED is minimized. Range defines what 0% and 100% contribution means.
    'RED': {'min': 0.5, 'max': 1.5},
    # Heat Capacity (Cp) is minimized (lower regeneration energy).
    'Cp': {'min': 1.5, 'max': 4.0},   # J/g.K (Joules per gram per Kelvin)
    # Density is maximized (lower equipment size/flow rate).
    'Density': {'min': 700.0, 'max': 1200.0},  # kg/m^3 (Kilograms per cubic meter)
}

# --- GROUP CONTRIBUTION DATA (SCIENTIFIC PLACEHOLDERS) ---
# IMPORTANT: These values are placeholders. The user must replace these with the
# actual data from the cited literature (Hukkerikar et al. [1], Rayer et al. [2])
# for the final MSc project.
#
# Data structure for each group:
# [0: Molecular Weight (g/mol), 1: RED Contribution, 2: Cp Contribution (J/g.K),
#  3: Density Contribution (kg/m^3), 4: Tm Contribution (K), 5: Tbp Contribution (K),
#  6: Simplified Valency Contribution (for structural check)]
GROUP_DATA = {
    'CH3':   [15.035,  0.05,  0.40,  80.0,  5.0,  20.0,  1],  # Methyl group (Chain end)
    'CH2':   [14.027,  0.03,  0.35,  90.0,  3.0,  25.0,  2],  # Methylene group (Chain backbone)
    'NH2':   [16.023, -0.15,  0.25, 110.0, -5.0,  30.0,  1],  # Primary Amine (Active CO2 absorption site)
    'OH':    [17.008, -0.20,  0.50, 150.0, 10.0,  45.0,  1],  # Hydroxyl group (Polarity, viscosity control)
}

# Define the set of functional groups available for molecule construction.
GROUPS = list(GROUP_DATA.keys())

def create_camd_model(weights=None):
    """
    Constructs the Pyomo Concrete Model for the Computer-Aided Molecular Design (CAMD)
    of the CO2 capture solvent.
    
    Args:
        weights (dict, optional): Dictionary containing weights for the RED, Cp,
                                  and Density objectives. Defaults to equal weights (1/3 each).
    """
    model = pyo.ConcreteModel()

    # Set default weights for the objective function if none are provided (Task 1 & 2)
    if weights is None:
        weights = {'RED': 1/3, 'Cp': 1/3, 'Density': 1/3}

    # --- SETS and PARAMETERS ---
    model.G = pyo.Set(initialize=GROUPS, doc='Set of Available Functional Groups')

    # Define Pyomo Parameters to store the Group Contribution values
    model.MW_GC = pyo.Param(model.G, initialize={k: v[0] for k, v in GROUP_DATA.items()}, doc='Molecular Weight Group Contribution')
    model.RED_GC = pyo.Param(model.G, initialize={k: v[1] for k, v in GROUP_DATA.items()}, doc='Relative Energy Difference Group Contribution')
    model.Cp_GC = pyo.Param(model.G, initialize={k: v[2] for k, v in GROUP_DATA.items()}, doc='Heat Capacity Group Contribution [J/g.K]')
    model.DENSITY_GC = pyo.Param(model.G, initialize={k: v[3] for k, v in GROUP_DATA.items()}, doc='Liquid Density Group Contribution [kg/m^3]')
    model.TM_GC = pyo.Param(model.G, initialize={k: v[4] for k, v in GROUP_DATA.items()}, doc='Melting Temperature Group Contribution [K]')
    model.TBP_GC = pyo.Param(model.G, initialize={k: v[5] for k, v in GROUP_DATA.items()}, doc='Boiling Temperature Group Contribution [K]')
    model.VALENCY_GC = pyo.Param(model.G, initialize={k: v[6] for k, v in GROUP_DATA.items()}, doc='Simplified Valency Contribution for Structural Constraint')

    # Objective weights (used for multi-objective analysis in Task 3)
    model.w_RED = pyo.Param(initialize=weights['RED'], doc='Weight for RED objective term')
    model.w_Cp = pyo.Param(initialize=weights['Cp'], doc='Weight for Heat Capacity objective term')
    model.w_Density = pyo.Param(initialize=weights['Density'], doc='Weight for Density objective term')


    # --- VARIABLES ---
    # N_k: The number of times group k appears in the molecule.
    # This must be a non-negative integer, making this a Mixed-Integer (Linear or Non-Linear) Program.
    model.N = pyo.Var(model.G, domain=pyo.NonNegativeIntegers, doc='Number of groups k in the designed molecule')


    # --- CONSTRAINTS ---

    # C1: Total size constraint to limit the design space.
    model.C1_TotalGroups = pyo.Constraint(expr=sum(model.N[g] for g in model.G) <= N_MAX_GROUPS,
                                          doc='Upper limit on total number of groups (N_max)')
    # Ensure a molecule is formed (minimum 3 groups, e.g., NH2-CH2-OH)
    model.C1_MinGroups = pyo.Constraint(expr=sum(model.N[g] for g in model.G) >= 3,
                                        doc='Lower limit on total number of groups')

    # C2: Simplified Structural Feasibility Constraint (Acyclic molecule construction).
    # Ensures that there are enough chain-capping groups (CH3, NH2, OH) to terminate the backbone (CH2).
    # For a linear, acyclic molecule, there must be at least two capping groups.
    model.C2_Capping = pyo.Constraint(expr=model.N['CH3'] + model.N['NH2'] + model.N['OH'] >= 2,
                                      doc='Ensures a chemically plausible acyclic structure (min 2 end-caps)')

    # C3: Amine groups must be present for a chemisorption solvent.
    # This is a core requirement for CO2 reactive absorption.
    model.C3_AminePresent = pyo.Constraint(expr=model.N['NH2'] >= 1,
                                           doc='At least one primary amine group is required for chemisorption')

    # C4: Property Estimation and Screening Constraints

    def prop_calc(model, prop_param):
        """Helper function to calculate a property P based on P = sum(N_k * C_k)"""
        return sum(model.N[g] * prop_param[g] for g in model.G)

    # Define Expressions for all calculated properties
    model.RED = pyo.Expression(expr=prop_calc(model, model.RED_GC), doc='Relative Energy Difference (RED)')
    model.Cp = pyo.Expression(expr=prop_calc(model, model.Cp_GC), doc='Liquid Heat Capacity (Cp) [J/g.K]')
    model.Density = pyo.Expression(expr=prop_calc(model, model.DENSITY_GC), doc='Liquid Density (rho) [kg/m^3]')
    model.Tm = pyo.Expression(expr=prop_calc(model, model.TM_GC), doc='Normal Melting Temperature (Tm) [K]')
    model.Tbp = pyo.Expression(expr=prop_calc(model, model.TBP_GC), doc='Normal Boiling Temperature (Tbp) [K]')

    # Property Constraint C4.a: RED must be low for good CO2 solubility.
    model.C4a_RED = pyo.Constraint(expr=model.RED <= RED_MAX,
                                   doc='RED must be below maximum threshold (Condition a)')

    # Property Constraint C4.e: Melting Temperature must be below absorption temperature to avoid solidification.
    model.C4e_Tm = pyo.Constraint(expr=model.Tm <= T_ABS,
                                  doc=f'Tm must be less than T_Abs ({T_ABS} K) (Condition e)')

    # Property Constraint C4.f: Boiling Temperature must be above desorption temperature to prevent excessive solvent loss.
    model.C4f_Tbp = pyo.Constraint(expr=model.Tbp >= T_DES,
                                   doc=f'Tbp must be greater than T_Des ({T_DES} K) (Condition f)')


    # --- OBJECTIVE FUNCTION (Scaled Weighted Sum) ---
    # The objective combines three normalized properties: RED (Min), Cp (Min), and Density (Max).

    # 1. Normalize RED (Minimization)
    RED_RANGE = SCALING_RANGES['RED']['max'] - SCALING_RANGES['RED']['min']
    Z_RED = (model.RED - SCALING_RANGES['RED']['min']) / RED_RANGE

    # 2. Normalize Heat Capacity (Cp) (Minimization)
    CP_RANGE = SCALING_RANGES['Cp']['max'] - SCALING_RANGES['Cp']['min']
    Z_Cp = (model.Cp - SCALING_RANGES['Cp']['min']) / CP_RANGE

    # 3. Normalize Density (Maximization) - Invert the term for minimization
    DENSITY_RANGE = SCALING_RANGES['Density']['max'] - SCALING_RANGES['Density']['min']
    Z_Density = (SCALING_RANGES['Density']['max'] - model.Density) / DENSITY_RANGE

    # Combined Objective Rule: Minimize Z_total
    def obj_rule(model):
        return model.w_RED * Z_RED + model.w_Cp * Z_Cp + model.w_Density * Z_Density

    model.Objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize, doc='Total Scaled Weighted Sum Objective')

    return model

def solve_and_report(model):
    """
    Solves the Pyomo model using a suitable solver and prints the optimal result.
    """
    # Use GLPK for this simplified linear problem. For the final non-linear project,
    # the user should use a robust MINLP solver like 'bonmin' or 'couenne' with Pyomo.
    # Alternatively, the GAMS interface (via pyomo.environ.SolverFactory('gams', solver='CONOPT'))
    # can be used if GAMS is preferred, as requested in the original problem.
    opt = SolverFactory('glpk')

    current_weights = model.w_RED.value, model.w_Cp.value, model.w_Density.value
    print(f"\n--- Starting Solver with Weights: RED={current_weights[0]:.3f}, Cp={current_weights[1]:.3f}, Density={current_weights[2]:.3f} ---")

    try:
        # Solve the optimization problem
        results = opt.solve(model, tee=True)
    except Exception as e:
        print(f"Solver Execution Error: {e}")
        return

    # Check if a valid solution was found
    if (results.solver.status == pyo.SolverStatus.ok) and \
       (results.solver.termination_condition in (pyo.TerminationCondition.optimal,
                                                 pyo.TerminationCondition.feasible)):

        print("\n--- OPTIMAL SOLVENT CANDIDATE IDENTIFIED ---")

        # 1. Optimal Solution Summary
        total_groups = sum(pyo.value(model.N[g]) for g in model.G)
        print(f"Total Scaled Objective Value (Z): {pyo.value(model.Objective):.4f}")
        print(f"Total Number of Groups (N_total): {total_groups:.0f}")

        print("\nOptimal Group Counts:")
        for g in model.G:
            if pyo.value(model.N[g]) > 0:
                print(f"  {g}: {pyo.value(model.N[g]):.0f} units")

        # 2. Calculated Properties of the Optimal Solvent
        print("\nEstimated Physical Properties:")
        mw = prop_calc(model, model.MW_GC)
        print(f"  Molecular Weight (g/mol): {pyo.value(mw):.2f}")
        print(f"  RED: {pyo.value(model.RED):.4f} (Constraint: <= {RED_MAX:.1f})")
        print(f"  Cp (J/g.K): {pyo.value(model.Cp):.3f} (Lower is better)")
        print(f"  Density (kg/m^3): {pyo.value(model.Density):.1f} (Higher is better)")
        print(f"  Tm (K): {pyo.value(model.Tm):.1f} (Constraint: <= {T_ABS:.1f} K - Must be liquid)")
        print(f"  Tbp (K): {pyo.value(model.Tbp):.1f} (Constraint: >= {T_DES:.1f} K - Avoid loss)")

    else:
        print("\n--- SOLUTION COULD NOT BE FOUND ---")
        print(f"Solver Status: {results.solver.status}")
        print(f"Termination Condition: {results.solver.termination_condition}")
        print("Suggestion: Check feasibility constraints or try a different solver/reformulation.")


if __name__ == '__main__':
    # --- Task 2: Initial Run with Equal Weights ---
    initial_weights = {'RED': 1/3, 'Cp': 1/3, 'Density': 1/3}
    model = create_camd_model(weights=initial_weights)
    solve_and_report(model)

    # --- Task 3: Example of Multi-Objective Runs (Need 5 total) ---
    # The user must uncomment and execute 4 more weight scenarios for Task 3.

    # Example 2: Focus on Minimizing Heat Capacity (Cp)
    # model_2 = create_camd_model(weights={'RED': 0.1, 'Cp': 0.8, 'Density': 0.1})
    # solve_and_report(model_2)

    # Example 3: Focus on Maximizing Density
    # model_3 = create_camd_model(weights={'RED': 0.1, 'Cp': 0.1, 'Density': 0.8})
    # solve_and_report(model_3)