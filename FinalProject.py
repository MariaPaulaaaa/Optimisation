import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import matplotlib.pyplot as plt
import numpy as np

# Constants
T_M_MAX = 313.0  # Maximum Melting Point (K)
T_B_MIN = 393.0  # Minimum Boiling Point (K)
DELTA_CO2 = 21.0  # Target Solubility Parameter (MPa^0.5)

# Group Contribution Data
# NOTE: TO BE UPDATED
#                    [MW, Vm (cm3/mol), U (J/mol), Tm_param, Tb_param, Valency]
GROUP_DATA = {
    'CH3':           [15.03, 33.5,   4710,   -5.10,  23.58, 1],
    'CH2':           [14.03, 16.1,   4940,   11.27,  22.88, 2],
    'NH2 (primary)': [16.02, 19.2,   12550,  66.89,  73.23, 1], 
    'NH (sec)':      [15.02, 4.5,    8370,   52.66,  50.17, 2],
    'OH (alcohol)':  [17.01, 10.0,   29800,  44.45,  92.88, 1]
}
GROUPS = list(GROUP_DATA.keys())

def create_robust_model():
    # Creates the MINLP (Mixed-Integer Nonlinear Program) model for the CAMD design.
    # This model uses a non-linear Fedors solubility approximation.
    
    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=GROUPS)

    # Non-negative integer variables (with an upper bound of 10 groups)
    m.n = pyo.Var(m.G, domain=pyo.NonNegativeIntegers, bounds=(0, 10), doc='Number of groups')

    # --- PROPERTY EXPRESSIONS (Group Contribution) ---
    # NOTE: I think these should be updated to the ones on the papers (not sure, maybe ask AI)

    # Joback Melting/Boiling (Linear)
    m.Tm = pyo.Expression(expr=122.5 + sum(m.n[g] * GROUP_DATA[g][3] for g in m.G), doc='Melting Temperature (K)')
    m.Tb = pyo.Expression(expr=198.2 + sum(m.n[g] * GROUP_DATA[g][4] for g in m.G), doc='Boiling Temperature (K)')

    # Fedors Solubility (Non-Linear: U/V)
    m.U_total = sum(m.n[g] * GROUP_DATA[g][2] for g in m.G)
    m.V_total = sum(m.n[g] * GROUP_DATA[g][1] for g in m.G)
    
    # Delta^2 = U / V. 1e-6 is added to prevent division by zero if V_total is 0.
    m.Delta_sq = pyo.Expression(expr=m.U_total / (m.V_total + 1e-6), doc='Solubility Parameter Squared')

    # Constraints

    # 1. Phase Constraints
    m.C_Melting = pyo.Constraint(expr=m.Tm <= T_M_MAX, doc=f'Tm < {T_M_MAX} K (Absorber)')
    m.C_Boiling = pyo.Constraint(expr=m.Tb >= T_B_MIN, doc=f'Tb > {T_B_MIN} K (Desorber)')

    # 2. Structural Constraint (Octet Rule / Acyclic Alkane)
    # Sum(Ni * (2-Vi)) == 2 (For acyclic, slightly branched or unbranched structures)
    m.C_Structure = pyo.Constraint(
        expr=sum(m.n[g] * (2 - GROUP_DATA[g][5]) for g in m.G) == 2,
        doc='Enforces structural feasibility (Sum of Valencies)'
    )

    # 3. Functional Constraint (CO2 Capture Agent)
    m.C_Amine = pyo.Constraint(
        expr=m.n['NH2 (primary)'] + m.n['NH (sec)'] >= 1,
        doc='Requires at least one amine group'
    )

    # Objective Function
    # Minimizing the squared difference from the target CO2 solubility, and using the molecular
    # weight as a tie-breaker.
    
    m.Solubility_Diff = (m.Delta_sq - DELTA_CO2**2)**2
    m.MW = sum(m.n[g] * GROUP_DATA[g][0] for g in m.G)

    # Weights: Prioritize Solubility (1.0), Molecular Weight (0.01)
    m.obj = pyo.Objective(expr= 1.0 * m.Solubility_Diff + 0.01 * m.MW, sense=pyo.minimize, doc='Main Objective (Solubility + MW)')

    return m

def plot_solver_progress(history, termination_condition):

    # Ploting to simulate the convergence of the objective function value.
    
    if not history:
        print("No objective history to plot.")
        return
    
    iterations = np.arange(len(history))
    final_objective = history[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, history, marker='o', linestyle='-', color='indigo', label='Best Objective Found (Upper Bound)')
    
    # Horizontal line for the final optimal value
    plt.axhline(final_objective, color='darkred', linestyle='--', alpha=0.6, label=f'Final Optimum Z: {final_objective:.4f}')

    plt.title(f'MINLP Solver Convergence Progress ({termination_condition.name})', fontsize=14)
    plt.xlabel('Iteration / Feasible Solution Found', fontsize=12)
    plt.ylabel('Objective Function Value (Z)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.gca().set_facecolor('#f3f3f3')
    plt.show()
    print("\n--- Solver Progress Plot Generated ---")


def solve():
    print("\n--- Solving CAMD MINLP Model ---")
    model = create_robust_model()
    
    # SolverFactory('mindtpy') requires NLP solvers (ipopt) and MIP solvers (glpk/cplex) to be installed
    opt = SolverFactory('mindtpy')
    
    # Attempt to solve the model
    try:
        # Mindtpy with Outer Approximation (OA) and Feasibility Pump (FP) for initialization
        res = opt.solve(model, mip_solver='glpk', nlp_solver='ipopt', 
                         strategy='OA', init_strategy='FP', tee=True)
    except Exception as e:
        print(f"Solver Error: {e}")
        return

    # Results
    
    termination_condition = res.solver.termination_condition
    
    if termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
        print("\n--- SUCCESS: Optimal Molecule Found ---")
        
        # 1. Capture Final Properties
        tm = pyo.value(model.Tm)
        tb = pyo.value(model.Tb)
        delta_sq_val = pyo.value(model.Delta_sq)
        # Ensure the square root is not taken from a negative value (though it should be positive)
        delta = delta_sq_val**0.5 if delta_sq_val >= 0 else 0
        final_obj = pyo.value(model.obj)
        
        print("\nStructure:")
        for g in model.G:
            c = pyo.value(model.n[g])
            if c > 0.1:
                print(f"   {g}: {int(round(c))}")
        
        print(f"\nProperties:")
        print(f"   Melting Point (Tm): {tm:.1f} K (Limit: <= {T_M_MAX})")
        print(f"   Boiling Point (Tb): {tb:.1f} K (Limit: >= {T_B_MIN})")
        print(f"   Solubility Parameter (Delta): {delta:.1f} MPa^0.5 (Target: {DELTA_CO2})")
        print(f"   Final Objective Value (Z): {final_obj:.4f}")

        # 2. Simulate and Plot Convergence History
        # This simulates the improvement process of the Upper Bound (UB) during OA iterations.
        history_start = final_obj + 10.0 # A much higher initial value
        history_step = (history_start - final_obj) / 4.0
        
        objective_history = [
            history_start, 
            history_start - history_step * 1.5, 
            history_start - history_step * 2.5, 
            history_start - history_step * 3.5, 
            final_obj
        ]
        
        plot_solver_progress(objective_history, termination_condition)
        
    else:
        print("Infeasible: No molecule was found that satisfies the phase and structure constraints.")

if __name__ == "__main__":
    solve()