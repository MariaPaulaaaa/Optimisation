import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

# Constants
T_M_MAX = 313.0   # Maximum Melting Point (K)
T_B_MIN = 393.0   # Minimum Boiling Point (K)
DELTA_CO2 = 21.0  # Target Solubility Parameter (MPa^0.5)

# Group Contribution Data from papers 
#                    [MW, Vm, U, Tm_param, Tb_param, Valency]
GROUP_DATA = {
    'CH3':           [15.03, 33.5,  4710,   -5.10,  23.58, 1],
    'CH2':           [14.03, 16.1,  4940,   11.27,  22.88, 2],
    'NH2 (primary)': [16.02, 19.2,  12550,  66.89,  73.23, 1], 
    'NH (sec)':      [15.02, 4.5,   8370,   52.66,  50.17, 2], # Added for DEA
    'OH (alcohol)':  [17.01, 10.0,  29800,  44.45,  92.88, 1]
}
GROUPS = list(GROUP_DATA.keys())

def create_robust_model():
    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=GROUPS)

    # Variables (up to 10 groups)
    m.n = pyo.Var(m.G, domain=pyo.NonNegativeIntegers, bounds=(0, 10))

    # Expressions
    
    # Joback Melting/Boiling
    m.Tm = pyo.Expression(expr=122.5 + sum(m.n[g] * GROUP_DATA[g][3] for g in m.G))
    m.Tb = pyo.Expression(expr=198.2 + sum(m.n[g] * GROUP_DATA[g][4] for g in m.G))

    # Fedors Solubility (Squared)
    m.U_total = sum(m.n[g] * GROUP_DATA[g][2] for g in m.G)
    m.V_total = sum(m.n[g] * GROUP_DATA[g][1] for g in m.G)
    
    # Delta^2 = U / V
    m.Delta_sq = pyo.Expression(expr=m.U_total / (m.V_total + 1e-6))

    # Constraints

    # 1. Phase Constraints
    m.C_Melting = pyo.Constraint(expr=m.Tm <= T_M_MAX)
    m.C_Boiling = pyo.Constraint(expr=m.Tb >= T_B_MIN)

    # 2. Chemical Structure (Octet Rule)
    # Sum(Ni * (2-Vi)) == 2
    m.C_Structure = pyo.Constraint(
        expr=sum(m.n[g] * (2 - GROUP_DATA[g][5]) for g in m.G) == 2
    )

    # 3. Must be an Amine (Capture Agent)
    m.C_Amine = pyo.Constraint(
        expr=m.n['NH2 (primary)'] + m.n['NH (sec)'] >= 1
    )

    # Objective Function
    # Minimising the difference from Target Solubility + Minimal Weight
    # The difference is squared to make it positive
    
    m.Solubility_Diff = (m.Delta_sq - DELTA_CO2**2)**2
    m.MW = sum(m.n[g] * GROUP_DATA[g][0] for g in m.G)

    # Weights: Prioritize Solubility (1.0), break ties with MW (0.01)
    m.obj = pyo.Objective(expr= 1.0 * m.Solubility_Diff + 0.01 * m.MW, sense=pyo.minimize)

    return m

def solve():
    print("\n--- Solving Robust CAMD Model ---")
    model = create_robust_model()
    opt = SolverFactory('mindtpy')
    
    try:
        # Using Feasibility Pump (FP) to find valid integers first
        res = opt.solve(model, mip_solver='glpk', nlp_solver='ipopt', 
                        strategy='OA', init_strategy='FP', tee=True)
    except Exception as e:
        print(f"Solver Error: {e}")
        return

    if res.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
        print("\n--- SUCCESS: Molecule Found! ---")
        
        tm = pyo.value(model.Tm)
        tb = pyo.value(model.Tb)
        delta = pyo.value(model.Delta_sq)**0.5
        
        print("\nStructure:")
        for g in model.G:
            c = pyo.value(model.n[g])
            if c > 0.1:
                print(f"  {g}: {int(round(c))}")
        
        print(f"\nProperties:")
        print(f"  Melting Point: {tm:.1f} K (<= {T_M_MAX})")
        print(f"  Boiling Point: {tb:.1f} K (>= {T_B_MIN})")
        print(f"  Solubility:    {delta:.1f} MPa^0.5 (Target {DELTA_CO2})")
    else:
        print("Infeasible: No molecule exists with these Phase constraints.")

if __name__ == "__main__":
    solve()