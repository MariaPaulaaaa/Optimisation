import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

# --- Hukkerikar Model Constants (GC+) ---
# Source: Hukkerikar et al. (2012), Table 5
T_M0 = 143.5706   # Melting Temperature, K
T_B0 = 244.5165   # Boiling Temperature, K
V_M0 = 0.0160     # Molar Volume, m3/kmol

# --- Process Constraints ---
# Source: Project Brief
T_M_MAX = 313.0  # Melting Temperature, K
T_B_MIN = 393.0  # Boiling Temperature, K

# --- First-order groups, step-wise regression method ---
# Source: Hukkerikar (2012) & Rayer (2011)
GROUP_DATA = {
    # GROUP             [MW,    Tm,      Tb,      Vm,      Fd,      Fp,       Fh,      Cp_Rayer, Valency]
    #                   g/mol   (C_Tm)   (C_Tb)   m3/kmol  MPa^.5   MPa^.5    MPa^.5   J/mol.K   (-)
    'CH3':              [15.03, 0.6699,  0.8853,  0.0241,  7.5697,  1.9996,   2.2105,  43.56,    1], 
    'CH2':              [14.03, 0.2992,  0.5815,  0.0165, -0.0018, -0.1492,  -0.2150,  31.40,    2],
    'NH2 (primary)':    [16.02, 3.4368,  2.3212,  0.0281,  8.1717,  5.2964,   6.7984,  56.47,    1], # CH2NH2
    'NH (sec)':         [15.02, 2.0673,  1.3838,  0.0260,  0.2374,  0.1072,   1.4183,  41.05,    2], # CH2NH
    'OH (alcohol)':     [17.01, 3.2702,  2.1385,  0.0044,  8.0236,  4.9598,  11.8005,  55.37,    1]
}
GROUPS = list(GROUP_DATA.keys())

# CO2 Properties for RED calculation
# Source: Hansen (2007)
D_D_CO2, D_P_CO2, D_H_CO2, R0_CO2 = 15.7, 6.3, 5.7, 3.3

def get_g(group, index):
    return GROUP_DATA[group][index]

def create_model():
    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=GROUPS)
    
    # Variables: Number of groups (0 to 15)
    # Initialize=1 prevents starting with all zeros, as log(0) will display an error
    # Increased upper bound to 15 to allow larger molecules if needed.
    m.n = pyo.Var(m.G, domain=pyo.NonNegativeIntegers, bounds=(0, 15), initialize=1)

    # Temperatures (Hukkerikar Logarithmic Model)
    # Tm = T_m0 * ln( Sum Ni * C_Tmi )
    m.Tm_sum = pyo.Expression(expr=sum(m.n[g] * get_g(g, 1) for g in m.G) + 0.001)
    m.Tb_sum = pyo.Expression(expr=sum(m.n[g] * get_g(g, 2) for g in m.G) + 0.001)
    
    m.Tm = pyo.Expression(expr=T_M0 * pyo.log(m.Tm_sum))
    m.Tb = pyo.Expression(expr=T_B0 * pyo.log(m.Tb_sum))

    # 2. Molar Volume (m3/kmol)
    # Vm = Vm0 + Sum(Ni * Ci)
    m.Vm = pyo.Expression(expr=V_M0 + sum(m.n[g] * get_g(g, 3) for g in m.G))
    
    # 3. Solubility Parameters (Hansen)
    # d = Sum Ni * Fi
    m.dd = pyo.Expression(expr=sum(m.n[g] * get_g(g, 4) for g in m.G))
    m.dp = pyo.Expression(expr=sum(m.n[g] * get_g(g, 5) for g in m.G))
    m.dh = pyo.Expression(expr=sum(m.n[g] * get_g(g, 6) for g in m.G))

    # 4. RED (Relative Energy Difference)
    m.Ra2 = pyo.Expression(expr=4*(m.dd - D_D_CO2)**2 + (m.dp - D_P_CO2)**2 + (m.dh - D_H_CO2)**2)
    m.RED = pyo.Expression(expr=(m.Ra2**0.5) / R0_CO2)

    # 5. Specific Heat Capacity (Cp)
    # Adding 1e-6 to avoid division by zero if all n=0
    m.MW = pyo.Expression(expr=sum(m.n[g] * get_g(g, 0) for g in m.G) + 1e-6)
    m.Cp_mol = pyo.Expression(expr=sum(m.n[g] * get_g(g, 7) for g in m.G))
    m.Cp_mass = pyo.Expression(expr=m.Cp_mol / m.MW)

    # --- Constraints ---
    m.C_Tm = pyo.Constraint(expr=m.Tm <= T_M_MAX)
    m.C_Tb = pyo.Constraint(expr=m.Tb >= T_B_MIN)
    m.C_RED = pyo.Constraint(expr=m.RED <= 1.0) 

    m.C_Amine = pyo.Constraint(expr=m.n['NH2 (primary)'] + m.n['NH (sec)'] >= 1)
    m.C_Struc = pyo.Constraint(expr=sum(m.n[g] for g in m.G) >= 3) # Minimum 3 groups

    # Valence/Structural Feasibility Rule
    # Sum(Ni * (2 - Vi)) = 2
    m.C_Valence = pyo.Constraint(expr=sum(m.n[g] * (2 - get_g(g, 8)) for g in m.G) == 2)

    # --- Objective ---
    # Minimize Cp and RED
    m.obj = pyo.Objective(expr=0.5*m.RED + 0.5*(m.Cp_mass/4.0), sense=pyo.minimize)

    return m

def solve():
    print("\n--- Running GAMS Solver ---")
    model = create_model()
    opt = SolverFactory('gams')
    
    try:
        # Using DICOPT (MINLP)
        res = opt.solve(model, solver='dicopt', tee=True)
        
        if res.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
            print("\n--- MOLECULE FOUND! ---")
            print("Groups:")
            for g in model.G:
                val = pyo.value(model.n[g])
                if val > 0.1: print(f"  {g}: {int(val)}")
            
            print(f"\nProperties:")
            print(f"  Tm: {pyo.value(model.Tm):.1f} K (<= {T_M_MAX})")
            print(f"  Tb: {pyo.value(model.Tb):.1f} K (>= {T_B_MIN})")
            print(f"  Cp: {pyo.value(model.Cp_mass):.3f} J/g.K")
            print(f"  RED: {pyo.value(model.RED):.3f}")
        else:
            print("No feasible solution found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    solve()