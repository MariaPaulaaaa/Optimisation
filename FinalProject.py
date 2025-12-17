import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import pandas as pd

# Hukkerikar Model Constants (GC+)
# Source: Hukkerikar et al. (2012), Table 5
T_M0 = 143.5706   # Melting Temperature, K
T_B0 = 244.5165   # Boiling Temperature, K
V_M0 = 0.0160     # Molar Volume, m3/kmol

# Process Constraints
# Source: Project Brief
T_M_MAX = 313.0  # Melting Temperature, K
T_B_MIN = 393.0  # Boiling Temperature, K
RED_MAX = 1.0    # RED

# First-order groups, step-wise regression method
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

# Scaling Ranges
# [Min, Max] ranges are defined to normalise properties between 0 and 1.
SCALING = {
    'RED': {'min': 0.0, 'max': 3.0},    # Dimensionless
    'Cp':  {'min': 1.0, 'max': 4.2},    # J/g.K
    'Rho': {'min': 700.0, 'max': 1600.0} # kg/m3
}

def get_g(group, index):
    return GROUP_DATA[group][index]

def create_model(weights, mode):
    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=GROUPS)
    
    # Variables: Number of groups (0 to 15)
    # Initialize=1 prevents starting with all zeros, as log(0) will display an error
    m.n = pyo.Var(m.G, domain=pyo.NonNegativeIntegers, bounds=(0, 15), initialize=1)

    # Slack Varbiales - Used when changing weights
    m.s_Tm = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
    m.s_Tb = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
    m.s_RED = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
    m.s_Design = pyo.Var(domain=pyo.NonNegativeReals, initialize=0) # Slack for forced designs

    # 1. Temperatures (Hukkerikar Logarithmic Model)
    # Tm = T_m0 * ln( Sum Ni * C_Tmi )
    m.Tm_sum = pyo.Expression(expr=sum(m.n[g] * get_g(g, 1) for g in m.G) + 0.001)
    m.Tb_sum = pyo.Expression(expr=sum(m.n[g] * get_g(g, 2) for g in m.G) + 0.001)
    
    m.Tm = pyo.Expression(expr=T_M0 * pyo.log(m.Tm_sum))
    m.Tb = pyo.Expression(expr=T_B0 * pyo.log(m.Tb_sum))

    # Ensuring log argument is >= 1.1 to avoid negative Kelvin temps
    m.C_LogSafetyTm = pyo.Constraint(expr=m.Tm_sum >= 1.1)
    m.C_LogSafetyTb = pyo.Constraint(expr=m.Tb_sum >= 1.1)

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

    # 5. Molecular Weight (g/mol or kg/kmol)
    # Adding 1e-6 to avoid division by zero if all n=0
    m.MW = pyo.Expression(expr=sum(m.n[g] * get_g(g, 0) for g in m.G) + 1e-6)

    # 6. Specific Heat Capacity (Cp)
    m.Cp_mol = pyo.Expression(expr=sum(m.n[g] * get_g(g, 7) for g in m.G))
    m.Cp_mass = pyo.Expression(expr=m.Cp_mol / m.MW)

    # 7. Density (rho)
    # rho = MW / Vm
    m.Rho = pyo.Expression(expr=m.MW / m.Vm)

    # 8. Scaling
    # Scaled = (Val - Min) / (Max - Min)
    m.Z_RED = pyo.Expression(expr=(m.RED - SCALING['RED']['min']) / (SCALING['RED']['max'] - SCALING['RED']['min']))
    m.Z_Cp  = pyo.Expression(expr=(m.Cp_mass - SCALING['Cp']['min']) / (SCALING['Cp']['max'] - SCALING['Cp']['min']))
    # Maximize Rho = Minimize (Max - Rho)
    m.Z_Rho = pyo.Expression(expr=(SCALING['Rho']['max'] - m.Rho) / (SCALING['Rho']['max'] - SCALING['Rho']['min']))

    # Softened Constraints - Used when changing weights
    m.C_Tm = pyo.Constraint(expr=m.Tm <= T_M_MAX + m.s_Tm)
    m.C_Tb = pyo.Constraint(expr=m.Tb >= T_B_MIN - m.s_Tb)

    # Structural Constraints (Hard)
    m.C_Valence = pyo.Constraint(expr=sum(m.n[g] * (2 - get_g(g, 8)) for g in m.G) == 2)
    m.C_MinSize = pyo.Constraint(expr=sum(m.n[g] for g in m.G) >= 3)

    # Different weight scenarios (Q3)
    # Forcing modes and relaxed constraints are being added as the solver was getting the same 
    # results even though weights changed
    if mode == 'Force_OH':
        # Must have OH
        m.C_Design = pyo.Constraint(expr=1 <= m.n['OH (alcohol)'] + m.s_Design)
        # Relaxed size limit to find a feasible chain
        m.C_Size = pyo.Constraint(expr=sum(m.n[g] for g in m.G) <= 10)
        
    elif mode == 'Force_Short':
        # Goal: Short chain <= 4
        m.C_Size = pyo.Constraint(expr=sum(m.n[g] for g in m.G) <= 4 + m.s_Design)
        m.C_Design = pyo.Constraint(expr=m.s_Design == 0) # Dummy
        
    elif mode == 'Force_Primary':
        m.C_Design = pyo.Constraint(expr=m.n['NH (sec)'] <= 0 + m.s_Design)
        m.C_Size = pyo.Constraint(expr=sum(m.n[g] for g in m.G) <= 8)
        
    elif mode == 'Force_Long':
        m.C_Design = pyo.Constraint(expr=8 <= m.n['CH2'] + m.s_Design)
        m.C_Size = pyo.Constraint(expr=sum(m.n[g] for g in m.G) <= 15)
        
    else: # Base Case (no relaxed contraints, must follow the brief)
        m.C_Size = pyo.Constraint(expr=sum(m.n[g] for g in m.G) <= 7)
        m.C_Design = pyo.Constraint(expr=m.s_Design == 0)

    # Objective Function
    w_red, w_cp, w_rho = weights
    penalty = 1000.0
    # Lower penalty for design slacks to avoid numerical shock
    design_penalty = 500.0 
    
    m.obj = pyo.Objective(expr=
        w_red*m.Z_RED + w_cp*m.Z_Cp + w_rho*m.Z_Rho + 
        penalty*(m.s_Tm + m.s_Tb + m.s_RED) +
        design_penalty*m.s_Design,
        sense=pyo.minimize)

    return m

def solve_scenarios():
    # (RED, Cp, rho), Mode, Description
    scenarios = [
        ((1/3, 1/3, 1/3),    'Base',          "Base Case"),
        ((0.90, 0.05, 0.05), 'Force_OH',      "Solubility Priority"),
        ((0.05, 0.90, 0.05), 'Force_Long',    "Energy Efficiency Priority"),
        ((0.10, 0.10, 0.80), 'Force_Short',   "Density Priority"),
        ((0.50, 0.50, 0.00), 'Force_Primary', "Thermo Trade-Off"),
        ((0.45, 0.1, 0.45),  'Force_Short',   "CAPEX")
    ]
   

    results_list = []
    opt = SolverFactory('gams')

    print("\nRunning Scenarios")

    for i, (w, mode, desc) in enumerate(scenarios):
        print(f"\nScenario {i+1}: {desc}")
        model = create_model(w, mode)
        
        try:
            res = opt.solve(model, solver='dicopt', tee=False)
            
            # Retrieve values regardless of optimal status if feasible
            struct = ""
            for g in model.G:
                val = int(round(pyo.value(model.n[g])))
                if val > 0: struct += f"{val}{g} "
            
            # Violation Check
            
            tm_val = pyo.value(model.Tm)
            tb_val = pyo.value(model.Tb)
            red_val = pyo.value(model.RED)
            rho_val = pyo.value(model.Rho)
            
            notes = []
            if tm_val > T_M_MAX: 
                notes.append(f"Tm High (+{tm_val - T_M_MAX:.1f}K)")
            
            if tb_val < T_B_MIN: 
                notes.append(f"Tb Low (-{T_B_MIN - tb_val:.1f}K)")
            
            if red_val > RED_MAX: 
                notes.append(f"RED High (+{red_val - RED_MAX:.2f})")
            
            # Check Density (Soft Warning)
            if rho_val < 700:
                notes.append(f"Low Rho ({rho_val:.0f})")

            # Check Design Compliance (Slack check is enough here)
            if pyo.value(model.s_Design) > 0.1: 
                notes.append("Design Unmet")

            note_str = ", ".join(notes) if notes else "Ok"

            results_list.append({
                'Scenario': i+1,
                'Structure': struct,
                'RED': round(red_val, 2),
                'Cp': round(pyo.value(model.Cp_mass), 2),
                'Tb': round(tb_val, 1),
                'Tm': round(tm_val, 1),
                'Rho': round(rho_val, 1),
                'Notes': note_str
            })
            print(f"     Found: {struct}")

        except Exception as e:
            print(f"    Error: {e}")
            results_list.append({'Scenario': i+1, 'Structure': 'Solver Crash'})

    print("\n" + "="*120)
    print("FINAL RESULTS")
    print("="*120)
    df = pd.DataFrame(results_list)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))

if __name__ == "__main__":

    solve_scenarios()