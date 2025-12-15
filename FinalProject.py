import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import matplotlib.pyplot as plt
import numpy as np

# --- PROCESS CONSTANTS ---
T_M_MAX = 313.0   # (K) Max Melting Temperature (Absorber condition)
T_B_MIN = 393.0   # (K) Min Boiling Temperature (Desorber condition)

# --- SOLUTE PROPERTIES (CO2) FOR RED CALCULATION ---
# Standard Hansen Solubility Parameters for CO2 (MPa^0.5)
D_D_CO2 = 15.7
D_P_CO2 = 6.3
D_H_CO2 = 5.7
R0_CO2 = 3.5      # Typical Interaction Radius for CO2

# --- GROUP CONTRIBUTION DATA ---
# 1. MW: Molecular Weight (g/mol)
# 2. Tm: Melting Temp Contribution (K)
# 3. Tb: Boiling Temp Contribution (K)
# 4. Vm: Molar Volume Contribution (m3/kmol)
# 5. Fd: Dispersion Contribution (MPa^0.5 * m3/kmol)
# 6. Fp: Polarity Contribution (MPa^0.5 * m3/kmol)
# 7. Fh: Hydrogen Bonding Contribution (MPa^0.5 * m3/kmol)
# 8. Cp: Heat Capacity Contribution (J/mol.K) - From Rayer et al.

GROUP_DATA = {
    # GROUP             [MW,    Tm,    Tb,     Vm,     Fd,    Fp,    Fh,  Cp_Rayer]
    'CH3':              [15.03, 492,   662,    324,    73,    57,    64,   43.56], 
    'CH2':              [14.03, 492,   662,    324,    73,    57,    64,   31.4],
    'NH2 (primary)':    [16.02, 374,   369,    76,     76,    76,    75,   56.47],
    'NH (sec)':         [15.02, 374,   369,    76,     76,    76,    75,   41.05],
    'OH (alcohol)':     [17.01, 1493,  1187,   352,    443,   444,   444,  55.37]
}
GROUPS = list(GROUP_DATA.keys())

# Helper to access data
def get_g(group, index):
    return GROUP_DATA[group][index]

def create_scientific_model():
    """
    Creates the MINLP model using scientific Group Contribution equations.
    """
    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=GROUPS)

    # Variables: Number of groups (Integer, 0 to 10)
    m.n = pyo.Var(m.G, domain=pyo.NonNegativeIntegers, bounds=(0, 10))

    # --- PROPERTY EQUATIONS ---

    # 1. Total Molar Volume (Vm) in m3/kmol
    # Used for density and solubility parameter scaling.
    m.Vm_total = pyo.Expression(expr=sum(m.n[g] * get_g(g, 3) for g in m.G) + 1e-6)

    # 2. Transition Temperatures (Tm, Tb)
    # Using simple additive approximation for robustness: T = Sum(Ni * Ti)
    # (Check if paper uses exp(Sum...) - if values are < 1.0, you might need exp())
    m.Tm = pyo.Expression(expr=sum(m.n[g] * get_g(g, 1) for g in m.G)) 
    m.Tb = pyo.Expression(expr=sum(m.n[g] * get_g(g, 2) for g in m.G))

    # 3. Hansen Solubility Parameters (delta_d, delta_p, delta_h)
    # Formula: delta_i = (Sum Ni * F_i) / Vm_total
    # This ensures the property is INTENSIVE (doesn't grow to infinity with size).
    m.delta_d = pyo.Expression(expr=sum(m.n[g] * get_g(g, 4) for g in m.G) / m.Vm_total)
    m.delta_p = pyo.Expression(expr=sum(m.n[g] * get_g(g, 5) for g in m.G) / m.Vm_total)
    m.delta_h = pyo.Expression(expr=sum(m.n[g] * get_g(g, 6) for g in m.G) / m.Vm_total)

    # 4. RED (Relative Energy Difference)
    # Ra^2 calculation based on distance from CO2 target parameters
    m.Ra_sq = pyo.Expression(expr=
        4 * (m.delta_d - D_D_CO2)**2 +
            (m.delta_p - D_P_CO2)**2 +
            (m.delta_h - D_H_CO2)**2
    )
    # RED = Ra / R0 (Target: RED < 1.0)
    m.RED = pyo.Expression(expr= (m.Ra_sq**0.5) / R0_CO2)

    # 5. Heat Capacity (Cp)
    # Calculated per mole (J/mol.K) then converted to specific heat (J/g.K)
    m.MW_total = pyo.Expression(expr=sum(m.n[g] * get_g(g, 0) for g in m.G) + 1e-3)
    m.Cp_mol = pyo.Expression(expr=sum(m.n[g] * get_g(g, 7) for g in m.G))
    m.Cp_mass = pyo.Expression(expr=m.Cp_mol / m.MW_total) # Result in J/g.K

    # --- CONSTRAINTS ---

    # Process Constraints
    m.C_Tm = pyo.Constraint(expr=m.Tm <= T_M_MAX, doc="Melting Point < 313 K")
    m.C_Tb = pyo.Constraint(expr=m.Tb >= T_B_MIN, doc="Boiling Point > 393 K")
    m.C_RED = pyo.Constraint(expr=m.RED <= 1.0, doc="Miscibility with CO2")

    # Structural Constraints
    # 1. Must be an amine (chemisorption requirement)
    m.C_Amine = pyo.Constraint(expr=m.n['NH2 (primary)'] + m.n['NH (sec)'] >= 1)
    # 2. Minimum molecule size to be a stable liquid
    m.C_Size = pyo.Constraint(expr=sum(m.n[g] for g in m.G) >= 3)
    
    # --- OBJECTIVE FUNCTION ---
    # Multi-objective: Minimize RED (Thermodynamics) and Minimize Cp (Energy)
    # Weights are set to 0.5 each. Cp is normalized by 4.0 to be comparable to RED.
    m.obj = pyo.Objective(expr= 0.5 * m.RED + 0.5 * (m.Cp_mass / 4.0), sense=pyo.minimize)

    return m

def solve_scientific():
    print("\n--- Starting Scientific CAMD Solver via GAMS ---")
    
    # Safety check for empty data
    if GROUP_DATA['CH3'][1] == 0.0:
        print("ALERT: Group Data contains zeros.")
        print("Please update the GROUP_DATA dictionary with values from the Hukkerikar/Rayer papers.")
        return

    model = create_scientific_model()
    
    # CAMBIO IMPORTANTE: Usar la interfaz de GAMS
    # Asegúrate de tener GAMS instalado y en tu PATH del sistema.
    try:
        opt = SolverFactory('gams')
        
        # 'dicopt' es un solver excelente para MINLP (Outer Approximation) incluido en GAMS
        # Si tienes licencia para 'baron', puedes cambiar 'dicopt' por 'baron'.
        res = opt.solve(model, solver='dicopt', tee=True)
        
        if res.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
            print("\n--- OPTIMAL MOLECULE FOUND VIA GAMS! ---")
            print("Structure:")
            for g in model.G:
                val = pyo.value(model.n[g])
                if val > 0.5:
                    print(f"  {g}: {int(round(val))}")
            
            print("\nEstimated Properties:")
            print(f"  MW:  {pyo.value(model.MW_total):.2f} g/mol")
            print(f"  Tm:  {pyo.value(model.Tm):.1f} K")
            print(f"  Tb:  {pyo.value(model.Tb):.1f} K")
            print(f"  Vm:  {pyo.value(model.Vm_total):.4f} m3/kmol")
            print(f"  Cp:  {pyo.value(model.Cp_mass):.3f} J/g.K")
            print(f"  RED: {pyo.value(model.RED):.3f}")
        else:
            print("No feasible solution found via GAMS.")
            
    except Exception as e:
        print(f"Solver Error: {e}")
        print("Tip: Ensure GAMS is installed and readable by Pyomo.")
        print("If you don't have GAMS, try installing glpk/ipopt via conda.")

if __name__ == "__main__":
    solve_scientific()