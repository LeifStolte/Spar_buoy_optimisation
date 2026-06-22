import numpy as np
import os
import matplotlib.pyplot as plt
import run_optimizer2 as r02

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_variable_name(idx):
    """Maps index to readable variable name."""
    if idx < r02.N_NODES: return f"Diameter Node {idx}"
    if idx < 2 * r02.N_NODES: return f"Length Frac Node {idx - r02.N_NODES}"
    if idx == 2 * r02.N_NODES: return "Draft"
    if idx == 2 * r02.N_NODES + 1: return "Ballast Height"
    return "Unknown"

def check_stepsize(idx=0):
    # Setup state
    initial_diameters = [14.0] * r02.N_NODES
    initial_fractions = [1.0 / r02.N_NODES] * r02.N_NODES
    x0 = np.array([*initial_diameters, *initial_fractions, 120.0, 15.0], dtype=float)
    f = r02.objective
    
    # 1. Calculate Truth (Central diff at 1e-9)
    eps_truth = 1e-9
    x_plus = x0.copy(); x_plus[idx] += eps_truth
    x_minus = x0.copy(); x_minus[idx] -= eps_truth
    truth = (f(x_plus) - f(x_minus)) / (2 * eps_truth)
    
    # 2. Iterate epsilon range
    epsilons = np.logspace(-16, -1, 20)
    forward, backward, central = [], [], []
    
    for e in epsilons:
        xp = x0.copy(); xp[idx] += e
        xm = x0.copy(); xm[idx] -= e
        
        forward.append(abs((f(xp) - f(x0)) / e - truth))
        backward.append(abs((f(x0) - f(xm)) / e - truth))
        central.append(abs((f(xp) - f(xm)) / (2 * e) - truth))
        
    # 3. Plotting
    plt.figure(figsize=(8, 5))
    plt.loglog(epsilons, forward, label='Forward', marker='o', markersize=4)
    plt.loglog(epsilons, backward, label='Backward', marker='s', markersize=4)
    plt.loglog(epsilons, central, label='Central', marker='^', markersize=4)
    
    var_name = get_variable_name(idx)
    plt.xlabel('Step Size (epsilon)')
    plt.ylabel('Absolute Error')
    plt.title(f'Gradient Sensitivity: {var_name}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    out_dir = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'stepsize_check_{idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Total parameters: 10 (D) + 10 (L_frac) + 1 (Draft) + 1 (Ballast) = 22
    total_vars = 2 * r02.N_NODES + 2
    print(f"Analyzing {total_vars} variables...")
    
    for i in range(total_vars):
        print(f"Checking index {i}: {get_variable_name(i)}")
        check_stepsize(idx=i)
    
    print("Done. Plots saved to 'outputFig/' folder.")