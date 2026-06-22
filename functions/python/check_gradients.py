import numpy as np

def check_gradients(x0, fun, jac_analytic):
    """
    Verifies analytic Jacobian against numerical central difference.
    """
    n = len(x0)
    eps = 1e-6
    J_num = np.zeros(n)
    
    # 1. Compute Numerical Gradient
    for i in range(n):
        xp = x0.copy(); xm = x0.copy()
        xp[i] += eps; xm[i] -= eps
        J_num[i] = (fun(xp) - fun(xm)) / (2 * eps)
    
    # 2. Get Analytic Gradient
    _, J_ana = jac_analytic(x0)
    
    # 3. Compare
    diff = np.abs(J_ana - J_num)
    rel_err = diff / (np.abs(J_num) + 1e-12)
    
    print(f"{'Idx':<5} | {'Analytic':<12} | {'Numerical':<12} | {'Rel Error':<10}")
    print("-" * 50)
    for i in range(n):
        print(f"{i:<5} | {J_ana[i]:.6e} | {J_num[i]:.6e} | {rel_err[i]:.6e}")
        
    if np.all(rel_err < 1e-4):
        print("\n✅ Verification Passed: Gradients are consistent.")
    else:
        print("\n❌ Verification Failed: Significant discrepancies found.")

# Usage with your code:
# from run_optimizer3 import objective_and_jac
# check_gradients(x0, lambda x: objective_and_jac(x)[0], objective_and_jac)