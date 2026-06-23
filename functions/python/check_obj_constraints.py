"""
check_obj_constraints.py
========================
Standalone validation script for the objective and constraint functions
defined in run_optimizer2.py.

Tests performed
---------------
1. Point evaluation   – evaluates objective + constraints at the initial design
                        and prints a human-readable table with physical units.
2. Constraint sign check – verifies sign convention (>= 0 == satisfied).
3. 1-D sensitivity sweeps – perturbs each design variable individually and
                             records how f and every constraint respond.
                             Plots are saved to assignment5/python/outputFig/.
4. Perturbation consistency – compares forward / central finite-difference
                              approximations to detect discontinuities.

Usage
-----
    python check_obj_constraints.py

No optimisation is run.  The environment (waves + wind) is loaded from the
on-disk cache if available, otherwise generated fresh.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── make sure run_optimizer2 is importable ─────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_optimizer2 as r02

# ── output directory ───────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(_HERE))
OUT_DIR = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
os.makedirs(OUT_DIR, exist_ok=True)

# ── constraint labels and physical conversion back from normalised form ─────
CON_NAMES  = ["Surge RMS", "Pitch RMS", "Buoyancy margin", "GM (stability)"]
CON_UNITS  = ["m", "deg", "kg", "m"]
CON_LIMITS = [
    f"<= {r02.SURGE_RMS_MAX:.1f} m",
    f"<= {r02.PITCH_RMS_MAX_DEG:.1f} deg",
    "> 0 kg",
    f"> {r02.GM_MIN:.2f} m",
]

def _phys_constraint(c_vec):
    """Convert the normalised constraint vector back to physical quantities."""
    surge  = "not simulated" if np.isclose(c_vec[0], -10.0) or np.isclose(c_vec[0], -3.0) else f"{r02.SURGE_RMS_MAX * (1.0 - c_vec[0]):.3f}"
    pitch  = "not simulated" if np.isclose(c_vec[1], -10.0) or np.isclose(c_vec[1], -3.0) else f"{r02.PITCH_RMS_MAX_DEG * (1.0 - c_vec[1]):.3f}"
    buoy   = f"{c_vec[2] * r02.BUOYANCY_SCALE + r02.FLOATING_MARGIN:.3f}"
    gm     = f"{c_vec[3] * r02.GM_MIN + r02.GM_MIN:.3f}"
    return [surge, pitch, buoy, gm]

def _var_name(idx):
    n = r02.N_NODES
    if idx < n:              return f"D_node_{idx+1} [m]"
    if idx < 2 * n:          return f"L_frac_{idx-n+1}"
    if idx == 2 * n:         return "Draft [m]"
    if idx == 2 * n + 1:     return "Ballast mass [kg]"
    return "Unknown"

# ── reference design ───────────────────────────────────────────────────────
def _make_x0():
    n = r02.N_NODES
    D0 = [14.0] * n
    L0 = [1.0 / n] * n
    return np.array([*D0, *L0, 120.0, 8e6], dtype=float)


# =============================================================================
# 1.  POINT EVALUATION
# =============================================================================
def check_point(x0, c_vec, obj_val, waves, wind, t_int, t_trans):
    """Print a detailed table for one design point."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  OBJECTIVE + CONSTRAINT POINT EVALUATION")
    print(sep)

    # --- Objective ---
    D, L, draft, ballast = r02._split_design(x0)
    s = r02.build_structure(draft, D, L, ballast, r02.SPAR0)
    print(f"\n  Objective: Steel mass + Ballast mass = {obj_val:.4e} kg")
    print(f"    Steel shell mass  : {s.ms:.4e} kg  ({s.ms/1e3:.1f} t)")
    print(f"    Ballast mass      : {s.M_Ballast:.4e} kg  ({s.M_Ballast/1e3:.1f} t)")
    print(f"    Draft             : {draft:.2f} m")
    print(f"    Mean diameter     : {float(np.mean(D)):.2f} m")

    # --- Structural feasibility (no simulation) ---
    buoyancy_margin = s.buoyant_mass - s.MTot
    gm_val          = (s.IAA / max(s.V_disp, 1e-9)) + s.zCB - s.zCM_Tot
    print(f"\n  Structural feasibility (static, no simulation):")
    print(f"    Buoyancy margin   : {buoyancy_margin:.3e} kg  "
          f"({'OK' if buoyancy_margin > 0 else 'VIOLATED'})")
    print(f"    GM                : {gm_val:.4f} m  "
          f"({'OK' if gm_val >= r02.GM_MIN else 'VIOLATED'}, limit >= {r02.GM_MIN} m)")
    print(f"    zCB               : {s.zCB:.3f} m")
    print(f"    zCM_Tot           : {s.zCM_Tot:.3f} m")

    # --- Constraint vector ---
    phys = _phys_constraint(c_vec)
    print(f"\n  {'Constraint':<20} {'Normalised c':>14} {'Physical value':>16} "
          f"{'Limit':>22} {'Status':>10}")
    print(f"  {'-'*84}")
    for name, unit, limit, cn, pv in zip(CON_NAMES, CON_UNITS, CON_LIMITS, c_vec, phys):
        status = "OK" if cn >= 0 else "VIOLATED"
        unit_text = "" if pv == "not simulated" else unit
        print(f"  {name:<20} {cn:>14.4f} {pv:>14} {unit_text:<2} {limit:>22} {status:>10}")

    feasible = all(cn >= 0 for cn in c_vec)
    print(f"\n  Overall feasibility: {'FEASIBLE' if feasible else 'INFEASIBLE'}")
    print(sep)


# =============================================================================
# 2.  SIGN CONVENTION CHECK
# =============================================================================
def check_sign_convention(c_vec):
    """Assert that each constraint with c >= 0 is indeed satisfied physically."""
    print("\n  Sign convention check (c >= 0 must mean satisfied):")
    all_ok = True
    for i, (name, cn) in enumerate(zip(CON_NAMES, c_vec)):
        if cn >= 0:
            msg = "  satisfied (c>=0) [OK]"
        else:
            msg = "  violated  (c<0)  - check physical limit!"
            all_ok = False
        print(f"    c[{i}] = {cn:+.4f}  [{name}]{msg}")
    return all_ok


# =============================================================================
# 3.  1-D SENSITIVITY SWEEPS
# =============================================================================
def sweep_variable(idx, x0, waves, wind, t_int, t_trans, n_pts=12):
    """
    Vary design variable `idx` over a symmetric ±20 % range around x0[idx]
    and record objective + all constraint values.

    Returns
    -------
    x_sweep : (n_pts,) array of variable values
    f_vals  : (n_pts,) array of objective values
    c_vals  : (n_pts, 4) array of constraint values
    """
    xc = x0[idx]
    # choose a sensible sweep range
    if idx < r02.N_NODES:                   # diameter
        lo, hi = max(xc * 0.6, 4.0), min(xc * 1.4, r02.UPPER_BOUND_D)
    elif idx < 2 * r02.N_NODES:             # length fractions
        lo, hi = max(xc * 0.5, 0.03), min(xc * 2.0, 0.35)
    elif idx == 2 * r02.N_NODES:            # draft
        lo, hi = max(xc * 0.8, r02.MIN_DRAFT), min(xc * 1.2, r02.MAX_DRAFT)
    else:                                   # ballast mass
        lo, hi = max(xc * 0.5, 0.0), min(xc * 2.0, r02.MAX_BALLAST_MASS)

    x_sweep = np.linspace(lo, hi, n_pts)
    f_vals  = np.zeros(n_pts)
    c_vals  = np.zeros((n_pts, 4))

    for k, val in enumerate(x_sweep):
        xk = x0.copy()
        xk[idx] = val
        f_vals[k]   = r02.objective(xk)
        c_vals[k, :] = r02.constraints_fun(
            xk, waves, wind, t_int, t_trans
        )

    return x_sweep, f_vals, c_vals


def plot_sweep(idx, x_sweep, f_vals, c_vals):
    """Save a figure with objective and constraint sweeps for one variable."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    var = _var_name(idx)

    # — Objective —
    ax_f = axes[0]
    ax_f.plot(x_sweep, f_vals / 1e6, 'b-o', markersize=4, label='Objective [10⁶ kg]')
    ax_f.axvline(x_sweep[len(x_sweep) // 2], color='gray', linestyle='--',
                 linewidth=0.8, label='Reference value')
    ax_f.set_ylabel('Objective [1e6 kg]')
    ax_f.set_title(f'1-D sweep: {var}')
    ax_f.legend(fontsize=8)
    ax_f.grid(True, alpha=0.3)

    # — Constraints —
    ax_c = axes[1]
    colours = ['steelblue', 'darkorange', 'seagreen', 'crimson']
    for j, (name, col) in enumerate(zip(CON_NAMES, colours)):
        ax_c.plot(x_sweep, c_vals[:, j], color=col, marker='s',
                  markersize=4, linewidth=1.4, label=f'c[{j}] {name}')
    ax_c.axhline(0.0, color='k', linewidth=1.2, linestyle='-', label='Feasibility boundary (c=0)')
    ax_c.axvline(x_sweep[len(x_sweep) // 2], color='gray', linestyle='--', linewidth=0.8)
    ax_c.set_xlabel(var)
    ax_c.set_ylabel('Normalised constraint value')
    ax_c.legend(fontsize=7, ncol=2)
    ax_c.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f'check_obj_con_sweep_var{idx:02d}.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


# =============================================================================
# 4.  PERTURBATION CONSISTENCY  (forward vs central FD)
# =============================================================================
def check_perturbation_consistency(x0, waves, wind, t_int, t_trans,
                                   eps=1e-3, var_indices=None):
    """
    For each selected variable, compare
        forward  : [f(x+e) - f(x)] / e
        central  : [f(x+e) - f(x-e)] / (2e)
    and flag large discrepancies that hint at kinks or discontinuities.
    """
    n_total = len(x0)
    if var_indices is None:
        # By default check one representative from each group
        var_indices = [0, r02.N_NODES, 2 * r02.N_NODES, 2 * r02.N_NODES + 1]

    f0 = r02.objective(x0)
    c0 = r02.constraints_fun(x0, waves, wind, t_int, t_trans)

    print(f"\n  Perturbation consistency check (eps = {eps:.1e})")
    print(f"  {'Var':<26} {'fwd_obj':>12} {'cen_obj':>12} "
          f"{'|d|_obj':>10} {'fwd_c0':>10} {'cen_c0':>10} {'|d|_c0':>10}")
    print(f"  {'-'*100}")

    all_consistent = True
    for idx in var_indices:
        xp = x0.copy(); xp[idx] += eps
        xm = x0.copy(); xm[idx] -= eps

        fp = r02.objective(xp)
        fm = r02.objective(xm)
        cp = r02.constraints_fun(xp, waves, wind, t_int, t_trans)
        cm = r02.constraints_fun(xm, waves, wind, t_int, t_trans)

        fwd_obj = (fp - f0) / eps
        cen_obj = (fp - fm) / (2.0 * eps)
        fwd_c0  = (cp[0] - c0[0]) / eps
        cen_c0  = (cp[0] - cm[0]) / (2.0 * eps)

        delta_obj = abs(fwd_obj - cen_obj) / (abs(cen_obj) + 1e-12)
        delta_c0  = abs(fwd_c0  - cen_c0)  / (abs(cen_c0)  + 1e-12)

        flag = " <-- INCONSISTENT" if (delta_obj > 0.5 or delta_c0 > 0.5) else ""
        if flag:
            all_consistent = False

        print(f"  {_var_name(idx):<26} {fwd_obj:>12.4e} {cen_obj:>12.4e} "
              f"{delta_obj:>10.3e} {fwd_c0:>10.4e} {cen_c0:>10.4e} "
              f"{delta_c0:>10.3e}{flag}")

    if all_consistent:
        print("\n  Result: All checked variables are consistent (no discontinuities detected).")
    else:
        print("\n  Result: Some variables show inconsistencies - investigate kinks!")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  check_obj_constraints.py - Validation of f and c")
    print("=" * 70)

    # ── environment ────────────────────────────────────────────────────────
    print("\n[ENV] Loading/generating environment (waves + wind)...")
    waves = r02._load_or_compute_waves()
    wind  = r02._load_or_compute_wind()

    # Use a short integration window to keep runtime low during validation
    T_VAL       = 300.0   # s  (enough to see dynamics, not full optimisation)
    DT          = 1.0     # s
    T_TRANS     = 100.0   # s  transient to discard
    t_int       = np.arange(0.0, T_VAL, 2.0 * DT)   # matches constraints_fun convention

    print(f"[ENV] Integration window: {T_VAL:.0f} s  (transient: {T_TRANS:.0f} s)\n")

    # ── reference point ────────────────────────────────────────────────────
    x0 = _make_x0()
    print(f"  Reference design x0  (D={x0[0]:.1f} m, draft={x0[2*r02.N_NODES]:.0f} m,"
          f" ballast={x0[-1]/1e6:.1f} Mt)")

    # ── evaluate ───────────────────────────────────────────────────────────
    obj_val = r02.objective(x0)
    c_vec   = r02.constraints_fun(x0, waves, wind, t_int, T_TRANS)

    # 1. Point evaluation
    check_point(x0, c_vec, obj_val, waves, wind, t_int, T_TRANS)

    # 2. Sign convention
    check_sign_convention(c_vec)

    # 3. 1-D sweeps for selected variables
    sweep_vars = [
        0,                        # first diameter node
        r02.N_NODES,              # first length fraction
        2 * r02.N_NODES,          # draft
        2 * r02.N_NODES + 1,      # ballast mass
    ]
    print(f"\n  Running 1-D sweeps for {len(sweep_vars)} representative variables ...")
    saved_figs = []
    for idx in sweep_vars:
        print(f"    Sweeping {_var_name(idx)} ...", end="", flush=True)
        xs, fs, cs = sweep_variable(idx, x0, waves, wind, t_int, T_TRANS)
        fname = plot_sweep(idx, xs, fs, cs)
        saved_figs.append(fname)

        # Quick monotonicity / smoothness flag
        if np.any(np.isnan(fs)) or np.any(np.isnan(cs)):
            print(f" WARNING: NaN detected!")
        elif np.any(np.isinf(fs)) or np.any(np.isinf(cs)):
            print(f" WARNING: Inf detected!")
        else:
            print(f" OK  -> {fname}")

    print(f"\n  Sweep figures saved to: {OUT_DIR}")

    # 4. Perturbation consistency
    check_perturbation_consistency(x0, waves, wind, t_int, T_TRANS,
                                   eps=1e-2, var_indices=sweep_vars)

    # ── final summary ──────────────────────────────────────────────────────
    feasible = all(c >= 0 for c in c_vec)
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"    Objective at x0    : {obj_val:.4e} kg  ({obj_val/1e6:.2f} Mt)")
    print(f"    Reference feasible : {'YES' if feasible else 'NO  (expected for a raw initial guess)'}")
    print(f"    Saved sweep plots  : {len(saved_figs)}")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
