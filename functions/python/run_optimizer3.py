"""
Lightweight optimizer wrapper that reuses existing simulation code but
adds MATLAB-inspired scaling and an analytic objective gradient for
steel + ballast mass w.r.t. diameters, thickness and ballast height.

Usage: run this file from the `functions/python` folder or via
`python -m functions.python.run_optimizer3` (adjust PYTHONPATH accordingly).
"""

import argparse
import os
import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

# Import the existing runner to reuse build_structure and helpers
import runpy
import types
R2_path = os.path.join(os.path.dirname(__file__), 'RUN_OPTIMIZER2.PY')
R2_globals = runpy.run_path(R2_path)
R2 = types.SimpleNamespace(**R2_globals)

# Short-hands and constants copied from RUN_OPTIMIZER2
N_NODES = R2.N_NODES
COST_STEEL_PER_KG = R2.COST_STEEL_PER_KG
COST_CONCRETE_PER_KG = R2.COST_CONCRETE_PER_KG
RHO_CONCRETE = R2.RHO_CONCRETE

EPS_FD = 1e-6


def _split_design(x):
    return R2._split_design(x)


# Simple scaling scaffolding so we can expand later (currently identity)
class Scaling:
    def __init__(self):
        # characteristic mass and variables - placeholders
        self.char_m = 1.0
        self.char_var = np.ones(3)


def objective_and_jac(x):
    """Return objective (cost in same units as R2.objective) and analytic jac.
    Analytic derivatives implemented for D_nodes, th_nodes and ballast_height.
    Other variables (L_fracs, draft) use finite differences.
    """
    x = np.asarray(x, dtype=float).ravel()
    D_nodes, th_nodes, L_fracs, draft, ballast_height = _split_design(x)

    # replicate section geometry used in R2.build_structure (consistent)
    L_fracs = np.maximum(np.asarray(L_fracs, dtype=float), 1e-9)
    L_fracs = L_fracs / np.sum(L_fracs)
    section_lengths = L_fracs * float(draft)

    zbot = -float(draft)
    z_edges = np.zeros(N_NODES + 1)
    z_edges[0] = zbot
    for i in range(N_NODES):
        z_edges[i+1] = z_edges[i] + section_lengths[i]

    # step profile
    D_nodes_s = R2._step_profile(D_nodes, N_NODES)
    th_nodes_s = R2._step_profile(th_nodes, N_NODES)

    rhos = float((R2.SPAR0.get('rho_Steel', None) or 7850.0))

    # steel mass per section: shell_area * rho * length
    # shell_area = pi*(D*t - t^2)  (derived from outer-inner simplification)
    outer_area = np.pi * (D_nodes_s / 2.0) ** 2
    inner_area = np.pi * np.maximum((D_nodes_s - 2.0 * th_nodes_s) / 2.0, 0.0) ** 2
    shell_area = outer_area - inner_area
    section_steel_masses = shell_area * rhos * section_lengths

    ms = float(np.sum(section_steel_masses))

    # ballast per section (depends on ballast_height)
    ballast_top = zbot + float(np.clip(ballast_height, 0.0, float(draft) * 0.999))
    ballast_mass = 0.0
    h_in_section = np.zeros(N_NODES)
    inner_diam = np.maximum(D_nodes_s - 2.0 * th_nodes_s, 0.0)
    inner_area_section = (np.pi / 4.0) * (inner_diam ** 2)
    for i in range(N_NODES):
        z_low = z_edges[i]
        z_high = z_edges[i+1]
        h = max(0.0, min(z_high, ballast_top) - z_low)
        h_in_section[i] = h
        ballast_mass += RHO_CONCRETE * inner_area_section[i] * h

    total_mass = ms + ballast_mass

    # objective scaling and cost units identical to R2.objective
    obj = (ms * COST_STEEL_PER_KG + ballast_mass * COST_CONCRETE_PER_KG) / 1e6

    # Gradient vector (same ordering as x): D_nodes (N), th_nodes (N), L_fracs (N), draft (1), ballast_height (1)
    grad = np.zeros_like(x)

    # analytic parts for D and th and ballast_height
    # d(ms_section)/dD = pi * rhos * t * L
    # d(ms_section)/dt = pi * rhos * (D - 2t) * L
    # ballast_mass_section = rho_concrete * (pi/4) * (D - 2t)^2 * h
    # d(ballast)/dD = rho_concrete * (pi/2) * (D - 2t) * h
    # d(ballast)/dt = -rho_concrete * (pi/2) * (D - 2t) * h

    # Precompute
    L = section_lengths
    t = th_nodes_s
    D = D_nodes_s
    h = h_in_section

    coef_steel = np.pi * rhos
    coef_ballast = RHO_CONCRETE * (np.pi / 2.0)

    dms_dD = coef_steel * t * L
    dms_dt = coef_steel * (D - 2.0 * t) * L

    dballast_dD = coef_ballast * (D - 2.0 * t) * h
    dballast_dt = -coef_ballast * (D - 2.0 * t) * h

    # assemble into gradient (apply cost per kg and scaling 1e6)
    factor = 1.0 / 1e6
    for i in range(N_NODES):
        grad[i] = (dms_dD[i] * COST_STEEL_PER_KG + dballast_dD[i] * COST_CONCRETE_PER_KG) * factor
        grad[N_NODES + i] = (dms_dt[i] * COST_STEEL_PER_KG + dballast_dt[i] * COST_CONCRETE_PER_KG) * factor

    # Analytic derivatives for L_fracs (raw variables a) and draft
    # a = raw L_fracs values in x; actual normalized L = a / s
    idx_L_start = 2 * N_NODES
    a = np.asarray(x[idx_L_start:idx_L_start + N_NODES], dtype=float)
    s = np.sum(a)
    if s <= 0:
        s = 1.0
    L = a / s

    # shell area S_i (depends on D and t only)
    S = shell_area  # from above

    # d(ms)/da_k = rhos * draft * (S_k - (S dot L)) / s
    S_dot_L = float(np.sum(S * L))
    dms_da = (rhos * float(draft)) * (S - S_dot_L) / s

    # d(ms)/ddraft = rhos * (S dot L)
    dms_ddraft = rhos * S_dot_L

    # Ballast derivatives: compute per-section h and identify filled/partial sections
    # inner_area_section = (pi/4) * (D - 2t)^2 computed earlier
    inner_area = inner_area_section

    ballast_top = zbot + float(np.clip(ballast_height, 0.0, float(draft) * 0.999))
    z_edges = np.zeros(N_NODES + 1)
    z_edges[0] = zbot
    for i in range(N_NODES):
        z_edges[i+1] = z_edges[i] + L[i] * float(draft)

    # Precompute dL_i/da_k = (delta_ik * s - a_i) / s^2
    inv_s2 = 1.0 / (s * s)
    dL_dak = np.zeros((N_NODES, N_NODES), dtype=float)
    for i in range(N_NODES):
        for k in range(N_NODES):
            dL_dak[i, k] = (1.0 if i == k else 0.0) * s * inv_s2 - (a[i] * inv_s2)

    # dh_i/da_k and dh_i/ddraft
    dh_da = np.zeros((N_NODES, N_NODES), dtype=float)
    dh_ddraft = np.zeros(N_NODES, dtype=float)
    # cumulative L before section i
    cumL = np.concatenate(([0.0], np.cumsum(L)[:-1]))

    # find partial fill section index
    partial_idx = None
    for i in range(N_NODES):
        if z_edges[i+1] > ballast_top:
            partial_idx = i
            break

    for i in range(N_NODES):
        if partial_idx is None or i < partial_idx:
            # fully filled
            # h_i = L_i * draft
            for k in range(N_NODES):
                dh_da[i, k] = float(draft) * dL_dak[i, k]
            dh_ddraft[i] = L[i]
        elif i == partial_idx:
            # partial filled: h_i = ballast_height - draft * cumL[i]
            for k in range(N_NODES):
                # dh/da_k = -draft * sum_{j< i} dL_j/da_k
                dh_da[i, k] = -float(draft) * np.sum(dL_dak[:i, k])
            dh_ddraft[i] = -cumL[i]
        else:
            # empty section: derivatives zero
            pass

    # assemble ballast derivatives
    dballast_da = np.zeros(N_NODES, dtype=float)
    for k in range(N_NODES):
        # dballast/da_k = rho_concrete * sum_i inner_area_i * dh_i/da_k
        dballast_da[k] = float(RHO_CONCRETE) * float(np.sum(inner_area * dh_da[:, k]))

    dballast_ddraft = float(RHO_CONCRETE) * float(np.sum(inner_area * dh_ddraft))

    # place into gradient vector
    for k in range(N_NODES):
        grad[idx_L_start + k] = (dms_da[k] * COST_STEEL_PER_KG + dballast_da[k] * COST_CONCRETE_PER_KG) / 1e6

    idx_draft = 3 * N_NODES
    grad[idx_draft] = (dms_ddraft * COST_STEEL_PER_KG + dballast_ddraft * COST_CONCRETE_PER_KG) / 1e6

    # ballast height analytic derivative: sum(inner_area * indicator of partial section) * rho_concrete * (pi/4)
    # compute which section contains ballast_top (if any)
    dballast_dh = 0.0
    for i in range(N_NODES):
        z_low = z_edges[i]
        z_high = z_edges[i+1]
        # if ballast_top in this section (partial fill), derivative 1; else 0
        if (z_low < ballast_top) and (ballast_top < z_high):
            dballast_dh += (np.pi / 4.0) * (D[i] - 2.0 * t[i]) ** 2 * RHO_CONCRETE
            break
    # multiply by cost per kg and scaling
    grad[-1] = (dballast_dh * COST_CONCRETE_PER_KG) / 1e6

    return obj, grad


def hydrostatic_constraints_and_jac(x):
    """Compute hydrostatic constraints and (partial) analytic jacobian.
    Returns (c1, c2), J where c1 = (buoyancy_margin - FLOATING_MARGIN)/1e6,
    c2 = (gm - GM_MIN)/GM_MIN
    """
    x = np.asarray(x, dtype=float).ravel()
    D_nodes, th_nodes, a_raw, draft, ballast_height = _split_design(x)

    # normalize a_raw to obtain L fractions
    a = np.maximum(np.asarray(a_raw, dtype=float), 1e-12)
    s = np.sum(a)
    if s <= 0:
        s = 1.0
    L = a / s
    section_lengths = L * float(draft)

    # step profiles
    D = R2._step_profile(D_nodes, N_NODES)
    t = R2._step_profile(th_nodes, N_NODES)

    # areas
    outer_area = (np.pi / 4.0) * D ** 2
    inner_area = (np.pi / 4.0) * np.maximum((D - 2.0 * t) ** 2, 0.0)
    shell_area = outer_area - inner_area

    # displaced volume and buoyant mass
    V_disp = float(np.sum(outer_area * section_lengths))
    rhow = float(R2.SPAR0.get('rho_Water', None) or 1025.0)
    buoyant_mass = rhow * V_disp

    # structural steel mass
    rhos = float((R2.SPAR0.get('rho_Steel', None) or 7850.0))
    ms = float(np.sum(shell_area * section_lengths * rhos))

    # ballast mass
    zbot = -float(draft)
    ballast_top = zbot + float(np.clip(ballast_height, 0.0, float(draft) * 0.999))
    inner_diam = np.maximum(D - 2.0 * t, 0.0)
    inner_area_section = (np.pi / 4.0) * inner_diam ** 2
    ballast_mass = 0.0
    z_edges = np.zeros(N_NODES + 1)
    z_edges[0] = zbot
    for i in range(N_NODES):
        z_edges[i+1] = z_edges[i] + section_lengths[i]
        z_low = z_edges[i]
        z_high = z_edges[i+1]
        h = max(0.0, min(z_high, ballast_top) - z_low)
        ballast_mass += float(RHO_CONCRETE) * inner_area_section[i] * h

    mt = float(R2.SPAR0.get('M_Tower', 0.0) or 0.0)
    mtu = float(R2.SPAR0.get('M_Turbine', 0.0) or 0.0)
    tot_mass = ms + ballast_mass + mt + mtu

    buoyancy_margin = buoyant_mass - tot_mass

    # GM calculation (approx): IAA / V_disp + zCB - zCMtot
    # IAA uses waterline diameter (last section)
    D_waterline = float(D[-1])
    IAA = np.pi * D_waterline ** 4 / 64.0

    # compute centroids via sums
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    zCB = float(np.sum(outer_area * section_lengths * z_mid) / max(V_disp, 1e-12))
    zCMs = float(np.sum(shell_area * section_lengths * z_mid * rhos) / max(ms, 1e-12)) if ms > 0 else 0.0
    zballst = float(np.sum((inner_area_section * np.array([max(0.0, min(z_edges[i+1], ballast_top) - z_edges[i]) for i in range(N_NODES)]) * z_mid)) / max(ballast_mass, 1e-12)) if ballast_mass > 0 else zbot
    zCMf = (zCMs * ms + zballst * ballast_mass) / max((ms + ballast_mass), 1e-12)
    zCMt = float(R2.SPAR0.get('z_CM_Tower', 0.0) or 0.0)
    zturb = float(R2.SPAR0.get('z_CM_Turbine', 0.0) or 0.0)
    zCMtot = (zCMf * (ms + ballast_mass) + zCMt * mt + zturb * mtu) / max((ms + ballast_mass + mt + mtu), 1e-12)

    gm = IAA / max(V_disp, 1e-12) + zCB - zCMtot

    # Build constraint values matching R2 scaling
    c1 = (buoyancy_margin - R2.FLOATING_MARGIN) / 1e6
    c2 = (gm - R2.GM_MIN) / R2.GM_MIN

    # Jacobian: provide analytic parts for buoyancy w.r.t D, th, a, draft, ballast_height
    n = x.size
    J1 = np.zeros(n, dtype=float)
    J2 = np.zeros(n, dtype=float)

    # dV_disp/dD_i = (pi/2 * D_i) * section_lengths[i]
    dV_dD = (np.pi / 2.0) * D * section_lengths
    dV_dt = np.zeros_like(dV_dD)
    # dV/da_k = draft * sum_i A_i * dL_i/da_k
    inv_s2 = 1.0 / (s * s)
    dL_dak = np.zeros((N_NODES, N_NODES), dtype=float)
    for i in range(N_NODES):
        for k in range(N_NODES):
            dL_dak[i, k] = ((1.0 if i == k else 0.0) * s - a[i]) * inv_s2
    dV_da = np.zeros(N_NODES, dtype=float)
    for k in range(N_NODES):
        dV_da[k] = float(draft) * float(np.sum(outer_area * dL_dak[:, k]))
    dV_ddraft = float(np.sum(outer_area * L))

    # buoyant_mass derivative
    dbuoy_dD = rhow * dV_dD
    dbuoy_dt = rhow * dV_dt
    dbuoy_da = rhow * dV_da
    dbuoy_ddraft = rhow * dV_ddraft

    # ms derivatives (analytic): from shell_area * rhos * section_lengths
    dms_dD = (np.pi * rhos / 2.0) * t * section_lengths
    dms_dt = np.pi * rhos * (D / 2.0 - t) * section_lengths  # approximate
    dms_da = rhos * float(draft) * (shell_area - float(np.sum(shell_area * L))) / s
    dms_ddraft = rhos * float(np.sum(shell_area * L))

    # ballast derivatives: similar to objective routine
    # inner_area_section already computed
    # dh_i/da_k computed as in objective_and_jac logic
    # reuse same approach for dh/da
    cumL = np.concatenate(([0.0], np.cumsum(L)[:-1]))
    partial_idx = None
    for i in range(N_NODES):
        if z_edges[i+1] > ballast_top:
            partial_idx = i
            break
    dh_da = np.zeros((N_NODES, N_NODES), dtype=float)
    dh_ddraft = np.zeros(N_NODES, dtype=float)
    for i in range(N_NODES):
        if partial_idx is None or i < partial_idx:
            for k in range(N_NODES):
                dh_da[i, k] = float(draft) * dL_dak[i, k]
            dh_ddraft[i] = L[i]
        elif i == partial_idx:
            for k in range(N_NODES):
                dh_da[i, k] = -float(draft) * np.sum(dL_dak[:i, k])
            dh_ddraft[i] = -cumL[i]
    dballast_da = float(RHO_CONCRETE) * np.sum(inner_area[:, None] * dh_da, axis=0)
    dballast_ddraft = float(RHO_CONCRETE) * float(np.sum(inner_area * dh_ddraft))
    dballast_dD = np.zeros(N_NODES, dtype=float)

    # derivative of ballast mass w.r.t ballast height (partial section's inner area)
    dballast_dh = 0.0
    for i in range(N_NODES):
        z_low = z_edges[i]
        z_high = z_edges[i+1]
        if (z_low < ballast_top) and (ballast_top < z_high):
            dballast_dh = float(RHO_CONCRETE) * inner_area[i]
            break

    # assemble J1 (buoyancy_margin - tot_mass)
    # tot_mass derivatives: dtot = dms + dballast (mt, mtu const)
    idx_L_start = 2 * N_NODES
    for i in range(N_NODES):
        J1[i] = dbuoy_dD[i] - dms_dD[i] - dballast_dD[i]
        J1[N_NODES + i] = dbuoy_dt[i] - dms_dt[i]
        J1[idx_L_start + i] = dbuoy_da[i] - dms_da[i] - dballast_da[i]
    J1[3 * N_NODES] = dbuoy_ddraft - dms_ddraft - dballast_ddraft
    J1[-1] = -dballast_dh

    # Scale J1 as c1 = (buoyancy_margin - FLOATING_MARGIN)/1e6
    J1 = J1 / 1e6

    # For GM gradient, use finite differences for centroid terms and analytic for IAA/V parts
    eps = 1e-6
    J2 = np.zeros(n, dtype=float)
    base_gm = gm
    for k in range(n):
        xp = x.copy(); xm = x.copy()
        delta = max(eps, abs(x[k]) * eps)
        xp[k] += delta; xm[k] -= delta
        # recompute gm with small perturbation via reuse of this function (cheap)
        Dp, thp, ap, draftp, bhp = _split_design(xp)
        sp = R2.build_structure(float(draftp), Dp, thp, ap, float(bhp), R2.SPAR0)
        Vp = sp.V_disp
        IAAp = sp.IAA
        zCBp = sp.zCB
        zCMp = sp.zCM_Tot
        gmp = IAAp / max(Vp, 1e-12) + zCBp - zCMp
        J2[k] = (gmp - base_gm) / (2.0 * delta)

    # scale J2 to c2 = (gm - GM_MIN)/GM_MIN
    J2 = J2 / R2.GM_MIN

    return np.array([c1, c2]), np.vstack((J1, J2))


def run_stage_with_jac(x0, bounds, **stage_kwargs):
    """Adapted stage executor that supplies analytic jac to SLSQP."""
    # reuse environment precompute from R2.execute_optimization_stage
    return R2.execute_optimization_stage(
        x0=x0, bounds=bounds,
        time_final=stage_kwargs.get('time_final', 400.0),
        time_step=stage_kwargs.get('time_step', 0.5),
        transient_time=stage_kwargs.get('transient_time', 150.0),
        maxiter=stage_kwargs.get('maxiter', 30),
        ftol=stage_kwargs.get('ftol', 1e-4),
        eps=stage_kwargs.get('eps', 1e-6),
        stage_name=stage_kwargs.get('stage_name', 'CUSTOM_STAGE')
    )


def run(maxiter=40, ftol=1e-3):
    # build bounds and initial guess consistent with R2.run
    d_bounds = [(R2.MIN_REALISTIC_D, R2.UPPER_BOUND_D)] * N_NODES
    th_bounds = [(R2.MIN_THICKNESS, R2.MAX_THICKNESS)] * N_NODES
    frac_bounds = [(0.03, 0.35)] * N_NODES
    draft_bounds = [(R2.MIN_DRAFT, R2.MAX_DRAFT)]
    ballast_bounds = [(R2.MIN_BALLAST_HEIGHT, R2.MAX_BALLAST_HEIGHT)]
    bounds = d_bounds + th_bounds + frac_bounds + draft_bounds + ballast_bounds

    x0 = np.array([13.0] * N_NODES + [0.06] * N_NODES + [1.0 / N_NODES] * N_NODES + [110.0, 3.0], dtype=float)

    # Stage 1: try SLSQP with analytic jac for objective (constraints still FD inside R2)
    print('Stage 1: SLSQP with analytic objective gradient (constraints remain as in original)')

    # We wrap objective_and_jac to provide separate fun and jac to minimize via SLSQP.
    def fun(x):
        return objective_and_jac(x)[0]

    def jac(x):
        return objective_and_jac(x)[1]

    # Use cached evaluator for constraints from R2
    t_integration = np.arange(0.0, 400.0, 2.0 * 0.5)
    # Build waves and wind using the same steps as RUN_OPTIMIZER2
    waves = R2.Waves.from_mapping({
        'Hs': R2.SPAR0.get('significant_wave', 6.0),
        'Tp': R2.SPAR0.get('wave_period', 10.0),
        'TDur': 400.0,
        'dt': 1.0,
        'fHighCut': 0.5,
        'h': R2.TIME_INFO.get('water_depth', 320.0),
        'z': np.linspace(-R2.MAX_DRAFT, 0.0, 120),
        't': t_integration,
    }).calculate_jonswap_spectrum()
    waves = R2.generateRandomPhases(waves, seed=2)
    waves = waves.calculate_free_surface_elevation_time_series_fft()
    waves = waves.calculate_kinematics_fft()

    wind = R2.Wind.from_mapping({
        'V_10': R2.SPAR0.get('wind_speed', 8.0),
        'I': R2.TIME_INFO.get('turbulence_intensity', 0.05),
        'l': R2.TIME_INFO.get('turbulence_length_scale', 340.2),
        'TDur': 400.0,
        'dt': 1.0,
        'fHighCut': 0.5,
        't': t_integration,
    }).calculate_kaimal_spectrum()
    wind = R2.generateRandomPhases(wind, seed=1)
    wind = wind.calculate_time_series_fft()

    evaluate = R2._make_cached_evaluator(waves, wind, t_integration, 150.0, tqdm(desc='Stage1 Evals', unit='eval'))

    cons = []
    for idx in range(4):
        def make_fun(j):
            return lambda x: evaluate(x)['constraints'][j]
        cons.append({'type': 'ineq', 'fun': make_fun(idx)})

    def make_taper_fun(i):
        return lambda x: R2.MAX_ADJACENT_TAPER - abs(x[i+1] - x[i])

    for i in range(N_NODES - 1):
        cons.append({'type': 'ineq', 'fun': make_taper_fun(i)})

    # Add hydrostatic constraints with analytic jac (where available)
    cons.append({
        'type': 'ineq',
        'fun': lambda x: hydrostatic_constraints_and_jac(x)[0][0],
        'jac': lambda x: hydrostatic_constraints_and_jac(x)[1][0]
    })
    cons.append({
        'type': 'ineq',
        'fun': lambda x: hydrostatic_constraints_and_jac(x)[0][1],
        'jac': lambda x: hydrostatic_constraints_and_jac(x)[1][1]
    })

    res = minimize(fun, x0, method='SLSQP', jac=jac, bounds=bounds, constraints=cons,
                   options={'maxiter': int(maxiter), 'ftol': float(ftol), 'eps': 1e-6, 'disp': True})

    print('Stage 1 done; status:', res.message)

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the analytic-jac optimization experiment.')
    parser.add_argument('--maxiter', type=int, default=40, help='SLSQP iteration limit for Stage 1')
    parser.add_argument('--ftol', type=float, default=1e-3, help='SLSQP tolerance for Stage 1')
    args = parser.parse_args()
    run(maxiter=args.maxiter, ftol=args.ftol)
