import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm

from common import loadFromJSON, generateRandomPhases
from waves import Waves
from wind import Wind
from integration import ode4
from floaterIntegration import dqdt as floater_dqdt
from models import Structure
from rotor import Rotor

# Input files
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_VARS = os.path.join(BASE, 'assignment5', 'python', 'inputVariables')

TIME_JSON = os.path.join(INPUT_VARS, 'time.json')
SPAR_JSON = os.path.join(INPUT_VARS, 'SparBuoyData.json')
ROTOR_JSON = os.path.join(INPUT_VARS, 'iea22mw.json')

# Load inputs
TIME_INFO = loadFromJSON(TIME_JSON)
SPAR0 = loadFromJSON(SPAR_JSON)
ROTOR0 = loadFromJSON(ROTOR_JSON)

# Compile Rotor ONCE outside the loop to eliminate initialization overhead
GLOBAL_ROTOR = Rotor.from_mapping(ROTOR0)
if not GLOBAL_ROTOR.ARotor:
    GLOBAL_ROTOR.ARotor = 0.25 * np.pi * (ROTOR0.get('DRotor', 0.0) ** 2)
GLOBAL_ROTOR.gamma = 0.0
GLOBAL_ROTOR.active = True

# Limits and model setup
SURGE_RMS_MAX = 30 
PITCH_RMS_MAX_DEG = 20
GM_MIN = 0.5  
FLOATING_MARGIN = 0.0  
FIXED_THICKNESS = 0.05  
N_NODES = 10

# Optimization Bounds
UPPER_BOUND_D = 25.0
MIN_DRAFT = 50.0
MAX_DRAFT = 150.0
MIN_BALLAST_MASS = 0.0
MAX_BALLAST_MASS = 2e7     # 20 000 tonnes
MAX_ADJACENT_TAPER = 2.5  # max diameter change between adjacent nodes [m]

def _split_design(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    D_nodes = x[:N_NODES]
    L_fracs = x[N_NODES:2 * N_NODES]
    draft = float(x[-2])
    ballast_mass = float(x[-1])
    return D_nodes, L_fracs, draft, ballast_mass


def _step_profile(values, n_points):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        return np.full(n_points, values.item(), dtype=float)
    x_nodes = np.linspace(0.0, 1.0, values.size)
    x_points = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_points, x_nodes, values)


def _design_key(x):
    return np.asarray(x, dtype=float).reshape(-1).tobytes()


def _plot_diameter_profile(ax, structure, diameters, label, style='-'):
    z_submerged = np.asarray(structure.z_edges_computed[:-1], dtype=float)
    d_submerged = np.asarray(diameters, dtype=float).reshape(-1)

    ax.step(d_submerged, z_submerged, style, where='post', label=label)
    ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.25)

def _get_weight_distribution(structure, D_nodes, L_fracs, ballast_mass, thickness=FIXED_THICKNESS):
    """Cumulative mass from the bottom upward: steel shell + ballast lump at the lowest node."""
    D_nodes = np.asarray(D_nodes, dtype=float).reshape(-1)
    L_fracs = np.maximum(np.asarray(L_fracs, dtype=float), 1e-3)
    L_fracs /= np.sum(L_fracs)
    draft = float(structure.draft)
    section_lengths = L_fracs * draft

    rhos = float(structure.rho_Steel or 7850.0)

    zbot = -draft
    z_edges = np.zeros(N_NODES + 1)
    z_edges[0] = zbot
    for i in range(N_NODES):
        z_edges[i + 1] = z_edges[i] + section_lengths[i]

    D_nodes = _step_profile(D_nodes, N_NODES)
    th_nodes = np.full(N_NODES, float(thickness), dtype=float)

    outer_area = np.pi * (D_nodes / 2.0) ** 2
    inner_area = np.pi * np.maximum((D_nodes - 2.0 * th_nodes) / 2.0, 0.0) ** 2
    shell_area = outer_area - inner_area
    section_steel_masses = shell_area * rhos * section_lengths

    mass_per_section = section_steel_masses.copy()
    mass_per_section[0] += max(float(ballast_mass), 0.0)

    return z_edges[:-1], np.cumsum(mass_per_section)
    
def build_structure(draft, D_nodes, L_fracs, ballast_mass, spar_template, thickness=FIXED_THICKNESS):
    s = Structure.from_mapping(spar_template)
    s.draft = float(draft)
    
    fb = float(s.fb or 0.0)
    ls = draft + fb
    s.ls = ls

    rhos = float(s.rho_Steel or 7850.0)
    rhow = float(s.rho_Water or 1025.0)
    g = 9.81

    L_fracs = np.maximum(np.asarray(L_fracs, dtype=float), 1e-3)
    L_fracs /= np.sum(L_fracs)  
    section_lengths = L_fracs * draft

    zbot = -draft
    z_edges = np.zeros(N_NODES + 1)
    z_edges[0] = zbot
    for i in range(N_NODES):
        z_edges[i+1] = z_edges[i] + section_lengths[i]
    
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

    D_nodes = _step_profile(D_nodes, N_NODES)
    th_nodes = np.full(N_NODES, float(thickness), dtype=float)

    outer_area = np.pi * (D_nodes / 2.0) ** 2
    inner_area = np.pi * np.maximum((D_nodes - 2.0 * th_nodes) / 2.0, 0.0) ** 2
    shell_area = outer_area - inner_area
    section_volumes = outer_area * section_lengths
    section_masses = shell_area * rhos * section_lengths

    ms = float(np.sum(section_masses))
    
    ballast_mass = float(max(ballast_mass, 0.0))
    s.M_Ballast = ballast_mass
    
    mf = ms + ballast_mass

    mt = float(s.M_Tower or 0.0)
    mtu = float(s.M_Turbine or 0.0)
    mtot = mf + mt + mtu

    buoyant_volume = float(np.sum(section_volumes))
    zCB = float(np.sum(section_volumes * z_mid) / buoyant_volume) if buoyant_volume > 0 else zbot / 2.0
    zCMs = float(np.sum(section_masses * z_mid) / ms) if ms > 0 else fb - ls / 2.0
    
    # Ballast treated as a lump mass at the bottom node
    zballst = zbot
    s.Ballast_COG = zballst
    
    zCMf = (zCMs * ms + zballst * ballast_mass) / mf if mf > 0 else 0.0
    zCMt = float(s.z_CM_Tower or 0.0)
    zturb = float(s.z_CM_Turbine or 0.0)
    zCMtot = (zCMf * mf + zCMt * mt + zturb * mtu) / mtot if mtot > 0 else 0.0

    ICM_sections = (1.0 / 12.0) * section_masses * (3.0 * (D_nodes / 2.0) ** 2 + section_lengths ** 2)
    I_Spar = float(np.sum(ICM_sections + section_masses * (z_mid ** 2)))
    ICMt = float(s.I_CM_Tower or 0.0)
    I_tower = ICMt + mt * (zCMt ** 2)
    I0_turbine = mtu * (zturb ** 2)
    I0_ballast = ballast_mass * (zballst ** 2)
    IOtot = I_Spar + I_tower + I0_turbine + I0_ballast

    D_eff = float(np.mean(D_nodes))
    D_waterline = float(D_nodes[-1])

    s.M = np.array([[mtot, mtot * zCMtot], [mtot * zCMtot, IOtot]], dtype=float)
    cm_eff = float(s.CM or 2.0) - 1.0
    A1 = (np.pi / 4.0) * rhow * D_eff ** 2 * cm_eff * draft
    A15 = -(np.pi / 4.0) * rhow * D_eff ** 2 * cm_eff * draft ** 2 / 2.0
    A5 = (np.pi / 4.0) * rhow * D_eff ** 2 * cm_eff * draft ** 3 / 3.0
    s.A = np.array([[A1, A15], [A15, A5]], dtype=float)
    s.MA_inv = np.linalg.inv(s.M + s.A)
    s.B = np.array([[float(s.B11 or 0.0), 0.0], [0.0, 0.0]], dtype=float)

    IAA = np.pi * D_waterline ** 4 / 64.0
    Cs5 = rhow * g * IAA + mtot * g * (zCB - zCMtot)
    Chst = np.array([[0.0, 0.0], [0.0, Cs5]], dtype=float)
    Kmoor = float(s.K_Moor or 0.0)
    zmoor = float(s.z_Moor or 0.0)
    Cmoor = np.array([[Kmoor, Kmoor * zmoor], [Kmoor * zmoor, Kmoor * zmoor ** 2]], dtype=float)
    s.C = Chst + Cmoor

    s.zCB = zCB
    s.V_disp = buoyant_volume
    s.buoyant_mass = rhow * buoyant_volume
    s.MTot = mtot
    s.zCM_Tot = zCMtot
    s.IO_Tot = IOtot
    s.ms = ms
    s.mf = mf
    s.IAA = IAA
    s.z_edges_computed = z_edges  
    
    s.DProfile = _step_profile(D_nodes, 100)
    s.ThicknessProfile = np.full(100, float(thickness), dtype=float)
    s.DMonopile = D_eff
    s.Thickness = float(thickness)
    s.z = np.linspace(zbot, 0.0, 120)  
    s.zBeamNodal = np.linspace(zbot, 0.0, 101)
    s.phiNodal = np.ones_like(s.zBeamNodal)

    return s

def objective(x):
    D_nodes, L_fracs, draft, ballast_mass = _split_design(x)
    s = build_structure(draft, D_nodes, L_fracs, ballast_mass, SPAR0)
    return s.ms + s.M_Ballast


def constraints_fun(x, waves, wind, t_integration, transient_time, surge_rms_max=SURGE_RMS_MAX, pitch_rms_max_deg=PITCH_RMS_MAX_DEG):
    D_nodes, L_fracs, draft, ballast_mass = _split_design(x)
    D_nodes = np.maximum(D_nodes, 0.5)
    draft = max(draft, 10.0)
    
    s = build_structure(draft, D_nodes, L_fracs, ballast_mass, SPAR0)
    
    buoyancy_margin = s.buoyant_mass - s.MTot
    gm = (s.IAA / max(s.V_disp, 1e-9)) + s.zCB - s.zCM_Tot

    # If the design is physically impossible, use a large penalty to push it away
    if buoyancy_margin < 0 or gm < GM_MIN:
        # Increase these multipliers so the solver prioritizes stability over mass
        buoy_penalty = min(buoyancy_margin, 0.0) / 1e3
        gm_penalty = min(gm - GM_MIN, 0.0) * 1e4 
        
        return np.array([
            -10.0, -10.0, # High penalty for instability so it stays away
            (buoyancy_margin - FLOATING_MARGIN) / 1e6, 
            (gm - GM_MIN) / GM_MIN
        ])

    # Integrated simulation pass using the pre-compiled global instance
    q0 = np.zeros(5)
    q = ode4(floater_dqdt, t_integration, q0, s, GLOBAL_ROTOR, waves, wind)

    post_transient = t_integration > transient_time
    if not np.any(post_transient):
        post_transient = np.arange(len(t_integration)) >= int(0.5 * len(t_integration))

    if np.any(np.isnan(q)) or np.any(np.isinf(q)):
        return np.array([-3.0, -3.0, (buoyancy_margin - FLOATING_MARGIN) / 1e6, (gm - GM_MIN) / GM_MIN])

    surge_rms = np.sqrt(np.mean(q[post_transient, 0] ** 2))
    pitch_rms = np.sqrt(np.mean((np.rad2deg(q[post_transient, 1])) ** 2))

    # Clamp to [-3, ...] so unstable-simulation outliers don't wreck gradient estimation
    return np.array([
        max(-3.0, (surge_rms_max - surge_rms) / surge_rms_max),
        max(-3.0, (pitch_rms_max_deg - pitch_rms) / pitch_rms_max_deg),
        (buoyancy_margin - FLOATING_MARGIN) / 1e6,
        (gm - GM_MIN) / GM_MIN
    ])


def _make_cached_evaluator(waves, wind, t_integration, transient_time, feval_progress=None):
    cache = {}
    def evaluate(x):
        key = _design_key(x)
        if key not in cache:
            if feval_progress is not None:
                feval_progress.update(1)
            cache[key] = {
                'objective': objective(x),
                'constraints': constraints_fun(x, waves, wind, t_integration, transient_time),
            }
        return cache[key]
    return evaluate


def execute_optimization_stage(x0, bounds, time_final, time_step, transient_time, maxiter, ftol, stage_name):
    """Executes a single optimization flight profile stage with targeted environments."""
    t_integration = np.arange(0.0, time_final, 2.0 * time_step)
    integration_dt = 2.0 * time_step
    
    print(f"\n[{stage_name}] Pre-computing environmental conditions (T={time_final:.0f}s, dt={integration_dt:.2f}s)...")
    waves = Waves.from_mapping({
        'Hs': SPAR0.get('significant_wave', 6.0),
        'Tp': SPAR0.get('wave_period', 10.0),
        'TDur': time_final,
        'dt': integration_dt,
        'fHighCut': 0.5,
        'h': TIME_INFO.get('water_depth', 320.0),
        'z': np.linspace(-MAX_DRAFT, 0.0, 120), 
        't': t_integration,
    }).calculate_jonswap_spectrum()
    waves = generateRandomPhases(waves, seed=2)
    waves = waves.calculate_free_surface_elevation_time_series_fft()
    waves = waves.calculate_kinematics_fft()

    wind = Wind.from_mapping({
        'V_10': SPAR0.get('wind_speed', 8.0),
        'I': TIME_INFO.get('turbulence_intensity', 0.05),
        'l': TIME_INFO.get('turbulence_length_scale', 340.2),
        'TDur': time_final,
        'dt': integration_dt,
        'fHighCut': 0.5,
        't': t_integration,
    }).calculate_kaimal_spectrum()
    wind = generateRandomPhases(wind, seed=1)
    wind = wind.calculate_time_series_fft()

    feval_progress = tqdm(desc=f'{stage_name} Evals', unit='eval')
    evaluate = _make_cached_evaluator(waves, wind, t_integration, transient_time, feval_progress)

    iteration_state = {'k': 0}
    progress = tqdm(total=maxiter, desc=f'{stage_name} Main Loop', unit='iter')

    cons = [
        {'type': 'ineq', 'fun': lambda x: evaluate(x)['constraints'][0]},
        {'type': 'ineq', 'fun': lambda x: evaluate(x)['constraints'][1]},
        {'type': 'ineq', 'fun': lambda x: evaluate(x)['constraints'][2]},
        {'type': 'ineq', 'fun': lambda x: evaluate(x)['constraints'][3]},
    ]
    for i in range(N_NODES - 1):
        # Two smooth directional constraints replace the non-differentiable abs():
        #   D[i+1] - D[i] <=  MAX_ADJACENT_TAPER
        #   D[i]   - D[i+1] <= MAX_ADJACENT_TAPER
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: MAX_ADJACENT_TAPER - (x[idx + 1] - x[idx])})
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: MAX_ADJACENT_TAPER + (x[idx + 1] - x[idx])})

    def callback(xk):
        iteration_state['k'] += 1
        res_data = evaluate(xk)
        progress.update(1)
        D_nodes, _, draft, bm = _split_design(xk)
        D_summary = ", ".join(f"{d:.1f}" for d in np.round(D_nodes, 1))
        tqdm.write(
            f"[{stage_name} Iter {iteration_state['k']:02d}] "
            f"Draft={draft:.1f}m | Ballast={bm/1e3:.1f}t | "
            f"D=[{D_summary}] m | Mass={res_data['objective']:.1f} kg"
        )

    # Evaluate at the initial point to prime the cache and record c0
    c0 = evaluate(x0)['constraints']

    res = minimize(
        objective, x0, method='SLSQP', bounds=bounds, constraints=cons, callback=callback,
        options={'maxiter': maxiter, 'ftol': ftol, 'eps': 1e-4, 'disp': False}
    )

    c_final = evaluate(res.x)['constraints']

    progress.close()
    feval_progress.close()
    return res, c0, c_final

def print_summary(res, x0, title, c0, c_final):
    D_f, L_f, dr_f, bm_f = _split_design(res.x)
    D_0, L_0, dr_0, bm_0 = _split_design(x0)

    s0 = build_structure(dr_0, D_0, L_0, bm_0, SPAR0)
    sf = build_structure(dr_f, D_f, L_f, bm_f, SPAR0)

    sep = '=' * 74
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    # --- Design Variables ---
    print(f"\n  {'Parameter':<30} {'Initial':>16} {'Final':>16}")
    print(f"  {'-'*64}")
    print(f"  {'Draft [m]':<30} {dr_0:>16.3f} {dr_f:>16.3f}")
    print(f"  {'Ballast mass [kg]':<30} {bm_0:>16.1f} {bm_f:>16.1f}")
    for i in range(N_NODES):
        print(f"  {'D_node_' + str(i+1) + ' [m]':<30} {D_0[i]:>16.4f} {D_f[i]:>16.4f}")
    for i in range(N_NODES):
        print(f"  {'L_frac_' + str(i+1):<30} {L_0[i]:>16.5f} {L_f[i]:>16.5f}")

    # --- Constraints: un-normalise back to physical units ---
    # Convention: c[k] > 0 -> satisfied, c[k] = 0 -> active, c[k] < 0 -> violated
    surge_0  = SURGE_RMS_MAX      * (1.0 - c0[0])
    surge_f  = SURGE_RMS_MAX      * (1.0 - c_final[0])
    pitch_0  = PITCH_RMS_MAX_DEG  * (1.0 - c0[1])
    pitch_f  = PITCH_RMS_MAX_DEG  * (1.0 - c_final[1])
    buoy_0   = c0[2]      * 1e6 + FLOATING_MARGIN
    buoy_f   = c_final[2] * 1e6 + FLOATING_MARGIN
    gm_0     = c0[3]      * GM_MIN + GM_MIN
    gm_f     = c_final[3] * GM_MIN + GM_MIN

    CON_NAMES  = ["Surge RMS", "Pitch RMS", "Buoyancy", "GM (stability)"]
    CON_UNITS  = ["m", "deg", "kg", "m"]
    CON_LIMITS = [
        f"<= {SURGE_RMS_MAX:.1f} m",
        f"<= {PITCH_RMS_MAX_DEG:.1f} deg",
        "> 0 kg",
        f"> {GM_MIN:.2f} m",
    ]
    phys_init  = [surge_0, pitch_0, buoy_0, gm_0]
    phys_final = [surge_f, pitch_f, buoy_f, gm_f]
    ACTIVE_TOL = 0.05

    print(f"\n  {'Constraint':<18} {'Limit':<22} {'Initial':>12} {'Final':>12} {'Status':>10}")
    print(f"  {'-'*78}")
    for name, limit, unit, pv0, pvf, cvf in zip(
            CON_NAMES, CON_LIMITS, CON_UNITS, phys_init, phys_final, c_final):
        if cvf < -ACTIVE_TOL:
            status = "VIOLATED"
        elif abs(cvf) <= ACTIVE_TOL:
            status = "ACTIVE"
        else:
            status = "Inactive"
        print(f"  {name:<18} {limit:<22} {pv0:>12.3f} {pvf:>12.3f} {status:>10}")

    # --- Lagrange Multipliers / KKT information ---
    print(f"\n  Lagrange Multipliers / KKT information:")
    print(f"  {'-'*52}")
    if hasattr(res, 'v') and res.v is not None:
        lm_vals = np.asarray(res.v).reshape(-1)
        lm_labels = CON_NAMES[:len(lm_vals)]
        for lbl, mu in zip(lm_labels, lm_vals):
            print(f"    {lbl:<28}: {mu:>14.6e}")
    elif hasattr(res, 'jac') and res.jac is not None:
        jac = np.asarray(res.jac).reshape(-1)
        var_labels = (
            [f"D_node_{i+1}" for i in range(N_NODES)] +
            [f"L_frac_{i+1}" for i in range(N_NODES)] +
            ["Draft", "BallastM"]
        )
        print(f"  (Objective gradient df/dx at solution — active-constraint rows are the KKT proxy)")
        for lbl, g in zip(var_labels, jac):
            print(f"    {lbl:<28}: {g:>14.6e}")
    else:
        print("    Not available from solver.")

    print(sep)

def save_results_plot(res, x0, stage_name):
    """Saves diameter and mass plots for a given optimization result."""
    D, L, dr, b = _split_design(res.x)
    D0, L0, dr0, b0 = _split_design(x0)
    s = build_structure(dr, D, L, b, SPAR0)
    s0 = build_structure(dr0, D0, L0, b0, SPAR0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Diameter Profile
    _plot_diameter_profile(axes[0], s0, D0, 'Initial', style='--')
    _plot_diameter_profile(axes[0], s, D, 'Final', style='-')
    axes[0].set_xlabel('Diameter [m]')
    axes[0].set_ylabel('Elevation z [m]')
    axes[0].set_title(f'Diameter profile: {stage_name}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Cumulative mass distribution
    z, m = _get_weight_distribution(s, D, L, b)
    z0, m0 = _get_weight_distribution(s0, D0, L0, b0)
    axes[1].step(m0 / 1e3, z0, '--', where='post', label='Initial')
    axes[1].step(m  / 1e3, z,  '-',  where='post', label='Final')
    axes[1].set_xlabel('Cumulative mass [tonnes]')
    axes[1].set_ylabel('Elevation z [m]')
    axes[1].set_title(f'Cumulative mass: {stage_name}')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    out_dir = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
    os.makedirs(out_dir, exist_ok=True)
    fname = f'spar_diameter_and_mass_{stage_name.lower()}_ro2.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)


def run():
    # --- BOUNDS DEFINITION ---
    d_bounds = [(4.0, UPPER_BOUND_D)] * N_NODES 
    frac_bounds = [(0.03, 0.35)] * N_NODES  
    draft_bounds = [(MIN_DRAFT, MAX_DRAFT)]
    ballast_mass_bounds = [(MIN_BALLAST_MASS, MAX_BALLAST_MASS)]
    bounds = d_bounds + frac_bounds + draft_bounds + ballast_mass_bounds

    # Initial stable anchor point
    initial_diameters = [20.0, 20.0, 20.0, 20.0, 20.0, 15.0, 15.0, 12.0, 10.0, 8.0]
    initial_fractions = [0.1] * N_NODES 
    initial_draft = 140.0        # Deeper draft increases stability
    initial_ballast_mass = 15e6  # 15,000 tonnes (Higher ballast)
    
    x0 = np.array([*initial_diameters, *initial_fractions, initial_draft, initial_ballast_mass], dtype=float)

    # =========================================================================
    # STAGE 1: COARSE & SWIFT DESIGN REGIME SWEEP (Saves ~75% Runtime)
    # =========================================================================
    res_stage1, c0_s1, cf_s1 = execute_optimization_stage(
        x0=x0, bounds=bounds,
        time_final=400.0,      # Shorter operational simulation
        time_step=0.5,
        transient_time=150.0,  # Proportional baseline transient window
        maxiter=35, 
        ftol=1e-2,             # Loose engineering tolerance for rough shaping
        stage_name="STAGE_1_COARSE"
    )
    print_summary(res_stage1, x0, "STAGE 1 RESULTS", c0_s1, cf_s1)

    # =========================================================================
    # STAGE 2: HIGH-FIDELITY STRUCTURAL POLISH
    # =========================================================================
    print("\n[STAGE 2] Starting high-fidelity refinement from Stage 1 solution...")
    res_final, c0_s2, cf_s2 = execute_optimization_stage(
        x0=res_stage1.x, bounds=bounds,  # Hot start from Stage 1 result
        time_final=1200.0,     # Full validation length
        time_step=0.5,
        transient_time=600.0,
        maxiter=15,            # Few steps needed since geometry is already close
        ftol=1e-3,             # Tight optimization tolerance
        stage_name="STAGE_2_FINE"
    )
    print_summary(res_final, res_stage1.x, "STAGE 2 RESULTS", c0_s2, cf_s2)

    sep = '=' * 74
    print(f"\n{sep}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"  {res_final.message}")
    print(sep)

    out_dir = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
    os.makedirs(out_dir, exist_ok=True)

    # --- Per-stage plots ---
    save_results_plot(res_stage1, x0,        "stage1_coarse")
    save_results_plot(res_final,  res_stage1.x, "stage2_fine")

    # --- Three-way comparison plot (Initial / Stage-1 / Stage-2) ---
    D_stage1, L_stage1, dr_stage1, b_stage1 = _split_design(res_stage1.x)
    D0, L0, dr0, b0 = _split_design(x0)
    Df, Lf, drf, bf = _split_design(res_final.x)

    s0 = build_structure(dr0,       D0,       L0,       b0,       SPAR0)
    s1 = build_structure(dr_stage1, D_stage1, L_stage1, b_stage1, SPAR0)
    sf = build_structure(drf,       Df,       Lf,       bf,       SPAR0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_d = axes[0]
    _plot_diameter_profile(ax_d, s0, D0,       f'Initial guess  (draft={dr0:.0f} m)',       style='--')
    _plot_diameter_profile(ax_d, s1, D_stage1, f'Stage 1 coarse (draft={dr_stage1:.0f} m)', style='-.')
    _plot_diameter_profile(ax_d, sf, Df,       f'Stage 2 fine   (draft={drf:.0f} m)',       style='-')
    ax_d.set_xlabel('Diameter [m]')
    ax_d.set_ylabel('Elevation z [m]')
    ax_d.set_title('Diameter profile — all stages')
    ax_d.set_ylim(float(s0.z_edges_computed[0]), 0.0)
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(fontsize=9)

    ax_w = axes[1]
    z0, m0 = _get_weight_distribution(s0, D0,       L0,       b0)
    z1, m1 = _get_weight_distribution(s1, D_stage1, L_stage1, b_stage1)
    zf, mf_w = _get_weight_distribution(sf, Df,     Lf,       bf)
    ax_w.step(m0   / 1e3, z0, '--', where='post', label='Initial guess',  linewidth=2)
    ax_w.step(m1   / 1e3, z1, '-.',  where='post', label='Stage 1 coarse', linewidth=2)
    ax_w.step(mf_w / 1e3, zf, '-',   where='post', label='Stage 2 fine',   linewidth=2)
    ax_w.axhline(0.0, color='k', linewidth=0.8, alpha=0.25)
    ax_w.set_xlabel('Cumulative mass [tonnes]')
    ax_w.set_ylabel('Elevation z [m]')
    ax_w.set_title('Cumulative mass distribution — steel + ballast')
    ax_w.set_ylim(float(s0.z_edges_computed[0]), 0.0)
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(fontsize=9)

    fig.suptitle('Spar buoy optimisation — multi-stage comparison', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'spar_multistage_diameter_and_mass_ro2.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figures saved to: {out_dir}")

    return res_final


if __name__ == '__main__':
    run()