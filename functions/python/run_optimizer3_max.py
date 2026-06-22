import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm
from types import SimpleNamespace

from common import loadFromJSON, generateRandomPhases
from waves import Waves
from wind import Wind
from integration import ode4
from floaterIntegration import dqdt as floater_dqdt
from models import Structure
from rotor import Rotor

# =========================================================================
# 1. FILE PATHS & SETUP
# =========================================================================
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

# =========================================================================
# 2. DESIGN LIMITS & OPTIMIZATION BOUNDS
# =========================================================================
SURGE_RMS_MAX = 3000 
PITCH_RMS_MAX_DEG = 10
GM_MIN = 0.5  
FLOATING_MARGIN = 1e4 
FIXED_THICKNESS = 0.08 
N_NODES = 10

# Natural period constraints
MIN_PITCH_PERIOD = 25.0
MIN_HEAVE_PERIOD = 20.0

# Optimization Bounds
UPPER_BOUND_D = 20.0
MIN_DRAFT = 50.0
MAX_DRAFT = 200.0
MIN_BALLAST_MASS = 0
MAX_BALLAST_MASS = 4e7     # 10,000 tonnes
MAX_ADJACENT_TAPER = 5  # max diameter change between adjacent nodes [m]

# =========================================================================
# 3. HIGH-FIDELITY VECTOR TRANSFORMS & MATH UTILITIES
# =========================================================================
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


def _natural_periods(structure):
    fnat = getattr(structure, 'fnat', None)
    pitch_period = float('nan')
    if fnat is not None and len(fnat) > 1 and np.isfinite(fnat[1]) and float(fnat[1]) > 0.0:
        pitch_period = 1.0 / float(fnat[1])

    heave_period = float('nan')
    if getattr(structure, 'M', None) is not None:
        waterplane_area = np.pi * (float(getattr(structure, 'DMonopile', 0.0)) / 2.0) ** 2
        c_heave = float(getattr(structure, 'rho_Water', 1025.0)) * 9.81 * waterplane_area
        added_heave = 0.5 * float(structure.buoyant_mass or 0.0)
        mass_heave = float(structure.MTot or 0.0) + added_heave
        if c_heave > 0.0 and mass_heave > 0.0:
            heave_period = 2.0 * np.pi * np.sqrt(mass_heave / c_heave)

    return pitch_period, heave_period


def _plot_diameter_profile(ax, structure, diameters, label, style='-'):
    z_edges = np.asarray(structure.z_edges_computed, dtype=float)
    ax.step(diameters, z_edges[:-1], style, where='post', label=label)
    ax.set_ylim([min(z_edges) - 5, 5]) 
    ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.5)


def _append_monotonic_taper_constraints(cons, start_idx, n_diameters):
    for i in range(start_idx, start_idx + n_diameters - 1):
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: x[idx] - x[idx + 1]})


def _get_weight_distribution(structure, D_nodes, L_fracs, ballast_mass, thickness=FIXED_THICKNESS):
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


# =========================================================================
# 4. STRUCTURE FACTORY & OBJECTIVES
# =========================================================================
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

    s.fnat = np.array([
        np.sqrt(max(s.C[0, 0], 1e-12) / max(s.A[0, 0] + s.M[0, 0], 1e-12)) / (2.0 * np.pi),
        np.sqrt(max(s.C[1, 1], 1e-12) / max(s.A[1, 1] + s.M[1, 1], 1e-12)) / (2.0 * np.pi),
    ], dtype=float)
    s.Theave = 2.0 * np.pi * np.sqrt(max(mtot, 1e-12) / max(rhow * g * outer_area[-1], 1e-12))
    
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
    return (s.ms + s.M_Ballast)/1e7


def constraints_fun(x, waves, wind, t_integration, transient_time, surge_rms_max=SURGE_RMS_MAX, pitch_rms_max_deg=PITCH_RMS_MAX_DEG):
    D_nodes, L_fracs, draft, ballast_mass = _split_design(x)
    D_nodes = np.maximum(D_nodes, 0.5)
    draft = max(draft, 10.0)
    
    s = build_structure(draft, D_nodes, L_fracs, ballast_mass, SPAR0)
    
    buoyancy_margin = s.buoyant_mass - s.MTot
    gm = (s.IAA / max(s.V_disp, 1e-9)) + s.zCB - s.zCM_Tot
    pitch_period, heave_period = _natural_periods(s)

    q0 = np.zeros(5)
    q = ode4(floater_dqdt, t_integration, q0, s, GLOBAL_ROTOR, waves, wind)

    post_transient = t_integration > transient_time
    if not np.any(post_transient):
        post_transient = np.arange(len(t_integration)) >= int(0.5 * len(t_integration))

    if np.any(np.isnan(q)) or np.any(np.isinf(q)):
        return np.array([-3.0, -3.0, (buoyancy_margin - FLOATING_MARGIN) / 1e6, (gm - GM_MIN) / GM_MIN, -1.0, -1.0])

    surge_rms = np.sqrt(np.mean(q[post_transient, 0] ** 2))
    pitch_rms = np.sqrt(np.mean((np.rad2deg(q[post_transient, 1])) ** 2))

    return np.array([
        (surge_rms_max - surge_rms) / surge_rms_max,
        (pitch_rms_max_deg - pitch_rms) / pitch_rms_max_deg,
        buoyancy_margin / 1e7,  
        (gm - GM_MIN) / GM_MIN,
        (pitch_period - MIN_PITCH_PERIOD) / MIN_PITCH_PERIOD if np.isfinite(pitch_period) else -1.0,
        (heave_period - MIN_HEAVE_PERIOD) / MIN_HEAVE_PERIOD if np.isfinite(heave_period) else -1.0,
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


# =========================================================================
# 5. ENVIRONMENT GENERATION & DISK CACHING
# =========================================================================
_ENV_CACHE_DIR = os.path.join(BASE, 'assignment5', 'python', 'outputVariables')
_ENV_DURATION  = 1200.0   
_ENV_DT        = 1.0      


def _env_cache_path(prefix, seed, **extra):
    os.makedirs(_ENV_CACHE_DIR, exist_ok=True)
    tag = f"{prefix}_T{_ENV_DURATION:.0f}_dt{_ENV_DT:.4f}_seed{seed}"
    for k, v in sorted(extra.items()):
        tag += f"_{k}{v:.4g}"
    return os.path.join(_ENV_CACHE_DIR, f"env_cache_{tag}.npz")


def _load_or_compute_waves():
    Hs   = SPAR0.get('significant_wave', 6.0)
    Tp   = SPAR0.get('wave_period', 10.0)
    h    = TIME_INFO.get('water_depth', 320.0)
    seed = 2
    t    = np.arange(0.0, _ENV_DURATION, _ENV_DT)
    cache_file = _env_cache_path('waves', seed, Hs=Hs, Tp=Tp, h=h)

    if os.path.exists(cache_file):
        d = np.load(cache_file)
        if 'eta' not in d.files:
            os.remove(cache_file)  
        else:
            return SimpleNamespace(t=d['t'], u=d['u'], ut=d['ut'], z=d['z'], eta=d['eta'])

    waves = Waves.from_mapping({
        'Hs': Hs, 'Tp': Tp, 'TDur': _ENV_DURATION,
        'dt': _ENV_DT, 'fHighCut': 0.5,
        'h': h, 'z': np.linspace(-MAX_DRAFT, 0.0, 120),
        't': t,
    }).calculate_jonswap_spectrum()
    waves = generateRandomPhases(waves, seed=seed)
    waves = waves.calculate_free_surface_elevation_time_series_fft()
    waves = waves.calculate_kinematics_fft()
    eta = getattr(waves, 'eta', np.zeros_like(waves.t))
    np.savez_compressed(cache_file, t=waves.t, u=waves.u, ut=waves.ut, z=waves.z, eta=eta)
    print(f"[ENV] Waves computed and saved to {os.path.basename(cache_file)}")
    return waves


def _load_or_compute_wind():
    V_10 = SPAR0.get('wind_speed', 8.0)
    I    = TIME_INFO.get('turbulence_intensity', 0.05)
    l    = TIME_INFO.get('turbulence_length_scale', 340.2)
    seed = 1
    t    = np.arange(0.0, _ENV_DURATION, _ENV_DT)
    cache_file = _env_cache_path('wind', seed, V10=V_10, I=I, l=l)

    if os.path.exists(cache_file):
        d = np.load(cache_file)
        return SimpleNamespace(t=d['t'], V_10=float(d['V_10']), V_hub=d['V_hub'])

    wind = Wind.from_mapping({
        'V_10': V_10, 'I': I, 'l': l,
        'TDur': _ENV_DURATION, 'dt': _ENV_DT,
        'fHighCut': 0.5, 't': t,
    }).calculate_kaimal_spectrum()
    wind = generateRandomPhases(wind, seed=seed)
    wind = wind.calculate_time_series_fft()
    np.savez_compressed(cache_file, t=wind.t, V_10=np.array(wind.V_10), V_hub=wind.V_hub)
    print(f"[ENV] Wind computed and saved to {os.path.basename(cache_file)}")
    return wind


def plot_environment(waves, wind):
    out_dir = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
    os.makedirs(out_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()

    eta = getattr(waves, 'eta', np.zeros_like(waves.t))
    ax1.plot(waves.t, eta,         color='steelblue',  linewidth=0.8, alpha=0.9, label='Wave elevation [m]')
    ax2.plot(wind.t,  wind.V_hub,  color='darkorange', linewidth=0.8, alpha=0.9, label='Hub wind speed [m/s]')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Wave elevation [m]', color='steelblue')
    ax2.set_ylabel('Hub wind speed [m/s]', color='darkorange')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax1.set_title('Environmental conditions Realisation')
    ax1.set_xlim(waves.t[0], waves.t[-1])
    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fpath = os.path.join(out_dir, 'environment_waves_wind_1200s_ro2.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
# 6. STAGE 0 FUNCTIONS (FAST HYDROSTATIC SEED FINDER)
# =========================================================================
def _split_simplified_design(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    D_top, D_mid, D_bot = x[0], x[1], x[2]
    draft = float(x[3])
    ballast_mass = float(x[4])
    
    D_nodes = np.interp(np.linspace(0.0, 1.0, N_NODES), [0.0, 0.5, 1.0], [D_bot, D_mid, D_top])
    L_fracs = np.full(N_NODES, 1.0 / N_NODES)
    return D_nodes, L_fracs, draft, ballast_mass


def hydrostatic_objective(x):
    D_nodes, L_fracs, draft, ballast_mass = _split_simplified_design(x)
    s = build_structure(draft, D_nodes, L_fracs, ballast_mass, SPAR0)
    return (s.ms + s.M_Ballast) / 1e7


def hydrostatic_constraints(x):
    D_nodes, L_fracs, draft, ballast_mass = _split_simplified_design(x)
    s = build_structure(draft, D_nodes, L_fracs, ballast_mass, SPAR0)
    
    buoyancy_margin = s.buoyant_mass - s.MTot
    gm = (s.IAA / max(s.V_disp, 1e-9)) + s.zCB - s.zCM_Tot
    pitch_period, heave_period = _natural_periods(s)
    
    return np.array([
        buoyancy_margin / 1e7,                                                    
        (gm - GM_MIN) / GM_MIN,                                                   
        (pitch_period - MIN_PITCH_PERIOD) / MIN_PITCH_PERIOD if np.isfinite(pitch_period) else -1.0, 
        (heave_period - MIN_HEAVE_PERIOD) / MIN_HEAVE_PERIOD if np.isfinite(heave_period) else -1.0  
    ])


# =========================================================================
# 7. MULTI-STAGE SOLVER RUNTIME ENGINE
# =========================================================================
def execute_optimization_stage(x0, bounds, waves, wind, time_final, time_step, transient_time, maxiter, ftol, stage_name):
    t_integration = np.arange(0.0, time_final, 2.0 * time_step)
    print(f"\n[{stage_name}] Starting (T={time_final:.0f}s, {maxiter} iter max, ftol={ftol})...")

    feval_progress = tqdm(desc=f'{stage_name} Evals', unit='eval')
    evaluate = _make_cached_evaluator(waves, wind, t_integration, transient_time, feval_progress)

    iteration_state = {'k': 0}
    progress = tqdm(total=maxiter, desc=f'{stage_name} Main Loop', unit='iter')

    cons = [
        {'type': 'ineq', 'fun': lambda x: evaluate(x)['constraints'][0]},
        {'type': 'ineq', 'fun': lambda x: evaluate(x)['constraints'][1]},
        {'type': 'ineq', 'fun': lambda x: evaluate(x)['constraints'][2]},
        {'type': 'ineq', 'fun': lambda x: evaluate(x)['constraints'][3]},
        {'type': 'ineq', 'fun': lambda x: evaluate(x)['constraints'][4]},
        {'type': 'ineq', 'fun': lambda x: evaluate(x)['constraints'][5]},
    ]
    
    cons.append({'type': 'eq', 'fun': lambda x: np.sum(x[N_NODES:2 * N_NODES]) - 1.0})
    _append_monotonic_taper_constraints(cons, 0, N_NODES)

    def callback(xk):
        iteration_state['k'] += 1
        res_data = evaluate(xk)
        progress.update(1)
        _, _, draft, bm = _split_design(xk)
        tqdm.write(
            f"[{stage_name} Iter {iteration_state['k']:02d}] "
            f"Draft={draft:.1f}m | Ballast={bm/1e3:.1f}t | "
            f"Mass={res_data['objective'] * 1e7 / 1e3:.1f} tonnes"
        )

    _ = evaluate(x0)['constraints']
    step_size = 1e-2 if "COARSE" in stage_name else 5e-3

    res = minimize(
        objective, x0, method='SLSQP', bounds=bounds, constraints=cons, callback=callback,
        options={'maxiter': maxiter, 'ftol': ftol, 'eps': step_size, 'disp': False}
    )

    c_final = evaluate(res.x)['constraints']
    progress.close()
    feval_progress.close()
    return res, evaluate(x0)['constraints'], c_final


# =========================================================================
# 8. POST-PROCESSING & GRAPH UTILITIES
# =========================================================================
def print_summary(res, x0, title, c0, c_final):
    D_f, _, dr_f, bm_f = _split_design(res.x)
    D_0, _, dr_0, bm_0 = _split_design(x0)

    sep = '=' * 74
    print(f"\n{sep}\n  {title}\n{sep}")
    print(f"\n  {'Parameter':<30} {'Initial':>16} {'Final':>16}")
    print(f"  {'-'*64}")
    print(f"  {'Draft [m]':<30} {dr_0:>16.3f} {dr_f:>16.3f}")
    print(f"  {'Ballast mass [tonnes]':<30} {bm_0/1e3:>16.1f} {bm_f/1e3:>16.1f}")
    print(f"  {'-'*64}")
    for idx, (d0, df) in enumerate(zip(D_0, D_f)):
        print(f"  {f'Node {idx+1} Diameter [m]':<30} {d0:>16.2f} {df:>16.2f}")
    surge_0, surge_f  = SURGE_RMS_MAX * (1.0 - c0[0]), SURGE_RMS_MAX * (1.0 - c_final[0])
    pitch_0, pitch_f  = PITCH_RMS_MAX_DEG * (1.0 - c0[1]), PITCH_RMS_MAX_DEG * (1.0 - c_final[1])
    buoy_0, buoy_f    = c0[2] * 1e7, c_final[2] * 1e7
    gm_0, gm_f        = c0[3] * GM_MIN + GM_MIN, c_final[3] * GM_MIN + GM_MIN
    pitchT_0, pitchT_f = MIN_PITCH_PERIOD * (1.0 + c0[4]), MIN_PITCH_PERIOD * (1.0 + c_final[4])
    heaveT_0, heaveT_f = MIN_HEAVE_PERIOD * (1.0 + c0[5]), MIN_HEAVE_PERIOD * (1.0 + c_final[5])

    CON_NAMES  = ["Surge RMS", "Pitch RMS", "Buoyancy Margin", "GM Height", "Pitch Period", "Heave Period"]
    CON_LIMITS = [f"<= {SURGE_RMS_MAX}m", f"<= {PITCH_RMS_MAX_DEG}°", "> 0 kg", f">= {GM_MIN}m", f">= {MIN_PITCH_PERIOD}s", f">= {MIN_HEAVE_PERIOD}s"]
    phys_init  = [surge_0, pitch_0, buoy_0, gm_0, pitchT_0, heaveT_0]
    phys_final = [surge_f, pitch_f, buoy_f, gm_f, pitchT_f, heaveT_f]

    print(f"\n  {'Constraint':<18} {'Limit':<22} {'Initial':>12} {'Final':>12} {'Status':>10}")
    print(f"  {'-'*78}")
    for name, limit, pv0, pvf, cvf in zip(CON_NAMES, CON_LIMITS, phys_init, phys_final, c_final):
        status = "VIOLATED" if cvf < -0.05 else ("ACTIVE" if abs(cvf) <= 0.05 else "Inactive")
        print(f"  {name:<18} {limit:<22} {pv0:>12.2f} {pvf:>12.2f} {status:>10}")
    print(sep)


def save_results_plot(res, x0, stage_name):
    D, L, dr, b = _split_design(res.x)
    D0, L0, dr0, b0 = _split_design(x0)
    s = build_structure(dr, D, L, b, SPAR0)
    s0 = build_structure(dr0, D0, L0, b0, SPAR0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_diameter_profile(axes[0], s0, D0, 'Initial', style='--')
    _plot_diameter_profile(axes[0], s, D, 'Final', style='-')
    axes[0].set_xlabel('Diameter [m]')
    axes[0].set_ylabel('Elevation z [m]')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    z, m = _get_weight_distribution(s, D, L, b)
    z0, m0 = _get_weight_distribution(s0, D0, L0, b0)
    axes[1].step(m0 / 1e3, z0, '--', where='post', label='Initial')
    axes[1].step(m  / 1e3, z,  '-',  where='post', label='Final')
    axes[1].set_xlabel('Cumulative mass [tonnes]')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    out_dir = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
    fig.savefig(os.path.join(out_dir, f'spar_{stage_name.lower()}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def _simulate_pitch_response(x_design, waves, wind, time_final=_ENV_DURATION, time_step=_ENV_DT):
    D_nodes, L_fracs, draft, ballast_mass = _split_design(x_design)
    structure = build_structure(draft, D_nodes, L_fracs, ballast_mass, SPAR0)

    t_response = np.arange(0.0, float(time_final), float(time_step))
    q0 = np.zeros(5)
    q = ode4(floater_dqdt, t_response, q0, structure, GLOBAL_ROTOR, waves, wind)
    pitch_deg = np.rad2deg(q[:, 1])
    return t_response, pitch_deg


def save_pitch_response_comparison(x_stage0, x_stage1, x_stage2, waves, wind):
    out_dir = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
    os.makedirs(out_dir, exist_ok=True)

    t0, p0 = _simulate_pitch_response(x_stage0, waves, wind)
    t1, p1 = _simulate_pitch_response(x_stage1, waves, wind)
    tf, pf = _simulate_pitch_response(x_stage2, waves, wind)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t0, p0, '--', linewidth=1.1, label='Stage 0 Hydrostatic')
    ax.plot(t1, p1, '-.', linewidth=1.1, label='Stage 1 Coarse')
    ax.plot(tf, pf, '-', linewidth=1.3, label='Stage 2 Fine')
    ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pitch response [deg]')
    ax.set_title('Pitch Response Comparison in Turbulent Wind and Irregular Waves')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fpath = os.path.join(out_dir, 'spar_pitch_response_all_stages.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return fpath


# =========================================================================
# 9. CENTRAL COORDINATOR RUN PIPELINE
# =========================================================================
def run():
    # Setup 22-variable vector bounds
    bounds = ([(4.0, UPPER_BOUND_D)] * N_NODES + 
              [(0.03, 0.35)] * N_NODES + 
              [(MIN_DRAFT, MAX_DRAFT)] + 
              [(MIN_BALLAST_MASS, MAX_BALLAST_MASS)])

    # =========================================================================
    # STAGE 0: FAST HYDROSTATIC SEED SWEAK
    # =========================================================================
    print("\n[STAGE 0] Executing Pure Hydrostatic Pre-Optimization Sweep...")
    x0_hydro = np.array([12.0, 12.0, 12.0, 100.0, 1.5e7])
    bounds_hydro = [(4.0, 25.0), (4.0, 25.0), (4.0, 25.0), (50.0, 150.0), (0.0, 2e7)]
    cons_hydro = [
        {'type': 'ineq', 'fun': lambda x: hydrostatic_constraints(x)[0]},
        {'type': 'ineq', 'fun': lambda x: hydrostatic_constraints(x)[1]},
        {'type': 'ineq', 'fun': lambda x: hydrostatic_constraints(x)[2]},
        {'type': 'ineq', 'fun': lambda x: hydrostatic_constraints(x)[3]},
    ]
    _append_monotonic_taper_constraints(cons_hydro, 0, 3)
    
    res_hydro = minimize(
        hydrostatic_objective, x0_hydro, method='SLSQP', bounds=bounds_hydro, 
        constraints=cons_hydro, options={'maxiter': 40, 'ftol': 1e-5, 'disp': False}
    )
    
    if res_hydro.success:
        print("[STAGE 0] Hydrostatic seed converged beautifully. Mapping to 22 variables...")
        D_hyd, L_hyd, dr_hyd, bm_hyd = _split_simplified_design(res_hydro.x)
        x0 = np.array([*D_hyd, *L_hyd, dr_hyd, bm_hyd], dtype=float)
    else:
        print("[WARNING] Stage 0 failed. Falling back to simple cylinders.")
        x0 = np.array([*[20.0]*N_NODES, *[0.1]*N_NODES, 100.0, 1.5e7], dtype=float)

    # Environmental setup
    waves = _load_or_compute_waves()
    wind  = _load_or_compute_wind()
    plot_environment(waves, wind)

    # =========================================================================
    # STAGE 1: TIME DOMAIN COARSE ENGINE
    # =========================================================================
    res_stage1, c0_s1, cf_s1 = execute_optimization_stage(
        x0=x0, bounds=bounds, waves=waves, wind=wind,
        time_final=400.0, time_step=0.5, transient_time=150.0,
        maxiter=100, ftol=1e-6, stage_name="STAGE_1_COARSE"
    )
    print_summary(res_stage1, x0, "STAGE 1 COARSE RESULTS", c0_s1, cf_s1)

    # =========================================================================
    # STAGE 2: HIGH-FIDELITY REFINEMENT POLISH
    # =========================================================================
    res_final, c0_s2, cf_s2 = execute_optimization_stage(
        x0=res_stage1.x, bounds=bounds, waves=waves, wind=wind,
        time_final=1200.0, time_step=0.5, transient_time=600.0,
        maxiter=200, ftol=1e-7, stage_name="STAGE_2_FINE"
    )
    print_summary(res_final, res_stage1.x, "STAGE 2 FINE RESULTS", c0_s2, cf_s2)

    # Save outputs and plots
    save_results_plot(res_stage1, x0, "stage1_coarse")
    save_results_plot(res_final, res_stage1.x, "stage2_fine")

    # --- Three-way comparison plot generation ---
    D0, L0, dr0, b0 = _split_design(x0)
    D1, L1, dr1, b1 = _split_design(res_stage1.x)
    Df, Lf, drf, bf = _split_design(res_final.x)

    s0 = build_structure(dr0, D0, L0, b0, SPAR0)
    s1 = build_structure(dr1, D1, L1, b1, SPAR0)
    sf = build_structure(drf, Df, Lf, bf, SPAR0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_diameter_profile(axes[0], s0, D0, f'Stage 0 Hydrostatic (Draft={dr0:.1f}m)', style='--')
    _plot_diameter_profile(axes[0], s1, D1, f'Stage 1 Coarse (Draft={dr1:.1f}m)', style='-.')
    _plot_diameter_profile(axes[0], sf, Df, f'Stage 2 Fine Polish (Draft={drf:.1f}m)', style='-')
    axes[0].set_xlabel('Diameter [m]')
    axes[0].set_ylabel('Elevation z [m]')
    axes[0].set_title('Diameter Taper Profile Progression')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    z0, m0 = _get_weight_distribution(s0, D0, L0, b0)
    z1, m1 = _get_weight_distribution(s1, D1, L1, b1)
    zf, mf_w = _get_weight_distribution(sf, Df, Lf, bf)
    axes[1].step(m0 / 1e3, z0, '--', where='post', label='Stage 0')
    axes[1].step(m1 / 1e3, z1, '-.', where='post', label='Stage 1 Coarse')
    axes[1].step(mf_w / 1e3, zf, '-', where='post', label='Stage 2 Fine')
    axes[1].set_xlabel('Cumulative Mass [tonnes]')
    axes[1].set_ylabel('Elevation z [m]')
    axes[1].set_title('Cumulative Structural Weight Distribution')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    out_dir = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
    fpath = os.path.join(out_dir, 'spar_comparison_all_stages.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close(fig)

    pitch_plot_path = save_pitch_response_comparison(x0, res_stage1.x, res_final.x, waves, wind)
    print(f"\n[COMPLETE] Multi-stage process finished. Plots exported to: {fpath}")
    print(f"[COMPLETE] Pitch response comparison exported to: {pitch_plot_path}")


if __name__ == '__main__':
    run()


