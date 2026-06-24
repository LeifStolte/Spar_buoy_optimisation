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
PITCH_RMS_MAX_DEG = 10
GM_MIN = 0.5  
FLOATING_MARGIN = 0.0  
BUOYANCY_SCALE = 1e7
FIXED_THICKNESS = 0.05  
N_NODES = 10

# Fixed structural fractions and geometry constants
FIXED_L_FRACS = np.full(N_NODES, 1.0 / N_NODES, dtype=float)
FIXED_DRAFT = float(SPAR0['draft'])
FIXED_FREEBOARD = float(SPAR0['fb'])
FIXED_BALLAST_MASS = float(SPAR0['M_Ballast'])

# Optimization Bounds
UPPER_BOUND_D = 25.0
MAX_ADJACENT_TAPER = 4  # max diameter change between adjacent nodes [m]


def _split_design(x):
    """
    Splits the modified 10-element design vector into physical parameters.
    Injects the fixed static length fractions, draft, and ballast mass.
    """
    D_nodes = np.asarray(x, dtype=float).reshape(-1)
    return D_nodes, FIXED_L_FRACS, FIXED_DRAFT, FIXED_BALLAST_MASS


def _step_profile(values, n_points):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        return np.full(n_points, values.item(), dtype=float)
    x_nodes = np.linspace(0.0, 1.0, values.size)
    x_points = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_points, x_nodes, values)


def _design_key(x):
    return np.asarray(x, dtype=float).reshape(-1).tobytes()


def _bounds_arrays(bounds):
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    scale = ub - lb
    if np.any(scale <= 0.0):
        raise ValueError("All optimization bounds must have positive width.")
    return lb, ub, scale


def _localized_bounds(bounds, x_center, move_limits=None):
    if move_limits is None:
        return bounds

    x_center = np.asarray(x_center, dtype=float)
    move_limits = np.asarray(move_limits, dtype=float)
    localized = []
    for (lo, hi), xc, mv in zip(bounds, x_center, move_limits):
        localized.append((max(lo, xc - mv), min(hi, xc + mv)))
    return localized


def _to_scaled(x, lb, scale):
    return (np.asarray(x, dtype=float) - lb) / scale


def _from_scaled(y, lb, scale):
    return lb + np.asarray(y, dtype=float) * scale


def _taper_margins(D_nodes):
    D_nodes = np.asarray(D_nodes, dtype=float).reshape(-1)
    return MAX_ADJACENT_TAPER - np.abs(np.diff(D_nodes))


def _dynamic_response_is_penalty(c_vec):
    c_vec = np.asarray(c_vec, dtype=float).reshape(-1)
    return (
        np.isclose(c_vec[0], -10.0) or np.isclose(c_vec[1], -10.0) or
        np.isclose(c_vec[0], -3.0) or np.isclose(c_vec[1], -3.0)
    )


def _dynamic_response_text(c_val, limit):
    if np.isclose(c_val, -10.0) or np.isclose(c_val, -3.0):
        return "not simulated"
    return f"{limit * (1.0 - c_val):.3f}"


def _static_margins(x):
    D_nodes, L_fracs, draft, ballast_mass = _split_design(x)
    s = build_structure(draft, D_nodes, L_fracs, ballast_mass, SPAR0)
    buoyancy_margin = s.buoyant_mass - s.MTot
    gm = (s.IAA / max(s.V_disp, 1e-9)) + s.zCB - s.zCM_Tot
    min_taper_margin = float(np.min(_taper_margins(D_nodes)))
    return buoyancy_margin, gm, min_taper_margin


def _plot_diameter_profile(ax, structure, diameters, label, style='-'):
    z_edges = np.asarray(structure.z_edges_computed, dtype=float)
    # Append the final diameter to match the full length of z_edges to show the top profile row
    extended_diameters = np.append(diameters, diameters[-1])
    ax.step(extended_diameters, z_edges, style, where='post', label=label)
    ax.set_ylim([min(z_edges) - 5, FIXED_FREEBOARD + 5]) 
    ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.5)


def _get_weight_distribution(structure, D_nodes, L_fracs, ballast_mass, thickness=FIXED_THICKNESS):
    D_nodes = np.asarray(D_nodes, dtype=float).reshape(-1)
    L_fracs = np.maximum(np.asarray(L_fracs, dtype=float), 1e-3)
    L_fracs /= np.sum(L_fracs)
    
    total_length = float(structure.draft) + FIXED_FREEBOARD
    section_lengths = L_fracs * total_length

    rhos = float(structure.rho_Steel or 7850.0)

    zbot = -float(structure.draft)
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

    # Calculate horizontal plate masses at adjacent diameter changes
    plate_masses = np.zeros(N_NODES)
    for i in range(N_NODES - 1):
        plate_area = 0.25 * np.pi * np.abs(D_nodes[i+1]**2 - D_nodes[i]**2)
        # Assign half the mass to the lower section and half to the upper section for visual staging
        m_plate = plate_area * thickness * rhos
        plate_masses[i] += 0.5 * m_plate
        plate_masses[i+1] += 0.5 * m_plate

    mass_per_section = section_steel_masses + plate_masses
    mass_per_section[0] += max(float(ballast_mass), 0.0)

    return z_edges, np.append(np.cumsum(mass_per_section), np.cumsum(mass_per_section)[-1])


def build_structure(draft, D_nodes, L_fracs, ballast_mass, spar_template, thickness=FIXED_THICKNESS):
    s = Structure.from_mapping(spar_template)
    s.draft = float(draft)
    s.fb = FIXED_FREEBOARD
    
    total_length = draft + FIXED_FREEBOARD
    s.ls = total_length

    rhos = float(s.rho_Steel or 7850.0)
    rhow = float(s.rho_Water or 1025.0)
    g = 9.81

    L_fracs = np.maximum(np.asarray(L_fracs, dtype=float), 1e-3)
    L_fracs /= np.sum(L_fracs)  
    section_lengths = L_fracs * total_length

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
    
    submerged_lengths = np.clip(z_edges[1:], -draft, 0.0) - np.clip(z_edges[:-1], -draft, 0.0)
    section_volumes = outer_area * submerged_lengths
    section_masses = shell_area * rhos * section_lengths

    # --- Compute horizontal connecting plate properties ---
    m_plates_tot = 0.0
    m_plates_z = 0.0
    I_plates_tot = 0.0
    
    for i in range(N_NODES - 1):
        z_interface = z_edges[i+1]
        plate_area = 0.25 * np.pi * np.abs(D_nodes[i+1]**2 - D_nodes[i]**2)
        m_plate = plate_area * thickness * rhos
        
        m_plates_tot += m_plate
        m_plates_z += m_plate * z_interface
        
        # In-plane transverse moment of inertia of flat ring about its own centroidal axis
        r1, r2 = D_nodes[i]/2.0, D_nodes[i+1]/2.0
        I_cm_plate = 0.25 * m_plate * (r1**2 + r2**2)
        # Parallel axis theorem to find inertia about z=0
        I_plates_tot += I_cm_plate + m_plate * (z_interface ** 2)
    # -----------------------------------------------------

    ms = float(np.sum(section_masses)) + m_plates_tot
    ballast_mass = float(max(ballast_mass, 0.0))
    s.M_Ballast = ballast_mass
    mf = ms + ballast_mass

    mt = float(s.M_Tower or 0.0)
    mtu = float(s.M_Turbine or 0.0)
    mtot = mf + mt + mtu

    buoyant_volume = float(np.sum(section_volumes))
    zCB = float(np.sum(section_volumes * np.clip(z_mid, -draft, 0.0)) / buoyant_volume) if buoyant_volume > 0 else zbot / 2.0
    
    # Update zCMs to include horizontal plates
    zCMs = float((np.sum(section_masses * z_mid) + m_plates_z) / ms) if ms > 0 else FIXED_FREEBOARD - total_length / 2.0
    
    zballst = zbot
    s.Ballast_COG = zballst
    
    zCMf = (zCMs * ms + zballst * ballast_mass) / mf if mf > 0 else 0.0
    zCMt = float(s.z_CM_Tower or 0.0)
    zturb = float(s.z_CM_Turbine or 0.0)
    zCMtot = (zCMf * mf + zCMt * mt + zturb * mtu) / mtot if mtot > 0 else 0.0

    ICM_sections = (1.0 / 12.0) * section_masses * (3.0 * (D_nodes / 2.0) ** 2 + section_lengths ** 2)
    I_Spar = float(np.sum(ICM_sections + section_masses * (z_mid ** 2))) + I_plates_tot
    ICMt = float(s.I_CM_Tower or 0.0)
    I_tower = ICMt + mt * (zCMt ** 2)
    I0_turbine = mtu * (zturb ** 2)
    I0_ballast = ballast_mass * (zballst ** 2)
    IOtot = I_Spar + I_tower + I0_turbine + I0_ballast

    D_eff = float(np.mean(D_nodes))
    wl_idx = np.searchsorted(z_edges, 0.0) - 1
    wl_idx = np.clip(wl_idx, 0, N_NODES - 1)
    D_waterline = float(D_nodes[wl_idx])

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
    s.z = np.linspace(zbot, FIXED_FREEBOARD, 120)  
    s.zBeamNodal = np.linspace(zbot, FIXED_FREEBOARD, 101)
    s.phiNodal = np.ones_like(s.zBeamNodal)

    return s


def objective(x):
    D_nodes, L_fracs, draft, ballast_mass = _split_design(x)
    s = build_structure(draft, D_nodes, L_fracs, ballast_mass, SPAR0)
    return s.ms + s.M_Ballast


def constraints_fun(x, waves, wind, t_integration, transient_time, surge_rms_max=SURGE_RMS_MAX, pitch_rms_max_deg=PITCH_RMS_MAX_DEG):
    D_nodes, L_fracs, draft, ballast_mass = _split_design(x)
    D_nodes = np.maximum(D_nodes, 0.5)
    
    s = build_structure(draft, D_nodes, L_fracs, ballast_mass, SPAR0)
    
    buoyancy_margin = s.buoyant_mass - s.MTot
    gm = (s.IAA / max(s.V_disp, 1e-9)) + s.zCB - s.zCM_Tot

    if buoyancy_margin < 0 or gm < GM_MIN:
        return np.array([
            -10.0,
            -10.0,
            (buoyancy_margin - FLOATING_MARGIN) / BUOYANCY_SCALE,
            (gm - GM_MIN) / GM_MIN
        ])

    q0 = np.zeros(5)
    q = ode4(floater_dqdt, t_integration, q0, s, GLOBAL_ROTOR, waves, wind)

    post_transient = t_integration > transient_time
    if not np.any(post_transient):
        post_transient = np.arange(len(t_integration)) >= int(0.5 * len(t_integration))

    if np.any(np.isnan(q)) or np.any(np.isinf(q)):
        return np.array([
            -3.0,
            -3.0,
            (buoyancy_margin - FLOATING_MARGIN) / BUOYANCY_SCALE,
            (gm - GM_MIN) / GM_MIN
        ])

    surge_rms = np.sqrt(np.mean(q[post_transient, 0] ** 2))
    pitch_rms = np.sqrt(np.mean((np.rad2deg(q[post_transient, 1])) ** 2))

    return np.array([
        (surge_rms_max - surge_rms) / surge_rms_max,
        (pitch_rms_max_deg - pitch_rms) / pitch_rms_max_deg,
        (buoyancy_margin - FLOATING_MARGIN) / BUOYANCY_SCALE,
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
        'h': h, 'z': np.linspace(-FIXED_DRAFT * 1.2, 0.0, 120),
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
    ax1.set_title(
        f'Environmental conditions — {_ENV_DURATION:.0f} s realisation '
        f'(JONSWAP Hs={SPAR0.get("significant_wave", 6.0):.1f} m, '
        f'Kaimal V₁₀={SPAR0.get("wind_speed", 8.0):.1f} m/s)'
    )
    ax1.set_xlim(waves.t[0], waves.t[-1])
    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fpath = os.path.join(out_dir, 'environment_waves_wind_1200s_ro2.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ENV] Environment plot saved to {os.path.basename(fpath)}")


def execute_optimization_stage(x0, bounds, waves, wind, time_final, time_step, transient_time, maxiter, ftol, stage_name, move_limits=None):
    t_integration = np.arange(0.0, time_final, 2.0 * time_step)
    print(f"\n[{stage_name}] Starting (T={time_final:.0f}s, {maxiter} iter max, ftol={ftol})...")

    stage_bounds = _localized_bounds(bounds, x0, move_limits)
    lb, ub, x_scale = _bounds_arrays(stage_bounds)
    y0 = np.clip(_to_scaled(x0, lb, x_scale), 0.0, 1.0)
    y_bounds = [(0.0, 1.0)] * len(y0)
    objective_scale = 1e6
    if move_limits is not None:
        print(f"[{stage_name}] Using local move limits for scaled SLSQP variables.")

    feval_progress = tqdm(desc=f'{stage_name} Evals', unit='eval')
    evaluate = _make_cached_evaluator(waves, wind, t_integration, transient_time, feval_progress)

    def y_to_x(y):
        return _from_scaled(y, lb, x_scale)

    def objective_y(y):
        return objective(y_to_x(y)) / objective_scale

    def constraints_y(y):
        return evaluate(y_to_x(y))['constraints']

    def first_order_optimality(y):
        y = np.asarray(y, dtype=float)
        grad = np.zeros_like(y)
        h = 1e-6
        for j in range(len(y)):
            yp = y.copy()
            ym = y.copy()
            yp[j] = min(1.0, yp[j] + h)
            ym[j] = max(0.0, ym[j] - h)
            denom = yp[j] - ym[j]
            if denom > 0.0:
                grad[j] = (objective_y(yp) - objective_y(ym)) / denom
        return float(np.linalg.norm(grad, ord=np.inf))

    iteration_state = {
        'k': 0,
        'x_prev': np.copy(x0),
        'y_prev': np.copy(y0)
    }
    
    header = (
        f"\n{'Iter':>5}  {'Func-count':>10}  {'Fval':>14}  "
        f"{'Feasibility':>12}  {'Step Length':>12}  "
        f"{'Norm of step':>14}  {'First-order':>12}"
    )
    tqdm.write(header)
    tqdm.write("-" * len(header))

    cons = [
        {'type': 'ineq', 'fun': lambda y: constraints_y(y)[0]},
        {'type': 'ineq', 'fun': lambda y: constraints_y(y)[1]},
        {'type': 'ineq', 'fun': lambda y: constraints_y(y)[2]},
        {'type': 'ineq', 'fun': lambda y: constraints_y(y)[3]},
    ]
    for i in range(N_NODES - 1):
        cons.append({'type': 'ineq', 'fun': lambda y, idx=i: MAX_ADJACENT_TAPER - (y_to_x(y)[idx + 1] - y_to_x(y)[idx])})
        cons.append({'type': 'ineq', 'fun': lambda y, idx=i: MAX_ADJACENT_TAPER + (y_to_x(y)[idx + 1] - y_to_x(y)[idx])})

    def callback(yk):
        xk = y_to_x(yk)
        res_data = evaluate(xk)
        
        c_vals = res_data['constraints']
        taper_vals = _taper_margins(xk)
        violations = [max(0.0, -c) for c in c_vals]
        violations.extend(max(0.0, -c) for c in taper_vals)
        feasibility = max(violations) if violations else 0.0
        
        step_length = np.linalg.norm(yk - iteration_state['y_prev'])
        step_norm = np.linalg.norm(xk - iteration_state['x_prev'])
        first_order = first_order_optimality(yk)
        
        tqdm.write(
            f"{iteration_state['k']:5d}  "
            f"{feval_progress.n:10d}  "
            f"{res_data['objective']:14.6e}  "
            f"{feasibility:12.3e}  "
            f"{step_length:12.3e}  "
            f"{step_norm:14.3e}  "
            f"{first_order:12.3e}"
        )
        
        iteration_state['k'] += 1
        iteration_state['x_prev'] = np.copy(xk)
        iteration_state['y_prev'] = np.copy(yk)

    c0 = evaluate(x0)['constraints']
    
    c0_violations = [max(0.0, -c) for c in c0]
    c0_violations.extend(max(0.0, -c) for c in _taper_margins(x0))
    tqdm.write(
        f"{0:5d}  "
        f"{feval_progress.n:10d}  "
        f"{evaluate(x0)['objective']:14.6e}  "
        f"{max(c0_violations):12.3e}  "
        f"{0.000e+00:12.3e}  "
        f"{0.000e+00:14.3e}  "
        f"{first_order_optimality(y0):12.3e}"
    )
    iteration_state['k'] = 1

    res = minimize(
        objective_y, y0, method='SLSQP', bounds=y_bounds, constraints=cons, callback=callback,
        options={'maxiter': maxiter, 'ftol': ftol, 'eps': 1e-4, 'disp': False}
    )

    res.x_scaled = np.copy(res.x)
    res.x = y_to_x(res.x_scaled)
    res.fun = objective(res.x)
    res.jac = None

    c_final = evaluate(res.x)['constraints']
    feval_progress.close()
    
    return res, c0, c_final


def print_summary(res, x0, title, c0, c_final):
    D_f, L_f, dr_f, bm_f = _split_design(res.x)
    D_0, L_0, dr_0, bm_0 = _split_design(x0)

    sep = '=' * 74
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    print(f"\n  {'Parameter':<30} {'Initial':>16} {'Final':>16}")
    print(f"  {'-'*64}")
    print(f"  {'Draft [m] (FIXED)':<30} {dr_0:>16.3f} {dr_f:>16.3f}")
    print(f"  {'Ballast mass [kg] (FIXED)':<30} {bm_0:>16.1f} {bm_f:>16.1f}")
    for i in range(N_NODES):
        print(f"  {'D_node_' + str(i+1) + ' [m]':<30} {D_0[i]:>16.4f} {D_f[i]:>16.4f}")

    surge_0  = _dynamic_response_text(c0[0], SURGE_RMS_MAX)
    surge_f  = _dynamic_response_text(c_final[0], SURGE_RMS_MAX)
    pitch_0  = _dynamic_response_text(c0[1], PITCH_RMS_MAX_DEG)
    pitch_f  = _dynamic_response_text(c_final[1], PITCH_RMS_MAX_DEG)
    buoy_0   = f"{c0[2]      * BUOYANCY_SCALE + FLOATING_MARGIN:.3f}"
    buoy_f   = f"{c_final[2] * BUOYANCY_SCALE + FLOATING_MARGIN:.3f}"
    gm_0     = c0[3]      * GM_MIN + GM_MIN
    gm_f     = c_final[3] * GM_MIN + GM_MIN

    CON_NAMES  = ["Surge RMS", "Pitch RMS", "Buoyancy", "GM (stability)"]
    CON_LIMITS = [
        f"<= {SURGE_RMS_MAX:.1f} m",
        f"<= {PITCH_RMS_MAX_DEG:.1f} deg",
        " > 0 kg",
        f"> {GM_MIN:.2f} m",
    ]
    phys_init  = [surge_0, pitch_0, buoy_0, f"{gm_0:.3f}"]
    phys_final = [surge_f, pitch_f, buoy_f, f"{gm_f:.3f}"]
    ACTIVE_TOL = 0.05

    print(f"\n  {'Constraint':<18} {'Limit':<22} {'Initial':>12} {'Final':>12} {'Status':>10}")
    print(f"  {'-'*78}")
    for name, limit, pv0, pvf, cvf in zip(CON_NAMES, CON_LIMITS, phys_init, phys_final, c_final):
        if cvf < -ACTIVE_TOL:
            status = "VIOLATED"
        elif abs(cvf) <= ACTIVE_TOL:
            status = "ACTIVE"
        else:
            status = "Inactive"
        print(f"  {name:<18} {limit:<22} {pv0:>12} {pvf:>12} {status:>10}")

    taper_0 = _taper_margins(D_0)
    taper_f = _taper_margins(D_f)
    taper_status = "VIOLATED" if np.any(taper_f < -ACTIVE_TOL) else ("ACTIVE" if np.any(np.abs(taper_f) <= ACTIVE_TOL) else "Inactive")
    print(f"\n  {'Taper constraint':<18} {'Limit':<22} {'Initial':>12} {'Final':>12} {'Status':>10}")
    print(f"  {'-'*78}")
    print(
        f"  {'Adjacent D step':<18} "
        f"{'<= ' + format(MAX_ADJACENT_TAPER, '.2f') + ' m':<22} "
        f"{np.min(taper_0):>12.3f} "
        f"{np.min(taper_f):>12.3f} "
        f"{taper_status:>10}"
    )

    print(f"\n  Lagrange Multipliers / KKT information:")
    print(f"  {'-'*52}")
    if hasattr(res, 'jac') and res.jac is not None:
        jac = np.asarray(res.jac).reshape(-1)
        var_labels = [f"D_node_{i+1}" for i in range(N_NODES)]
        print(f"  (Objective gradient df/dx at solution - active-constraint rows are the KKT proxy)")
        for lbl, g in zip(var_labels, jac[:len(var_labels)]):
            print(f"      {lbl:<28}: {g:>14.6e}")
    else:
        print("    Not available from solver.")

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
    axes[0].set_title(f'Diameter profile: {stage_name}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

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


def generate_valid_raster_guesses(num_designs=5):
    """
    Generates a set of distinct initial guesses that are structurally sound,
    respecting bounds, taper limits, buoyancy, and GM stability requirements.
    """
    valid_guesses = []
    
    # Design 1: Uniform slender column (Our golden baseline anchor)
    safe_anchor = np.full(N_NODES, 20.0, dtype=float)
    valid_guesses.append(safe_anchor)
    
    # Design 2: Hourglass shape (narrow middle, wide base/top)
    valid_guesses.append(np.array([25.0, 21.0, 17.0, 13.0, 9.0, 9.0, 13.0, 17.0, 21.0, 25.0]))
    
    # Design 3: Heavy tapered pyramid (fat bottom, narrow top)
    valid_guesses.append(np.array([24.0, 21.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0]))
    
    # Design 4: Inverse pyramid (narrow bottom, fat top)
    valid_guesses.append(np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 21.0, 24.0]))

    # Design 5: The original linear stepping guess
    valid_guesses.append(np.array([5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 25.0, 25.0, 25.0, 25.0]))

    verified_guesses = []
    
    # Maximum physical boundary constraint check
    upper_bound = 25.0  # Match your UPPER_BOUND_D variable
    lower_bound = 4.0
    
    for idx, x in enumerate(valid_guesses[:num_designs]):
        if idx == 0:
            verified_guesses.append(x)
            continue
            
        alpha = 0.0
        max_attempts = 20
        attempt = 0
        
        # Make a working copy of the candidate shape
        x_candidate = np.copy(x)
        
        while attempt < max_attempts:
            # Enforce physical dimensional clipping bounds explicitly
            x_candidate = np.clip(x_candidate, lower_bound, upper_bound)
            
            # Check adjacent node step limitations (|D_i - D_i+1| <= 4.0)
            taper_violations = np.abs(np.diff(x_candidate)) > 4.0
            if np.any(taper_violations):
                # Smooth out steps that are too aggressive back toward the safe anchor
                x_candidate = 0.7 * x_candidate + 0.3 * safe_anchor
            
            # Extract static criteria
            buoy, gm, taper = _static_margins(x_candidate)
            
            # Condition: Must float (buoy > 0) AND be stable (gm > 0.5)
            if buoy >= 0.0 and gm >= 0.50:
                print(f"[REPAIR] Design {idx+1} passed verification at blend alpha={alpha:.2f} (Buoyancy Margin: {buoy/1e3:.1f}t, GM: {gm:.2f}m)")
                break
                
            # If failing, blend a bit more of the safe uniform cylinder profile into it
            attempt += 1
            alpha += 0.05
            x_candidate = (1.0 - alpha) * x + alpha * safe_anchor
            
        else:
            # Fallback safety toggle in case optimization loop maxes out
            print(f"[WARN] Design {idx+1} could not be fully repaired. Reverting to safe anchor.")
            x_candidate = np.copy(safe_anchor)
            
        verified_guesses.append(x_candidate)
        
    return verified_guesses


def run_raster_search(waves, wind):
    guesses = generate_valid_raster_guesses(num_designs=3)  # Keeping to 3 distinct seeds for timeline budgeting
    results_summary = []
    bounds = [(4.0, UPPER_BOUND_D)] * N_NODES
    
    # Define output directory for files
    output_dir = os.path.join(BASE, 'assignment5', 'python', 'outputVariables')
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, x0 in enumerate(guesses):
        design_id = idx + 1
        print(f"\n=== RUNNING MULTI-START RASTER SEARCH: DESIGN {design_id} ===")
        res, c0, c_final = execute_optimization_stage(
            x0=x0, bounds=bounds, waves=waves, wind=wind,
            time_final=400.0, time_step=0.5, transient_time=150.0,
            maxiter=50, ftol=1e-4, stage_name=f"Raster_{design_id}"
        )
        
        # Store results for the end comparison table
        results_summary.append({
            'id': design_id,
            'mass': res.fun,
            'constraints': c_final,
            'x': res.x
        })
        print(f"Design {design_id} Optimized Objective (Mass): {res.fun:.3e} kg")
        
        # --- EXTRACT INITIAL GUESS PROFILES ---
        D0, L0, dr0, b0 = _split_design(x0)
        s0 = build_structure(dr0, D0, L0, b0, SPAR0)
        z_edges0, cum_mass0 = _get_weight_distribution(s0, D0, L0, b0)
        extended_diameters0 = np.append(D0, D0[-1])
        
        # --- EXTRACT FINAL OPTIMIZED PROFILES ---
        D, L, dr, b = _split_design(res.x)
        s = build_structure(dr, D, L, b, SPAR0)
        z_edges, cum_mass = _get_weight_distribution(s, D, L, b)
        extended_diameters = np.append(D, D[-1])
        
        # --- EXPORT METADATA, INITIAL & FINAL VALUES TO CSV ---
        csv_filename = os.path.join(output_dir, f'optimized_design_raster_{design_id}.csv')
        try:
            with open(csv_filename, 'w', encoding='utf-8') as f:
                # Store Initial and Final Objective & Constraints as comments
                f.write(f"# Design_ID: {design_id}\n")
                f.write(f"# Initial_Mass_kg: {objective(x0):.6f}\n")
                f.write(f"# Final_Mass_Objective_kg: {res.fun:.6f}\n")
                f.write(f"# Initial_Surge_Margin: {c0[0]:.6f}\n")
                f.write(f"# Final_Surge_Margin: {c_final[0]:.6f}\n")
                f.write(f"# Initial_Pitch_Margin: {c0[1]:.6f}\n")
                f.write(f"# Final_Pitch_Margin: {c_final[1]:.6f}\n")
                
                # Tabular configuration columns
                f.write("Point_Index,Z_Elevation_m,Init_Diameter_m,Final_Diameter_m,Init_Cum_Mass_tonnes,Final_Cum_Mass_tonnes\n")
                for p_i in range(len(z_edges)):
                    f.write(
                        f"{p_i},{z_edges[p_i]:.4f},"
                        f"{extended_diameters0[p_i]:.4f},{extended_diameters[p_i]:.4f},"
                        f"{cum_mass0[p_i]/1e3:.4f},{cum_mass[p_i]/1e3:.4f}\n"
                    )
                    
            print(f"[CSV] Successfully saved initial and optimized profile for design {design_id} to {os.path.basename(csv_filename)}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV for design {design_id}: {e}")
        
    # =========================================================================
    # MULTI-START COMPARISON TABLE GENERATION
    # =========================================================================
    sep = '=' * 96
    print(f"\n{sep}")
    print(f"  MULTI-START RASTER SEARCH: FINAL OPTIMIZATION COMPARISON SUMMARY")
    print(f"  (Note: Constraint values shown are normalized margins; >= 0 indicates feasibility)")
    print(sep)
    
    header = f"  {'Design ID':<10} | {'Mass [kg]':<15} | {'Surge RMS':<12} | {'Pitch RMS':<12} | {'Buoyancy':<12} | {'GM margin':<10}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    
    for res_dict in results_summary:
        c = res_dict['constraints']
        print(
            f"  Design {res_dict['id']:<2} | "
            f"{res_dict['mass']:13.4e}   | "
            f"{c[0]:10.4f}   | "
            f"{c[1]:10.4f}   | "
            f"{c[2]:10.4f}   | "
            f"{c[3]:10.4f}"
        )
        
    print(sep)
    print()
        
    return [item['x'] for item in results_summary]
        
    # =========================================================================
    # MULTI-START COMPARISON TABLE GENERATION
    # =========================================================================
    sep = '=' * 96
    print(f"\n{sep}")
    print(f"  MULTI-START RASTER SEARCH: FINAL OPTIMIZATION COMPARISON SUMMARY")
    print(f"  (Note: Constraint values shown are normalized margins; >= 0 indicates feasibility)")
    print(sep)
    
    header = f"  {'Design ID':<10} | {'Mass [kg]':<15} | {'Surge RMS':<12} | {'Pitch RMS':<12} | {'Buoyancy':<12} | {'GM margin':<10}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    
    for res_dict in results_summary:
        c = res_dict['constraints']
        print(
            f"  Design {res_dict['id']:<2} | "
            f"{res_dict['mass']:13.4e}   | "
            f"{c[0]:10.4f}   | "
            f"{c[1]:10.4f}   | "
            f"{c[2]:10.4f}   | "
            f"{c[3]:10.4f}"
        )
        
    print(sep)
    print()
        
    return [item['x'] for item in results_summary]


def check_design_space_convexity(X_a, X_b, waves, wind):
    """
    Maps the trajectory between two optimized local minima to check for non-convexity.
    Formula: X = X_a * (1 - alpha) + X_b * alpha
    """
    alphas = np.linspace(0.0, 1.0, 11)  # 11 tracking slots over line profile segment
    objectives = []
    feasibility_flags = []
    
    t_integration = np.arange(0.0, 400.0, 1.0)
    
    print("\n=== STARTING CONVEXITY MAPPING LINE-SEARCH ===")
    print(f"{'Alpha':<8} | {'Objective (Mass)':<18} | {'Status':<12}")
    print("-" * 45)
    
    for alpha in alphas:
        # Calculate intermediate design vector
        X_interp = X_a * (1.0 - alpha) + X_b * alpha
        
        # Calculate objective
        obj_val = objective(X_interp)
        objectives.append(obj_val)
        
        # Calculate constraints
        c_vals = constraints_fun(X_interp, waves, wind, t_integration, transient_time=150.0)
        taper_vals = _taper_margins(X_interp)
        
        # Check if all elements in c_vals and taper_vals are >= 0
        is_feasible = np.all(c_vals >= 0) and np.all(taper_vals >= 0)
        feasibility_flags.append(is_feasible)
        
        status_str = "FEASIBLE" if is_feasible else "VIOLATED"
        print(f"{alpha:8.2f} | {obj_val:18.4e} | {status_str}")
        
    # Plotting the landscape line profile
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plot mass
    color = 'tab:blue'
    ax1.set_xlabel(r'Interpolation factor ($\alpha$) [0 = Design A, 1 = Design B]')
    ax1.set_ylabel('Total Mass [kg]', color=color)
    ax1.plot(alphas, objectives, marker='o', color=color, linewidth=2, label='Objective Path')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Highlight infeasible zones shading
    for i in range(len(alphas)-1):
        if not feasibility_flags[i] or not feasibility_flags[i+1]:
            ax1.axvspan(alphas[i], alphas[i+1], color='red', alpha=0.15, label='Infeasible Region' if i==0 else "")
            
    plt.title('Design Space Convexity Audit\n(Non-convex if peaks exist or shaded regions violate constraints)')
    fig.tight_layout()
    
    out_dir = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
    plt.savefig(os.path.join(out_dir, 'design_space_convexity_check.png'), dpi=200)
    plt.close()
    print("\n[POST] Convexity mapping visualization saved as 'design_space_convexity_check.png'")


def run():
    bounds = [(4.0, UPPER_BOUND_D)] * N_NODES 

    print("[ENV] Loading or computing 1200 s environment...")
    waves = _load_or_compute_waves()
    wind  = _load_or_compute_wind()
    print("[ENV] Environment ready.")
    plot_environment(waves, wind)

    # Initial guess
    x0 = np.array([5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 25.0, 25.0, 25.0, 25.0], dtype=float)

    buoy0, gm0, taper0 = _static_margins(x0)
    print(
        "[INIT] Static margins for selected design: "
        f"buoyancy={buoy0:.3e} kg, GM={gm0:.3f} m, "
        f"min taper margin={taper0:.3f} m"
    )
    
    stage1_move_limits = np.full(N_NODES, 4.0, dtype=float)
    stage2_move_limits = np.full(N_NODES, 2.0, dtype=float)

    # =========================================================================
    # STAGE 1: COARSE & SWIFT DESIGN REGIME SWEEP
    # =========================================================================
    res_stage1, c0_s1, cf_s1 = execute_optimization_stage(
        x0=x0, bounds=bounds, waves=waves, wind=wind,
        time_final=400.0,
        time_step=0.5,
        transient_time=150.0,
        maxiter=200,
        ftol=1e-5,
        stage_name="STAGE_1_COARSE",
        move_limits=stage1_move_limits
    )
    print_summary(res_stage1, x0, "STAGE 1 RESULTS", c0_s1, cf_s1)

    # =========================================================================
    # STAGE 2: HIGH-FIDELITY STRUCTURAL POLISH
    # =========================================================================
    print("\n[STAGE 2] Starting high-fidelity refinement from Stage 1 solution...")
    res_final, c0_s2, cf_s2 = execute_optimization_stage(
        x0=res_stage1.x, bounds=bounds, waves=waves, wind=wind,
        time_final=1200.0,
        time_step=0.5,
        transient_time=600.0,
        maxiter=15,
        ftol=1e-3,
        stage_name="STAGE_2_FINE",
        move_limits=stage2_move_limits
    )
    print_summary(res_final, res_stage1.x, "STAGE 2 RESULTS", c0_s2, cf_s2)

    sep = '=' * 74
    print(f"\n{sep}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"  {res_final.message}")
    print(sep)

    out_dir = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
    os.makedirs(out_dir, exist_ok=True)

    save_results_plot(res_stage1, x0,         "stage1_coarse")
    save_results_plot(res_final,  res_stage1.x, "stage2_fine")

    D_stage1, L_stage1, dr_stage1, b_stage1 = _split_design(res_stage1.x)
    D0, L0, dr0, b0 = _split_design(x0)
    Df, Lf, drf, bf = _split_design(res_final.x)

    s0 = build_structure(dr0,        D0,       L0,       b0,       SPAR0)
    s1 = build_structure(dr_stage1, D_stage1, L_stage1, b_stage1, SPAR0)
    sf = build_structure(drf,        Df,       Lf,       bf,       SPAR0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_d = axes[0]
    _plot_diameter_profile(ax_d, s0, D0, 'Initial Guess', style=':')
    _plot_diameter_profile(ax_d, s1, D_stage1, 'Stage 1 (Coarse)', style='--')
    _plot_diameter_profile(ax_d, sf, Df, 'Stage 2 (Final)', style='-')
    ax_d.set_xlabel('Diameter [m]')
    ax_d.set_ylabel('Elevation z [m]')
    ax_d.set_title('Profile Comparison')
    ax_d.grid(True, alpha=0.3)
    ax_d.legend()

    ax_m = axes[1]
    z0, m0 = _get_weight_distribution(s0, D0, L0, b0)
    z1, m1 = _get_weight_distribution(s1, D_stage1, L_stage1, b_stage1)
    zf, mf = _get_weight_distribution(sf, Df, Lf, bf)
    ax_m.step(m0 / 1e3, z0, ':',  where='post', label='Initial Guess')
    ax_m.step(m1 / 1e3, z1, '--', where='post', label='Stage 1 (Coarse)')
    ax_m.step(mf / 1e3, zf, '-',  where='post', label='Stage 2 (Final)')
    ax_m.set_xlabel('Cumulative mass [tonnes]')
    ax_m.set_ylabel('Elevation z [m]')
    ax_m.set_title('Mass Distribution Comparison')
    ax_m.grid(True, alpha=0.3)
    ax_m.legend()

    fpath = os.path.join(out_dir, 'three_way_comparison_ro2.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[POST] Three-way design comparison chart saved to {os.path.basename(fpath)}")

    # =========================================================================
    # MULTI-START RASTER SEARCH & CONVEXITY MAPPING
    # =========================================================================
    minima = run_raster_search(waves, wind)
    
    if len(minima) >= 2:
        check_design_space_convexity(minima[0], minima[1], waves, wind)


if __name__ == '__main__':
    run()