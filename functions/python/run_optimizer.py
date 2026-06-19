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

# Limits and model setup
SURGE_RMS_MAX = 25.0  # meters
PITCH_RMS_MAX_DEG = 4.0  # degrees
GM_MIN = 0.5  # meters
FLOATING_MARGIN = 0.0  # kg, buoyancy - weight
FIXED_DRAFT = float(SPAR0.get('draft', 100.0))
FIXED_BALLAST = float(SPAR0.get('M_Ballast', 0.0))
FIXED_THICKNESS = 0.04  # m
N_NODES = 10
UPPER_BOUND_D = 25

def _time_value(info, *keys, default=None):
    for key in keys:
        if key in info:
            return info[key]
    return default


def _split_design(x):
    x = np.asarray(x, dtype=float)
    return x[:N_NODES]


def _step_profile(values, n_points):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        return np.full(n_points, values.item(), dtype=float)
    idx = np.linspace(0, values.size - 1, n_points)
    return values[np.floor(idx).astype(int)]


def _design_key(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return x.tobytes()


def build_structure(draft, D_nodes, ballast_mass, spar_template, thickness=FIXED_THICKNESS):
    """Build a section-wise spar model and an effective scalar model for dynamics."""
    s = Structure.from_mapping(spar_template)
    s.draft = float(draft)
    s.M_Ballast = float(ballast_mass)

    fb = float(s.fb or 0.0)
    ls = draft + fb
    s.ls = ls

    rhos = float(s.rho_Steel or 7850.0)
    rhow = float(s.rho_Water or 1025.0)
    g = 9.81

    zbot = -draft
    z_edges = np.linspace(zbot, 0.0, N_NODES + 1)
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    section_lengths = np.diff(z_edges)

    D_nodes = _step_profile(D_nodes, N_NODES)
    th_nodes = np.full(N_NODES, float(thickness), dtype=float)

    # Section properties
    outer_area = np.pi * (D_nodes / 2.0) ** 2
    inner_area = np.pi * np.maximum((D_nodes - 2.0 * th_nodes) / 2.0, 0.0) ** 2
    shell_area = outer_area - inner_area
    section_volumes = outer_area * section_lengths
    section_masses = shell_area * rhos * section_lengths

    ms = float(np.sum(section_masses))
    mf = ms + ballast_mass

    mt = float(s.M_Tower or 0.0)
    mtu = float(s.M_Turbine or 0.0)
    mtot = mf + mt + mtu

    # Centers of mass / buoyancy
    buoyant_volume = float(np.sum(section_volumes))
    zCB = float(np.sum(section_volumes * z_mid) / buoyant_volume) if buoyant_volume > 0 else zbot / 2.0
    zCMs = float(np.sum(section_masses * z_mid) / ms) if ms > 0 else fb - ls / 2.0
    zballst = float(s.Ballast_COG or 0.0)
    zCMf = (zCMs * ms + zballst * ballast_mass) / mf if mf > 0 else 0.0
    zCMt = float(s.z_CM_Tower or 0.0)
    zturb = float(s.z_CM_Turbine or 0.0)
    zCMtot = (zCMf * mf + zCMt * mt + zturb * mtu) / mtot if mtot > 0 else 0.0

    # Inertia about flotation point
    ICM_sections = (1.0 / 12.0) * section_masses * (3.0 * (D_nodes / 2.0) ** 2 + section_lengths ** 2)
    ICMs = float(np.sum(ICM_sections))
    I_Spar = float(np.sum(ICM_sections + section_masses * (z_mid ** 2)))
    ICMt = float(s.I_CM_Tower or 0.0)
    I_tower = ICMt + mt * (zCMt ** 2)
    I0_turbine = mtu * (zturb ** 2)
    I0_ballast = ballast_mass * (zballst ** 2)
    IOtot = I_Spar + I_tower + I0_turbine + I0_ballast

    # Effective diameter for the dynamic floating model
    D_eff = float(np.mean(D_nodes))
    D_waterline = float(D_nodes[-1])

    # Matrices for the dynamic model
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

    # Fields used by the dynamic solver and for plotting
    s.zCB = zCB
    s.V_disp = buoyant_volume
    s.buoyant_mass = rhow * buoyant_volume
    s.MTot = mtot
    s.zCM_Tot = zCMtot
    s.IO_Tot = IOtot
    s.ms = ms
    s.mf = mf
    s.zCMs = zCMs
    s.zCMf = zCMf
    s.ICMs = ICMs
    s.IAA = IAA
    s.DProfile = _step_profile(D_nodes, 100)
    s.ThicknessProfile = np.full(100, float(thickness), dtype=float)
    s.DMonopile = D_eff
    s.Thickness = float(thickness)
    s.z = np.linspace(zbot, 0.0, 100)
    s.zBeamNodal = np.linspace(zbot, 0.0, 101)
    s.phiNodal = np.ones_like(s.zBeamNodal)

    return s


def objective(x):
    D_nodes = _split_design(x)
    s = build_structure(FIXED_DRAFT, D_nodes, FIXED_BALLAST, SPAR0)
    total_mass = s.mf + s.M_Tower + s.M_Turbine
    # Scale down by 1 million so the optimizer sees numbers around 1.0 to 5.0
    return total_mass / 1e6


def constraints_fun(x, surge_rms_max=SURGE_RMS_MAX, pitch_rms_max_deg=PITCH_RMS_MAX_DEG, waves=None, wind=None, t_integration=None, transient_time=None):
    D_nodes = _split_design(x)
    s = build_structure(FIXED_DRAFT, D_nodes, FIXED_BALLAST, SPAR0)
    
    if waves is None or wind is None or t_integration is None or transient_time is None:
        raise ValueError('waves, wind, t_integration, and transient_time must be provided')

    rotor = Rotor.from_mapping(ROTOR0)
    if not rotor.ARotor:
        rotor.ARotor = 0.25 * np.pi * (ROTOR0.get('DRotor', 0.0) ** 2)
    rotor.gamma = 0.0
    rotor.active = True

    q0 = np.zeros(5)

    q = ode4(floater_dqdt, t_integration, q0, s, rotor, waves, wind)

    post_transient = t_integration > transient_time
    if not np.any(post_transient):
        post_transient = np.arange(len(t_integration)) >= int(0.5 * len(t_integration))

    surge_rms = np.sqrt(np.mean(q[post_transient, 0] ** 2))
    pitch_rms = np.sqrt(np.mean((np.rad2deg(q[post_transient, 1])) ** 2))
    buoyancy_margin = s.buoyant_mass - s.MTot
    gm = (s.IAA / max(s.V_disp, 1e-9)) + s.zCB - s.zCM_Tot

    return np.array([
        (surge_rms_max - surge_rms) / surge_rms_max,           # e.g., 0.2 means 20% below max limit
        (pitch_rms_max_deg - pitch_rms) / pitch_rms_max_deg,   # e.g., 0.1 means 10% below max limit
        (buoyancy_margin - FLOATING_MARGIN) / 1e6,             # Scaled down to match objective magnitude
        (gm - GM_MIN) / GM_MIN                                 # Normalized against the target GM
    ])

def _make_cached_evaluator(surge_rms_max, pitch_rms_max_deg, waves, wind, t_integration, transient_time, feval_progress=None):
    cache = {}

    def evaluate(x):
        key = _design_key(x)
        if key not in cache:
            # update live eval progress if available
            try:
                if feval_progress is not None:
                    feval_progress.update(1)
                    feval_progress.set_postfix_str(f"nodes={len(np.asarray(x))}")
            except Exception:
                pass

            cache[key] = {
                'objective': objective(x),
                'constraints': constraints_fun(
                    x,
                    surge_rms_max=surge_rms_max,
                    pitch_rms_max_deg=pitch_rms_max_deg,
                    waves=waves,
                    wind=wind,
                    t_integration=t_integration,
                    transient_time=transient_time,
                ),
            }
        return cache[key]

    return evaluate


def run(maxiter=50, surge_rms_max=SURGE_RMS_MAX, pitch_rms_max_deg=PITCH_RMS_MAX_DEG):
    time_final = _time_value(TIME_INFO, 'time_final', 'TDur', default=1200.0)
    time_step = _time_value(TIME_INFO, 'time_step', 'dt', default=0.5)
    transient_time = _time_value(TIME_INFO, 'TTrans', default=600.0)

    # Pre-compute integration time grid first
    t_integration = np.arange(0.0, time_final, 2.0 * time_step)
    integration_dt = 2.0 * time_step  # Use integration resolution, not raw resolution
    
    # Pre-compute waves and wind once (they don't depend on design variables)
    # Generate at integration resolution to avoid unnecessary FFT overhead
    waves = Waves.from_mapping({
        'Hs': SPAR0.get('significant_wave', 6.0),
        'Tp': SPAR0.get('wave_period', 10.0),
        'TDur': time_final,
        'dt': integration_dt,
        'fHighCut': 0.5,
        'h': TIME_INFO.get('water_depth', 320.0),
        'z': np.linspace(-FIXED_DRAFT, 0.0, 100),
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

    x0 = np.array([*([6.0] * N_NODES)], dtype=float)
    bounds = [(2.0, UPPER_BOUND_D)] * N_NODES
    # Live function-evaluation progress bar (shows each *new* expensive eval)
    feval_progress = tqdm(desc='Func evals', unit='eval', leave=True, disable=False)
    evaluate = _make_cached_evaluator(
        surge_rms_max, pitch_rms_max_deg, waves, wind, t_integration, transient_time, feval_progress=feval_progress
    )

    print(f'Pre-computed environmental conditions for {len(t_integration)} time steps')
    print(f'Starting optimization: {maxiter} iterations, {len(x0)} design variables')
    print(f'Fixed thickness = {FIXED_THICKNESS:.3f} m')
    print('-' * 100)

    iteration_state = {'k': 0, 'feval_count': 0}
    progress = tqdm(total=maxiter, desc='Optimization', unit='iter', smoothing=0.1, disable=False)

    def surge_constraint(x):
        iteration_state['feval_count'] += 1
        return evaluate(x)['constraints'][0]

    def pitch_constraint(x):
        iteration_state['feval_count'] += 1
        return evaluate(x)['constraints'][1]

    def buoyancy_constraint(x):
        iteration_state['feval_count'] += 1
        return evaluate(x)['constraints'][2]

    def stability_constraint(x):
        iteration_state['feval_count'] += 1
        return evaluate(x)['constraints'][3]

    cons = (
        {'type': 'ineq', 'fun': surge_constraint},
        {'type': 'ineq', 'fun': pitch_constraint},
        {'type': 'ineq', 'fun': buoyancy_constraint},
        {'type': 'ineq', 'fun': stability_constraint},
    )

    def callback(xk):
        iteration_state['k'] += 1
        result = evaluate(xk)
        obj = result['objective']
        c = result['constraints']
        progress.update(1)
        # Show how the N_NODES design maps to a full profile and include feval counts
        Dk = _split_design(xk)
        nodes_str = ",".join(f"{v:.2f}" for v in Dk)
        profile = _step_profile(Dk, 100)
        sample = (profile[0], profile[len(profile) // 2], profile[-1])
        sample_str = ",".join(f"{v:.2f}" for v in sample)

        msg = (
            f"[iter {iteration_state['k']:03d}] "
            f"nodes=[{nodes_str}] | profile_sample=[{sample_str}] | "
            f"fevals={iteration_state['feval_count']} | t={FIXED_THICKNESS:.3f} m | "
            f"mass={obj:.0f} kg | Δsurge={c[0]:.2f} | Δpitch={c[1]:.2f}"
        )
        progress.set_postfix_str(msg)
        progress.refresh()
        tqdm.write(msg)

    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        callback=callback,
        options={'maxiter': maxiter, 'ftol': 1e-3, 'disp': False},
    )
    progress.close()
    try:
        feval_progress.close()
    except Exception:
        pass
    print('-' * 100)
    if res.success:
        tqdm.write(f'✓ Optimization converged in {iteration_state["k"]} iterations')
    else:
        tqdm.write(f'✗ Optimization stopped: {res.message}')
    print()

    # Final geometry figure: diameter profile with fixed thickness annotation
    D0 = _split_design(x0)
    Df = _split_design(res.x)
    # Use the same z-coordinates as in build_structure (from seabed to waterline)
    z_edges = np.linspace(-FIXED_DRAFT, 0.0, N_NODES + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Plot with diameter on the horizontal axis and beam position on the vertical axis
    ax.step(D0, z_edges[:-1], '--', where='post', label='Initial')
    ax.step(Df, z_edges[:-1], '-', where='post', label='Optimized')
    ax.set_xlabel('Diameter [m]')
    ax.set_ylabel('Elevation z [m] (seabed to waterline)')
    ax.set_title(f'Diameter profile, thickness fixed at {FIXED_THICKNESS:.3f} m')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(
        0.02,
        0.98,
        f'Fixed thickness = {FIXED_THICKNESS:.3f} m',
        transform=ax.transAxes,
        va='top',
        ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
    )

    fig.tight_layout()

    # Save to the repository's Spar_buoy_optimisation assignment5 outputFig (explicit BASE path)
    out_dir = os.path.join(BASE, 'assignment5', 'python', 'outputFig')
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'spar_geometry_optimization.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f'Saved geometry figure: {fig_path}')
    plt.show(block=False)
    plt.pause(0.1)

    print('Result:', res)
    return res


if __name__ == '__main__':
    run()
