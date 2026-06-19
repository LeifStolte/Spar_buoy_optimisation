import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from common import loadFromJSON, generateRandomPhases
from waves import (
    calculateJONSWAPSpectrum,
    calculateFreeSurfaceElevationTimeSeriesFFT,
    calculateKinematicsFFT,
)
from wind import calculateKaimalSpectrum, calculateWindTimeSeriesFFT
from integration import ode4
from floaterIntegration import dqdt as floater_dqdt

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
N_NODES = 20


def _time_value(info, *keys, default=None):
    for key in keys:
        if key in info:
            return info[key]
    return default


def _split_design(x):
    x = np.asarray(x, dtype=float)
    return x[:N_NODES], x[N_NODES:]


def _step_profile(values, n_points):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        return np.full(n_points, values.item(), dtype=float)
    idx = np.linspace(0, values.size - 1, n_points)
    return values[np.floor(idx).astype(int)]


def build_structure(draft, D_nodes, th_nodes, ballast_mass, spar_template):
    """Build a section-wise spar model and an effective scalar model for dynamics."""
    s = dict(spar_template)
    s['draft'] = float(draft)
    s['M_Ballast'] = float(ballast_mass)

    fb = float(spar_template.get('fb', 0.0))
    ls = draft + fb
    s['ls'] = ls

    rhos = float(spar_template.get('rho_Steel', 7850.0))
    rhow = float(spar_template.get('rho_Water', 1025.0))
    g = 9.81

    zbot = -draft
    z_edges = np.linspace(zbot, 0.0, N_NODES + 1)
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    section_lengths = np.diff(z_edges)

    D_nodes = _step_profile(D_nodes, N_NODES)
    th_nodes = _step_profile(th_nodes, N_NODES)

    # Section properties
    outer_area = np.pi * (D_nodes / 2.0) ** 2
    inner_area = np.pi * np.maximum((D_nodes - 2.0 * th_nodes) / 2.0, 0.0) ** 2
    shell_area = outer_area - inner_area
    section_volumes = outer_area * section_lengths
    section_masses = shell_area * rhos * section_lengths

    ms = float(np.sum(section_masses))
    mf = ms + ballast_mass

    mt = float(spar_template.get('M_Tower', 0.0))
    mtu = float(spar_template.get('M_Turbine', 0.0))
    mtot = mf + mt + mtu

    # Centers of mass / buoyancy
    buoyant_volume = float(np.sum(section_volumes))
    zCB = float(np.sum(section_volumes * z_mid) / buoyant_volume) if buoyant_volume > 0 else zbot / 2.0
    zCMs = float(np.sum(section_masses * z_mid) / ms) if ms > 0 else fb - ls / 2.0
    zballst = float(spar_template.get('Ballast_COG', 0.0))
    zCMf = (zCMs * ms + zballst * ballast_mass) / mf if mf > 0 else 0.0
    zCMt = float(spar_template.get('z_CM_Tower', 0.0))
    zturb = float(spar_template.get('z_CM_Turbine', 0.0))
    zCMtot = (zCMf * mf + zCMt * mt + zturb * mtu) / mtot if mtot > 0 else 0.0

    # Inertia about flotation point
    ICM_sections = (1.0 / 12.0) * section_masses * (3.0 * (D_nodes / 2.0) ** 2 + section_lengths ** 2)
    ICMs = float(np.sum(ICM_sections))
    I_Spar = float(np.sum(ICM_sections + section_masses * (z_mid ** 2)))
    ICMt = float(spar_template.get('I_CM_Tower', 0.0))
    I_tower = ICMt + mt * (zCMt ** 2)
    I0_turbine = mtu * (zturb ** 2)
    I0_ballast = ballast_mass * (zballst ** 2)
    IOtot = I_Spar + I_tower + I0_turbine + I0_ballast

    # Effective diameter for the dynamic floating model
    D_eff = float(np.mean(D_nodes))
    D_waterline = float(D_nodes[-1])

    # Matrices for the dynamic model
    s['M'] = np.array([[mtot, mtot * zCMtot], [mtot * zCMtot, IOtot]], dtype=float)
    cm_eff = float(spar_template.get('CM', 2.0)) - 1.0
    A1 = (np.pi / 4.0) * rhow * D_eff ** 2 * cm_eff * draft
    A15 = -(np.pi / 4.0) * rhow * D_eff ** 2 * cm_eff * draft ** 2 / 2.0
    A5 = (np.pi / 4.0) * rhow * D_eff ** 2 * cm_eff * draft ** 3 / 3.0
    s['A'] = np.array([[A1, A15], [A15, A5]], dtype=float)
    s['B'] = np.array([[float(spar_template.get('B11', 0.0)), 0.0], [0.0, 0.0]], dtype=float)

    IAA = np.pi * D_waterline ** 4 / 64.0
    Cs5 = rhow * g * IAA + mtot * g * (zCB - zCMtot)
    Chst = np.array([[0.0, 0.0], [0.0, Cs5]], dtype=float)
    Kmoor = float(spar_template.get('K_Moor', 0.0))
    zmoor = float(spar_template.get('z_Moor', 0.0))
    Cmoor = np.array([[Kmoor, Kmoor * zmoor], [Kmoor * zmoor, Kmoor * zmoor ** 2]], dtype=float)
    s['C'] = Chst + Cmoor

    # Fields used by the dynamic solver and for plotting
    s['zCB'] = zCB
    s['V_disp'] = buoyant_volume
    s['buoyant_mass'] = rhow * buoyant_volume
    s['MTot'] = mtot
    s['zCM_Tot'] = zCMtot
    s['IO_Tot'] = IOtot
    s['ms'] = ms
    s['mf'] = mf
    s['zCMs'] = zCMs
    s['zCMf'] = zCMf
    s['ICMs'] = ICMs
    s['IAA'] = IAA
    s['DProfile'] = _step_profile(D_nodes, 100)
    s['ThicknessProfile'] = _step_profile(th_nodes, 100)
    s['DMonopile'] = D_eff
    s['Thickness'] = float(np.mean(th_nodes))
    s['z'] = np.linspace(zbot, 0.0, 100)
    s['zBeamNodal'] = np.linspace(zbot, 0.0, 101)
    s['phiNodal'] = np.ones_like(s['zBeamNodal'])

    return s


def objective(x):
    D_nodes, th_nodes = _split_design(x)
    s = build_structure(FIXED_DRAFT, D_nodes, th_nodes, FIXED_BALLAST, SPAR0)
    return s['mf'] + s['M_Tower'] + s['M_Turbine']


def constraints_fun(x, surge_rms_max=SURGE_RMS_MAX, pitch_rms_max_deg=PITCH_RMS_MAX_DEG):
    D_nodes, th_nodes = _split_design(x)
    s = build_structure(FIXED_DRAFT, D_nodes, th_nodes, FIXED_BALLAST, SPAR0)

    time_final = _time_value(TIME_INFO, 'time_final', 'TDur', default=1200.0)
    time_step = _time_value(TIME_INFO, 'time_step', 'dt', default=0.05)
    transient_time = _time_value(TIME_INFO, 'TTrans', default=600.0)

    waves = {
        'Hs': SPAR0.get('significant_wave', 6.0),
        'Tp': SPAR0.get('wave_period', 10.0),
        'TDur': time_final,
        'dt': time_step,
        'fHighCut': 0.5,
        'h': TIME_INFO.get('water_depth', 320.0),
        'z': s['z'],
        't': np.arange(0.0, time_final, time_step),
    }
    waves = calculateJONSWAPSpectrum(waves)
    waves = generateRandomPhases(waves, seed=2)
    waves = calculateFreeSurfaceElevationTimeSeriesFFT(waves)
    waves = calculateKinematicsFFT(waves)

    wind = {
        'V_10': SPAR0.get('wind_speed', 8.0),
        'I': TIME_INFO.get('turbulence_intensity', 0.05),
        'l': TIME_INFO.get('turbulence_length_scale', 340.2),
        'TDur': time_final,
        'dt': time_step,
        'fHighCut': 0.5,
        't': np.arange(0.0, time_final, time_step),
    }
    wind = calculateKaimalSpectrum(wind)
    wind = generateRandomPhases(wind, seed=1)
    wind = calculateWindTimeSeriesFFT(wind)

    rotor = dict(ROTOR0)
    if 'ARotor' not in rotor:
        rotor['ARotor'] = 0.25 * np.pi * rotor.get('DRotor', 0.0) ** 2
    rotor['gamma'] = 0.0
    rotor['active'] = True

    t_integration = np.arange(0.0, time_final, 2.0 * time_step)
    q0 = np.zeros(5)

    try:
        q = ode4(floater_dqdt, t_integration, q0, s, rotor, waves, wind)
    except Exception as exc:
        print('Integration failure:', exc)
        return np.array([-1.0, -1.0, -1.0, -1.0])

    post_transient = t_integration > transient_time
    if not np.any(post_transient):
        post_transient = np.arange(len(t_integration)) >= int(0.5 * len(t_integration))

    surge_rms = np.sqrt(np.mean(q[post_transient, 0] ** 2))
    pitch_rms = np.sqrt(np.mean((np.rad2deg(q[post_transient, 1])) ** 2))
    buoyancy_margin = s['buoyant_mass'] - s['MTot']
    gm = (s['IAA'] / max(s['V_disp'], 1e-9)) + s['zCB'] - s['zCM_Tot']

    return np.array([
        surge_rms_max - surge_rms,
        pitch_rms_max_deg - pitch_rms,
        buoyancy_margin - FLOATING_MARGIN,
        gm - GM_MIN,
    ])


def run(maxiter=50, surge_rms_max=SURGE_RMS_MAX, pitch_rms_max_deg=PITCH_RMS_MAX_DEG):
    x0 = np.array([
        *([25.0] * N_NODES),
        *([0.10] * N_NODES),
    ], dtype=float)

    bounds = [(2.0, 25.0)] * N_NODES + [(0.0002, 0.15)] * N_NODES

    def surge_constraint(x):
        return constraints_fun(x, surge_rms_max=surge_rms_max, pitch_rms_max_deg=pitch_rms_max_deg)[0]

    def pitch_constraint(x):
        return constraints_fun(x, surge_rms_max=surge_rms_max, pitch_rms_max_deg=pitch_rms_max_deg)[1]

    def buoyancy_constraint(x):
        return constraints_fun(x, surge_rms_max=surge_rms_max, pitch_rms_max_deg=pitch_rms_max_deg)[2]

    def stability_constraint(x):
        return constraints_fun(x, surge_rms_max=surge_rms_max, pitch_rms_max_deg=pitch_rms_max_deg)[3]

    cons = (
        {'type': 'ineq', 'fun': surge_constraint},
        {'type': 'ineq', 'fun': pitch_constraint},
        {'type': 'ineq', 'fun': buoyancy_constraint},
        {'type': 'ineq', 'fun': stability_constraint},
    )

    iteration_state = {'k': 0}

    def callback(xk):
        iteration_state['k'] += 1
        obj = objective(xk)
        c = constraints_fun(xk, surge_rms_max=surge_rms_max, pitch_rms_max_deg=pitch_rms_max_deg)
        Dk, tk = _split_design(xk)
        print(
            f"iter {iteration_state['k']:03d} | "
            f"D[min/mean/max]={Dk.min():.3f}/{Dk.mean():.3f}/{Dk.max():.3f} m | "
            f"t[min/mean/max]={tk.min():.4f}/{tk.mean():.4f}/{tk.max():.4f} m | "
            f"mass={obj:.3f} kg | surge_margin={c[0]:.3f} | pitch_margin={c[1]:.3f} | "
            f"buoyancy_margin={c[2]:.3f} kg | GM_margin={c[3]:.3f} m"
        )

    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        callback=callback,
        options={'maxiter': maxiter, 'ftol': 1e-3, 'disp': True},
    )

    # Final geometry figure: step profiles along height
    D0, t0 = _split_design(x0)
    Df, tf = _split_design(res.x)
    height_edges = np.linspace(0.0, FIXED_DRAFT + SPAR0.get('fb', 0.0), N_NODES + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    axes[0].step(height_edges[:-1], D0, '--', where='post', label='Initial')
    axes[0].step(height_edges[:-1], Df, '-', where='post', label='Optimized')
    axes[0].set_xlabel('Diameter [m]')
    axes[0].set_ylabel('Height along spar [m]')
    axes[0].set_title('Diameter profile')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].step(height_edges[:-1], t0, '--', where='post', label='Initial')
    axes[1].step(height_edges[:-1], tf, '-', where='post', label='Optimized')
    axes[1].set_xlabel('Wall thickness [m]')
    axes[1].set_title('Thickness profile')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle('Spar geometry change along total spar height')
    fig.tight_layout()

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'assignment5',
        'python',
        'outputFig',
    )
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
