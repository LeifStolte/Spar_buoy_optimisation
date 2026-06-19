"""
Main script for assignment 5.

This script loads the input JSON files, computes the primary spar
preliminary quantities, writes SparBuoyDataComplete.json, and prints
the summary tables.
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HELPERS_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "functions", "python"))
if HELPERS_PATH not in sys.path:
    sys.path.insert(0, HELPERS_PATH)

from common import loadFromJSON, saveToJSON, loadConstants
from table import print_all_tables

from run_optimizer import run as run_optimizer


def compute_spar_summary(spar_data: dict, constants: dict) -> dict:
    """Compute the main spar summary fields used throughout the assignment."""
    result = dict(spar_data)

    g = constants["g"]
    rhow = constants["rho_water"]

    rhos = result.get("rho_Steel", 7850.0)
    mtu = result.get("M_Turbine", 0.0)
    zturb = result.get("z_CM_Turbine", 0.0)
    fb = result.get("fb", 0.0)
    draft = result.get("draft", 0.0)
    Dspar = result.get("DMonopile", 0.0)
    th = result.get("Thickness", 0.0)
    Kmoor = result.get("K_Moor", 0.0)
    zmoor = result.get("z_Moor", 0.0)
    Cm = result.get("CM", 1.0) - 1.0
    mt = result.get("M_Tower", 0.0)
    zCMt = result.get("z_CM_Tower", 0.0)
    ICMt = result.get("I_CM_Tower", 0.0)
    BallastCOG = result.get("Ballast_COG", 0.0)
    mb = result.get("M_Ballast", 0.0)

    zbot = -draft
    zballst = BallastCOG
    zCB = zbot / 2.0
    ls = draft + fb

    # Shell mass approximation
    ms = ((Dspar / 2.0) ** 2 * np.pi - ((Dspar - 2.0 * th) / 2.0) ** 2 * np.pi) * rhos * ls
    mf = ms + mb
    zCMs = fb - ls / 2.0
    zCMf = (zCMs * ms + zballst * mb) / mf if mf != 0 else 0.0

    ICMs = (1.0 / 12.0) * ms * (3.0 * (((Dspar - Dspar - 2.0 * th) / 2.0) ** 2) + ls ** 2)
    ICMf = (1.0 / 12.0) * mf * (3.0 * ((Dspar / 2.0 - (Dspar - 2.0 * th) / 2.0) ** 2) + ls ** 2)

    mtot = mf + mt + mtu
    zCMtot = (zCMf * mf + zCMt * mt + zturb * mtu) / mtot if mtot != 0 else 0.0

    I_Spar = ICMs + ms * (zCMs ** 2)
    I_tower = ICMt + mt * (zCMt ** 2)
    I0_turbine = mtu * (zturb ** 2)
    I0_ballast = mb * (zballst ** 2)
    IOtot = I_Spar + I_tower + I0_ballast + I0_turbine

    M = np.array([[mtot, mtot * zCMtot], [mtot * zCMtot, IOtot]], dtype=float)

    A1 = (np.pi / 4.0) * rhow * Dspar ** 2 * Cm * draft
    A15 = -(np.pi / 4.0) * rhow * Dspar ** 2 * Cm * draft ** 2 / 2.0
    A5 = (np.pi / 4.0) * rhow * Dspar ** 2 * Cm * draft ** 3 / 3.0
    A = np.array([[A1, A15], [A15, A5]], dtype=float)

    B11 = result.get("B11", 0.0)
    B = np.array([[B11, 0.0], [0.0, 0.0]], dtype=float)

    IAA = np.pi * Dspar ** 4 / 64.0
    Cs5 = rhow * g * IAA + mtot * g * (zCB - zCMtot)
    Chst = np.array([[0.0, 0.0], [0.0, Cs5]], dtype=float)
    Cmoor = np.array([[Kmoor, Kmoor * zmoor], [Kmoor * zmoor, Kmoor * zmoor ** 2]], dtype=float)
    C = Chst + Cmoor

    fnat = np.sqrt(np.linalg.eigvals(C @ np.linalg.inv(M + A))) / (2.0 * np.pi)

    result.update(
        {
            "zCB": float(zCB),
            "ls": float(ls),
            "ms": float(ms),
            "mf": float(mf),
            "zCMs": float(zCMs),
            "zCMf": float(zCMf),
            "ICMs": float(ICMs),
            "ICMf": float(ICMf),
            "IAA": float(IAA),
            "MTot": float(mtot),
            "zCM_Tot": float(zCMtot),
            "IO_Tot": float(IOtot),
            "M": M,
            "A": A,
            "B": B,
            "C": C,
            "fnat": fnat,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess spar data and optionally run the optimizer.")
    parser.add_argument("--no-optimize", action="store_true", help="Skip the geometry optimizer after preprocessing.")
    parser.add_argument("--maxiter", type=int, default=50, help="Maximum optimizer iterations.")
    parser.add_argument("--surge-rms-max", type=float, default=25.0, help="Maximum allowed surge RMS [m].")
    parser.add_argument("--pitch-rms-max", type=float, default=5.0, help="Maximum allowed pitch RMS [deg].")
    args = parser.parse_args()

    input_vars_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "assignment5", "python", "inputVariables"))
    time_json_path = os.path.join(input_vars_path, "time.json")
    spar_json_path = os.path.join(input_vars_path, "SparBuoyData.json")

    timeInfo = loadFromJSON(time_json_path)
    sparData = loadFromJSON(spar_json_path)
    constants = loadConstants()

    summary = compute_spar_summary(sparData, constants)

    out_dir = os.path.join(SCRIPT_DIR, "outputVariables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "SparBuoyDataComplete.json")
    saveToJSON(summary, out_path)

    print(f"Saved {out_path}")
    print_all_tables(out_path)

    if not args.no_optimize:
        print("\nStarting geometry optimization...")
        result = run_optimizer(
            maxiter=args.maxiter,
            surge_rms_max=args.surge_rms_max,
            pitch_rms_max_deg=args.pitch_rms_max,
        )

        x_opt = np.asarray(result.x, dtype=float)
        print("\nOptimized design variables:")
        print(f"  diameter = {x_opt[0]:.6g} m")
        print(f"  thickness = {x_opt[1]:.6g} m")
        print(f"  fixed draft = {summary.get('draft', sparData.get('draft', 0.0)):.6g} m")
        print(f"  fixed ballast = {sparData.get('M_Ballast', 0.0):.6g} kg")
        print(f"Objective (mass) = {result.fun:.6g}")
        print(f"Success = {result.success}")
        print(f"Message  = {result.message}")


if __name__ == "__main__":
    main()
