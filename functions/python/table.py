import numpy as np
import json

from models import Structure


def _fmt(val, unit):
    try:
        return f"{float(val):.2f} {unit}"
    except (ValueError, TypeError):
        return f"{val} {unit}" if val != 'N/A' else "N/A"

def print_preliminary_computations(data):
    print("\n=== Preliminary Computations ===")
    print(f"Center of Buoyancy: {_fmt(data.zCB, '[m]')}")
    print(f"Spar length: {_fmt(data.ls, '[m]')}")
    print(f"Spar mass: {_fmt(data.ms, '[kg]')}")
    print(f"Floater mass with ballast: {_fmt(data.mf, '[kg]')}")
    print(f"Spar center of mass without ballast: {_fmt(data.zCMs, '[m]')}")
    print(f"Floater center of mass with ballast: {_fmt(data.zCMf, '[m]')}")
    print(f"Spar inertia about CM: {data.ICMs if data.ICMs is not None else 'N/A'}")
    print(f"Floater inertia about CM: {data.ICMf if data.ICMf is not None else 'N/A'}")
    print(f"Water Plane Inertia: {data.IAA if data.IAA is not None else 'N/A'}")
    print("\n--- Total Structure Properties ---")
    print(f"Total mass (MTot): {_fmt(data.MTot, '[kg]')}")
    print(f"Total center of mass (zCM,Tot): {_fmt(data.zCM_Tot, '[m]')}")
    print(f"Total moment of inertia about O (IO,Tot): {data.IO_Tot if data.IO_Tot is not None else 'N/A'}")


def print_system_matrices(data):
    print("\n=== System Matrices ===")
    print("Mass Matrix (M):\n", np.array(data.M))
    print("Added Mass Matrix (A):\n", np.array(data.A))
    print("Damping Matrix (B):\n", np.array(data.B))
    print("Restoring Matrix (C):\n", np.array(data.C))


def print_natural_frequencies(data):
    print("\n=== Natural Frequencies ===")
    fnat = data.fnat if data.fnat is not None else ['N/A', 'N/A']
    def fmtf(val, unit, prec=4):
        try:
            return f"{float(val):.{prec}f} {unit}"
        except (ValueError, TypeError):
            return f"{val} {unit}" if val != 'N/A' else "N/A"
    print(f"Surge natural frequency: {fmtf(fnat[0], 'Hz')}")
    print(f"Pitch natural frequency: {fmtf(fnat[1], 'Hz')}")
    try:
        Tnat = [1/float(f) if float(f) != 0 else float('inf') for f in fnat]
        print(f"Surge period: {fmtf(Tnat[0], '[s]', 2)}")
        print(f"Pitch period: {fmtf(Tnat[1], '[s]', 2)}")
    except (ValueError, TypeError):
        print("Surge period: N/A")
        print("Pitch period: N/A")
    print(f"Heave period: {getattr(data, 'Theave', 'N/A')} [s]")


def print_all_tables(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = Structure.from_mapping(json.load(f))
    print_preliminary_computations(data)
    print_system_matrices(data)
    print_natural_frequencies(data)

# Example usage:
# print_all_tables('outputVariables/SparBuoyDataComplete.json')
