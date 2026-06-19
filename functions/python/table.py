import numpy as np
import json

def print_preliminary_computations(data):
    print("\n=== Preliminary Computations ===")
    def fmt(val, unit):
        try:
            return f"{float(val):.2f} {unit}"
        except (ValueError, TypeError):
            return f"{val} {unit}" if val != 'N/A' else "N/A"

    print(f"Center of Buoyancy: {fmt(data.get('zCB', 'N/A'), '[m]')}")
    print(f"Spar length: {fmt(data.get('ls', 'N/A'), '[m]')}")
    print(f"Spar mass: {fmt(data.get('ms', 'N/A'), '[kg]')}")
    print(f"Floater mass with ballast: {fmt(data.get('mf', 'N/A'), '[kg]')}")
    print(f"Spar center of mass without ballast: {fmt(data.get('zCMs', 'N/A'), '[m]')}")
    print(f"Floater center of mass with ballast: {fmt(data.get('zCMf', 'N/A'), '[m]')}")
    print(f"Spar inertia about CM: {data.get('ICMs', 'N/A')}")
    print(f"Floater inertia about CM: {data.get('ICMf', 'N/A')}")
    print(f"Water Plane Inertia: {data.get('IAA', 'N/A')}")
    print(f"\n--- Total Structure Properties ---")
    print(f"Total mass (MTot): {fmt(data.get('MTot', 'N/A'), '[kg]')}")
    print(f"Total center of mass (zCM,Tot): {fmt(data.get('zCM_Tot', 'N/A'), '[m]')}")
    print(f"Total moment of inertia about O (IO,Tot): {data.get('IO_Tot', 'N/A')}")


def print_system_matrices(data):
    print("\n=== System Matrices ===")
    print("Mass Matrix (M):\n", np.array(data.get('M', 'N/A')))
    print("Added Mass Matrix (A):\n", np.array(data.get('A', 'N/A')))
    print("Damping Matrix (B):\n", np.array(data.get('B', 'N/A')))
    print("Restoring Matrix (C):\n", np.array(data.get('C', 'N/A')))


def print_natural_frequencies(data):
    print("\n=== Natural Frequencies ===")
    fnat = data.get('fnat', ['N/A', 'N/A'])
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
    except Exception:
        print("Surge period: N/A")
        print("Pitch period: N/A")
    print(f"Heave period: {data.get('Theave', 'N/A')} [s]")


def print_all_tables(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    print_preliminary_computations(data)
    print_system_matrices(data)
    print_natural_frequencies(data)

# Example usage:
# print_all_tables('outputVariables/SparBuoyDataComplete.json')
