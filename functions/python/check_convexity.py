import os
import glob
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

CSV_DIR = os.path.join(BASE_DIR, 'assignment5', 'python', 'outputVariables')
FIG_DIR = os.path.join(BASE_DIR, 'assignment5', 'python', 'outputFig')
os.makedirs(FIG_DIR, exist_ok=True)

def parse_enriched_csv(filepath):
    """Parses custom metadata lines along with the execution data arrays."""
    metadata = {}
    tabular_lines = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                clean_line = line.lstrip('#').strip()
                parts = clean_line.split(':')
                if len(parts) == 2:
                    metadata[parts[0].strip()] = float(parts[1].strip())
            else:
                tabular_lines.append(line)
                
    data = np.genfromtxt(StringIO("".join(tabular_lines)), delimiter=',', names=True)
    return metadata, data

def generate_comparison_plots():
    csv_pattern = os.path.join(CSV_DIR, 'optimized_design_raster_*.csv')
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"[WARN] No optimized design records found in: {CSV_DIR}")
        return

    # Create a clean 2-Panel Plotting Layout (1 Row, 2 Columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Establish consistent tracking palettes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    print(f"\nProcessing {len(csv_files)} data runs for visual state overlay...")

    for idx, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        label = filename.replace('optimized_design_raster_', '').replace('.csv', '').replace('_', ' ').title()
        color = colors[idx % len(colors)]
        
        try:
            meta, data = parse_enriched_csv(filepath)
            
            # --- PANEL 1: DIAMETER PROFILE PROFILE LAYER ---
            # Initial guess configuration state
            ax1.step(data['Init_Diameter_m'], data['Z_Elevation_m'], where='post', 
                     color=color, linestyle='--', linewidth=1.5, alpha=0.65,
                     label=f'Design {idx+1} Initial')
            # Final optimal configuration state
            ax1.step(data['Final_Diameter_m'], data['Z_Elevation_m'], where='post', 
                     color=color, linestyle='-', linewidth=2.5,
                     label=f'Design {idx+1} Optimized')
            
            # --- PANEL 2: CUMULATIVE STRUCTURAL WEIGHT STEP LAYER ---
            # Initial weight distribution
            ax2.step(data['Init_Cum_Mass_tonnes'], data['Z_Elevation_m'], where='post', 
                     color=color, linestyle='--', linewidth=1.5, alpha=0.65)
            # Final optimized weight distribution
            ax2.step(data['Final_Cum_Mass_tonnes'], data['Z_Elevation_m'], where='post', 
                     color=color, linestyle='-', linewidth=2.5)
            
            print(f" -> Successfully mapped initial & optimized tracks for {filename}")
            
        except Exception as e:
            print(f"[ERROR] Problem parsing database trace {filename}: {e}")

    # Formatting and reference bounds
    for ax in [ax1, ax2]:
        ax.axhline(0.0, color='black', linewidth=1.2, linestyle='--', alpha=0.5)
        ax.grid(True, which='both', linestyle=':', alpha=0.4)
        ax.set_ylabel('Elevation Relative to Waterline z [m]', fontsize=11, fontweight='bold')

    # Column 1 Aesthetics
    ax1.set_xlabel('Structural Outer Diameter [m]', fontsize=11, fontweight='bold')
    ax1.set_title('Diameter Profile Structural Evolution', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlim([2.0, 27.0])
    # Place standard tracking legend explicitly here to maintain a balanced viewport space
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    # Column 2 Aesthetics
    ax2.set_xlabel('Cumulative Structural Weight [tonnes]', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Mass Distribution Transitions', fontsize=12, fontweight='bold', pad=10)

    plt.suptitle('Structural Layout vs Weight Landscape Convergence Comparison\n(Dashed Lines = Initial Guess | Solid Lines = Final Optimized Solution)', 
                 fontsize=13, fontweight='bold', y=0.97)
    fig.tight_layout()
    
    save_path = os.path.join(FIG_DIR, 'optimized_designs_raster_comparison.png')
    fig.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[SUCCESS] Double-panel profile comparison canvas generated and exported to:\n -> {save_path}\n")

if __name__ == '__main__':
    generate_comparison_plots()