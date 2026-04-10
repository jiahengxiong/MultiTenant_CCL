import matplotlib.pyplot as plt
import json
import numpy as np
import os

def plot_bar_chart(data, metric_suffix, title, ylabel, filename):
    # Extract tenant counts (sorted integers)
    tenant_counts = sorted([int(k) for k in data.keys()])
    
    # Define schemes and their display names/colors
    schemes = [
        ("Default", "Default"),
        ("Harmonics+Default", "Harmonics+Default"),
        ("CG", "CG"),
        ("Harmonics+CG", "Harmonics+CG")
    ]
    
    # Prepare data for plotting
    x = np.arange(len(tenant_counts))
    width = 0.2  # Width of each bar
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (key, label) in enumerate(schemes):
        # Construct key for data lookup (e.g., "Default" or "Default_avg")
        lookup_key = key
        if metric_suffix:
            lookup_key = f"{key}{metric_suffix}"
            
        values = [data[str(tc)][lookup_key] for tc in tenant_counts]
        
        # Offset bars
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, values, width, label=label)
        
        # Add labels on top of bars (optional, maybe too crowded)
        # ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=8)

    # Add labels and title
    ax.set_xlabel('Number of Tenants')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(tc) for tc in tenant_counts])
    ax.legend()
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")

if __name__ == "__main__":
    if not os.path.exists('experiment_results.json'):
        print("Error: experiment_results.json not found. Run run_experiments.py first.")
        exit(1)
        
    with open('experiment_results.json', 'r') as f:
        data = json.load(f)
        
    # Plot Makespan
    plot_bar_chart(
        data, 
        metric_suffix="", 
        title="Makespan vs Number of Tenants", 
        ylabel="Makespan (s)", 
        filename="makespan_comparison.png"
    )
    
    # Plot Avg JCT
    plot_bar_chart(
        data, 
        metric_suffix="_avg", 
        title="Average JCT vs Number of Tenants", 
        ylabel="Avg JCT (s)", 
        filename="avg_jct_comparison.png"
    )
