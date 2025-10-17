"""
Generate visualization plots from profiling data
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

def plot_profiling_results(json_file, output_file=None):
    """
    Create a simple step plot showing GPU metrics over time
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    stages = data['stages']
    total_time = data['total_time_seconds']
    
    # Get total GPU memory from nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits', '--id=1'],
            capture_output=True, text=True, timeout=2
        )
        total_gpu_memory_mb = float(result.stdout.strip())
    except:
        # Default to 48GB if query fails
        total_gpu_memory_mb = 48 * 1024
    
    # Create figure with 2 subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(hspace=0.2)
    
    # Build time series data for step plots
    time_points = []
    gpu_util_values = []
    gpu_mem_values = []
    gpu_mem_percent_values = []
    stage_boundaries = [0]
    stage_labels = []
    
    print(f"Total GPU Memory: {total_gpu_memory_mb:.0f} MB ({total_gpu_memory_mb/1024:.1f} GB)")
    
    # Get baseline memory (memory before loading model) for reference line
    baseline_memory_mb = data.get('baseline_memory_mb', 0)
    
    # Get model weights memory (first stage's memory)
    model_weights_mb = stages[0]['gpu_memory_peak_mb'] if stages else 0
    print(f"Model weights: {model_weights_mb:.0f} MB ({model_weights_mb/1024:.1f} GB)")
    
    cumulative_time = 0
    for stage in stages:
        # Use absolute memory values (including model weights)
        absolute_mem_mb = stage['gpu_memory_peak_mb']
        
        # Add start time
        time_points.append(cumulative_time)
        gpu_util_values.append(stage['gpu_utilization_mean'])
        gpu_mem_values.append(absolute_mem_mb)
        gpu_mem_percent_values.append((absolute_mem_mb / total_gpu_memory_mb) * 100)
        
        # Add end time
        cumulative_time += stage['time_seconds']
        time_points.append(cumulative_time)
        gpu_util_values.append(stage['gpu_utilization_mean'])
        gpu_mem_values.append(absolute_mem_mb)
        gpu_mem_percent_values.append((absolute_mem_mb / total_gpu_memory_mb) * 100)
        
        stage_boundaries.append(cumulative_time)
        stage_labels.append(stage['name'])
    
    # Plot 1: GPU Utilization (step plot)
    ax1.plot(time_points, gpu_util_values, 
             linewidth=2.5, color='#2ecc71', label='GPU Utilization', drawstyle='steps-post')
    ax1.fill_between(time_points, 0, gpu_util_values, 
                     step='post', alpha=0.3, color='#2ecc71')
    ax1.axhline(y=60, color='red', linestyle='--', alpha=0.4, linewidth=1, label='High threshold (60%)')
    ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.4, linewidth=1, label='Low threshold (30%)')
    ax1.set_ylabel('GPU Utilization (%)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title(f'Pipeline Profiling - Total Time: {total_time:.2f}s', 
                  fontsize=13, fontweight='bold', pad=10)
    
    # Add stage boundaries and labels
    for i, (boundary, label) in enumerate(zip(stage_boundaries[1:], stage_labels)):
        ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Plot 2: GPU Memory (step plot) with dual y-axis - showing absolute memory
    ax2.plot(time_points, gpu_mem_values, 
             linewidth=2.5, color='#9b59b6', label='GPU Memory (MB)', drawstyle='steps-post')
    ax2.fill_between(time_points, 0, gpu_mem_values, 
                     step='post', alpha=0.3, color='#9b59b6')
    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('GPU Memory (MB)', fontsize=11, fontweight='bold', color='#9b59b6')
    ax2.set_ylim(0, max(gpu_mem_values) * 1.1 if max(gpu_mem_values) > 0 else 1000)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.tick_params(axis='y', labelcolor='#9b59b6')
    
    # Add secondary y-axis for percentage
    ax2_percent = ax2.twinx()
    ax2_percent.plot(time_points, gpu_mem_percent_values, 
                    linewidth=0, alpha=0)  # Invisible line for alignment
    ax2_percent.set_ylabel('GPU Memory (%)', fontsize=11, fontweight='bold', color='#7d3c98')
    max_percent = (max(gpu_mem_values) * 1.1 / total_gpu_memory_mb) * 100 if max(gpu_mem_values) > 0 else 10
    ax2_percent.set_ylim(0, max_percent)
    ax2_percent.tick_params(axis='y', labelcolor='#7d3c98')
    
    # Add model weights reference line
    ax2.axhline(y=model_weights_mb, color='orange', linestyle='--', 
               alpha=0.6, linewidth=2, label=f'Model Weights ({model_weights_mb:.0f} MB)')
    
    # Add total memory reference line
    ax2.axhline(y=total_gpu_memory_mb, color='red', linestyle='--', 
               alpha=0.5, linewidth=1.5, label=f'Total GPU Memory ({total_gpu_memory_mb:.0f} MB)')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    ax2.legend(lines1, labels1, loc='lower right', fontsize=9)
    
    # Add stage boundaries and labels
    for i, (boundary, label) in enumerate(zip(stage_boundaries[1:], stage_labels)):
        ax2.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        # Add stage labels at boundaries
        if i < len(stage_labels):
            mid_point = (stage_boundaries[i] + stage_boundaries[i+1]) / 2
            # Only label if stage is wide enough
            stage_width = stage_boundaries[i+1] - stage_boundaries[i]
            if stage_width > cumulative_time * 0.05:
                ax2.text(mid_point, max(gpu_mem_values) * 1.05, 
                        label.replace('. ', '.\n'), 
                        ha='center', va='bottom', fontsize=9, 
                        rotation=0, style='italic', fontweight='bold')
    
    # Set x-axis limits for all plots
    ax1.set_xlim(0, cumulative_time)
    ax2.set_xlim(0, cumulative_time)
    
    # Save figure (PNG only)
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"profiling/profile_visualization_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_file}")
    plt.close()
    
    return output_file

def main():
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Find the latest profile JSON
        import glob
        json_files = glob.glob("profiling/profile_baseline_*.json")
        if not json_files:
            print("❌ No profiling JSON files found!")
            return
        json_file = max(json_files)
    
    print(f"📊 Generating visualization from: {json_file}")
    output_file = plot_profiling_results(json_file)
    print(f"\n✨ Visualization complete!")
    
    # Print summary
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n📈 Summary:")
    print(f"   Total time: {data['total_time_seconds']:.2f}s")
    print(f"   Number of stages: {len(data['stages'])}")
    
    # Find top bottleneck
    stages = data['stages']
    max_time_stage = max(stages, key=lambda x: x['time_seconds'])
    print(f"   Biggest bottleneck: {max_time_stage['name']} ({max_time_stage['time_seconds']:.2f}s)")

if __name__ == "__main__":
    main()

