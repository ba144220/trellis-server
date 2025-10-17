"""
Profiling utilities for tracking time, GPU memory, GPU utilization, and CPU usage.
"""

import time
import torch
import subprocess
import threading
import json
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import os


@dataclass
class StageMetrics:
    """Metrics for a single profiling stage"""
    name: str
    time_seconds: float
    gpu_memory_allocated_mb: float  # Current allocated memory
    gpu_memory_reserved_mb: float   # Current reserved memory
    gpu_memory_peak_mb: float       # Peak memory since last reset
    gpu_utilization_mean: float     # Mean GPU utilization during stage
    gpu_utilization_max: float      # Max GPU utilization during stage
    gpu_memory_used_mean_mb: float  # Mean GPU memory used (from nvidia-smi)
    gpu_memory_used_max_mb: float   # Max GPU memory used (from nvidia-smi)
    cpu_utilization_mean: float     # Mean CPU utilization during stage
    cpu_utilization_max: float      # Max CPU utilization during stage
    cpu_memory_used_mean_mb: float  # Mean CPU memory used (RSS)
    
    def to_dict(self):
        return asdict(self)


class ResourceMonitor:
    """Background thread to monitor GPU and CPU utilization and memory"""
    
    def __init__(self, device_id: int = 0, interval: float = 0.1):
        self.device_id = device_id
        self.interval = interval
        self.running = False
        self.thread = None
        self.gpu_utilizations: List[float] = []
        self.gpu_memory_used: List[float] = []  # In MB
        self.cpu_utilizations: List[float] = []  # Overall CPU %
        self.cpu_memory_used: List[float] = []   # Process RSS in MB
        self.pid = os.getpid()
        
    def _monitor(self):
        """Monitor GPU and CPU in background thread"""
        while self.running:
            try:
                # Query nvidia-smi for GPU utilization and memory
                result = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits',
                        f'--id={self.device_id}'
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    util, mem = output.split(',')
                    self.gpu_utilizations.append(float(util))
                    self.gpu_memory_used.append(float(mem))
            except Exception:
                pass  # Silently ignore errors
            
            try:
                # Get CPU usage via /proc/stat for overall system
                with open('/proc/stat', 'r') as f:
                    cpu_line = f.readline()
                    cpu_times = [int(x) for x in cpu_line.split()[1:]]
                    total_time = sum(cpu_times)
                    idle_time = cpu_times[3]  # idle time is 4th field
                    
                    if hasattr(self, '_last_total_time'):
                        total_delta = total_time - self._last_total_time
                        idle_delta = idle_time - self._last_idle_time
                        if total_delta > 0:
                            cpu_percent = 100.0 * (1.0 - idle_delta / total_delta)
                            self.cpu_utilizations.append(cpu_percent)
                    
                    self._last_total_time = total_time
                    self._last_idle_time = idle_time
                
                # Get process memory usage
                with open(f'/proc/{self.pid}/status', 'r') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            # VmRSS is in KB, convert to MB
                            mem_kb = int(line.split()[1])
                            self.cpu_memory_used.append(mem_kb / 1024.0)
                            break
            except Exception:
                pass  # Silently ignore errors
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring"""
        self.running = True
        self.gpu_utilizations = []
        self.gpu_memory_used = []
        self.cpu_utilizations = []
        self.cpu_memory_used = []
        # Initialize CPU tracking
        self._last_total_time = 0
        self._last_idle_time = 0
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return statistics"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        stats = {
            'gpu_utilization_mean': 0.0,
            'gpu_utilization_max': 0.0,
            'gpu_memory_used_mean_mb': 0.0,
            'gpu_memory_used_max_mb': 0.0,
            'cpu_utilization_mean': 0.0,
            'cpu_utilization_max': 0.0,
            'cpu_memory_used_mean_mb': 0.0
        }
        
        if len(self.gpu_utilizations) > 0:
            stats['gpu_utilization_mean'] = sum(self.gpu_utilizations) / len(self.gpu_utilizations)
            stats['gpu_utilization_max'] = max(self.gpu_utilizations)
            stats['gpu_memory_used_mean_mb'] = sum(self.gpu_memory_used) / len(self.gpu_memory_used)
            stats['gpu_memory_used_max_mb'] = max(self.gpu_memory_used)
        
        if len(self.cpu_utilizations) > 0:
            stats['cpu_utilization_mean'] = sum(self.cpu_utilizations) / len(self.cpu_utilizations)
            stats['cpu_utilization_max'] = max(self.cpu_utilizations)
        
        if len(self.cpu_memory_used) > 0:
            stats['cpu_memory_used_mean_mb'] = sum(self.cpu_memory_used) / len(self.cpu_memory_used)
        
        return stats


class Profiler:
    """Main profiler class"""
    
    def __init__(self, device_id: int = 0, nvidia_smi_device_id: int = None):
        """
        Initialize profiler.
        
        Args:
            device_id: CUDA device ID as seen by PyTorch (affected by CUDA_VISIBLE_DEVICES)
            nvidia_smi_device_id: Actual GPU ID for nvidia-smi (not affected by CUDA_VISIBLE_DEVICES).
                                  If None, uses device_id.
        """
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        self.stages: List[StageMetrics] = []
        
        # Use nvidia_smi_device_id for monitoring if provided, otherwise use device_id
        monitor_device_id = nvidia_smi_device_id if nvidia_smi_device_id is not None else device_id
        self.monitor = ResourceMonitor(device_id=monitor_device_id)
        
        # Initialize CUDA by setting device and doing a dummy operation
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            # Dummy operation to initialize CUDA
            _ = torch.zeros(1).to(self.device)
        
    def reset_peak_memory(self):
        """Reset peak memory statistics"""
        torch.cuda.reset_peak_memory_stats(self.device)
    
    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager to profile a single stage"""
        print(f"\n{'='*60}")
        print(f"Profiling stage: {stage_name}")
        print(f"{'='*60}")
        
        # Reset peak memory tracking
        self.reset_peak_memory()
        
        # Start GPU monitoring
        self.monitor.start()
        
        # Record start metrics
        start_time = time.time()
        torch.cuda.synchronize(self.device)
        
        try:
            yield
        finally:
            # Ensure all CUDA operations are complete
            torch.cuda.synchronize(self.device)
            
            # Record end metrics
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Stop GPU monitoring and get stats
            monitor_stats = self.monitor.stop()
            
            # Get memory metrics
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)    # MB
            memory_peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)   # MB
            
            # Create metrics object
            metrics = StageMetrics(
                name=stage_name,
                time_seconds=elapsed_time,
                gpu_memory_allocated_mb=memory_allocated,
                gpu_memory_reserved_mb=memory_reserved,
                gpu_memory_peak_mb=memory_peak,
                gpu_utilization_mean=monitor_stats['gpu_utilization_mean'],
                gpu_utilization_max=monitor_stats['gpu_utilization_max'],
                gpu_memory_used_mean_mb=monitor_stats['gpu_memory_used_mean_mb'],
                gpu_memory_used_max_mb=monitor_stats['gpu_memory_used_max_mb'],
                cpu_utilization_mean=monitor_stats['cpu_utilization_mean'],
                cpu_utilization_max=monitor_stats['cpu_utilization_max'],
                cpu_memory_used_mean_mb=monitor_stats['cpu_memory_used_mean_mb']
            )
            
            self.stages.append(metrics)
            
            # Print immediate feedback
            print(f"\n{'-'*60}")
            print(f"Stage: {stage_name}")
            print(f"  Time: {elapsed_time:.2f}s")
            print(f"  GPU Memory (torch):")
            print(f"    - Allocated: {memory_allocated:.2f} MB")
            print(f"    - Reserved: {memory_reserved:.2f} MB")
            print(f"    - Peak: {memory_peak:.2f} MB")
            print(f"  GPU Utilization:")
            print(f"    - Mean: {monitor_stats['gpu_utilization_mean']:.1f}%")
            print(f"    - Max: {monitor_stats['gpu_utilization_max']:.1f}%")
            print(f"  GPU Memory (nvidia-smi):")
            print(f"    - Mean: {monitor_stats['gpu_memory_used_mean_mb']:.2f} MB")
            print(f"    - Max: {monitor_stats['gpu_memory_used_max_mb']:.2f} MB")
            print(f"  CPU Utilization:")
            print(f"    - Mean: {monitor_stats['cpu_utilization_mean']:.1f}%")
            print(f"    - Max: {monitor_stats['cpu_utilization_max']:.1f}%")
            print(f"  CPU Memory (Process RSS):")
            print(f"    - Mean: {monitor_stats['cpu_memory_used_mean_mb']:.2f} MB")
            print(f"{'-'*60}")
    
    def get_summary(self) -> Dict:
        """Get summary of all profiled stages"""
        total_time = sum(stage.time_seconds for stage in self.stages)
        
        summary = {
            'total_time_seconds': total_time,
            'stages': [stage.to_dict() for stage in self.stages],
            'breakdown_percent': {
                stage.name: (stage.time_seconds / total_time * 100) 
                for stage in self.stages
            }
        }
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary table"""
        if not self.stages:
            print("No profiling data available")
            return
        
        total_time = sum(stage.time_seconds for stage in self.stages)
        
        print(f"\n{'='*100}")
        print(f"PROFILING SUMMARY")
        print(f"{'='*100}")
        print(f"\nTotal Time: {total_time:.2f} seconds\n")
        
        # Table header
        print(f"{'Stage':<30} {'Time(s)':<10} {'%':<8} {'Peak Mem(MB)':<15} {'GPU%':<10} {'CPU%':<10} {'Type':<12}")
        print(f"{'-'*100}")
        
        # Table rows
        for stage in self.stages:
            percent = (stage.time_seconds / total_time * 100)
            
            # Determine bottleneck type
            gpu_util = stage.gpu_utilization_mean
            cpu_util = stage.cpu_utilization_mean
            if gpu_util > 60 and gpu_util > cpu_util:
                bottleneck_type = "GPU-bound"
            elif cpu_util > 60 and cpu_util > gpu_util:
                bottleneck_type = "CPU-bound"
            elif gpu_util > 30 and cpu_util > 30:
                bottleneck_type = "Mixed"
            elif gpu_util < 30 and cpu_util < 30:
                bottleneck_type = "I/O-bound?"
            else:
                bottleneck_type = "Unknown"
            
            print(
                f"{stage.name:<30} "
                f"{stage.time_seconds:<10.2f} "
                f"{percent:<8.1f} "
                f"{stage.gpu_memory_peak_mb:<15.1f} "
                f"{gpu_util:<10.1f} "
                f"{cpu_util:<10.1f} "
                f"{bottleneck_type:<12}"
            )
        
        print(f"{'-'*100}")
        print(f"{'TOTAL':<30} {total_time:<10.2f} {100.0:<8.1f}")
        print(f"{'='*100}\n")
    
    def save_to_json(self, filepath: str):
        """Save profiling results to JSON file"""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Profiling results saved to: {filepath}")

