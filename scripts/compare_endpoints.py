#!/usr/bin/env python3
"""
Script to compare baseline vs optimized endpoints with multiple images
Sends requests with 1 second intervals without waiting for completion
"""

import requests
import time
import argparse
from pathlib import Path
from threading import Thread
from queue import Queue

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
IMAGE_DIR = PROJECT_DIR / "assets/example_image"
OUTPUT_DIR = PROJECT_DIR / "outputs"
API_BASE = "http://localhost:8000"

# Select 10 images to test
IMAGE_PATHS = [
    IMAGE_DIR / "T.png",
    IMAGE_DIR / "typical_building_building.png",
    IMAGE_DIR / "typical_vehicle_biplane.png",
    IMAGE_DIR / "typical_vehicle_helicopter.png",
    IMAGE_DIR / "typical_vehicle_pirate_ship.png",
    # IMAGE_DIR / "typical_misc_gate.png",
    # IMAGE_DIR / "typical_creature_dragon.png",
    # IMAGE_DIR / "typical_humanoid_goblin.png",
    # IMAGE_DIR / "typical_misc_lantern.png",
    # IMAGE_DIR / "typical_building_castle.png",
]

OUTPUT_DIR.mkdir(exist_ok=True)

def test_endpoint_worker(url, image_path, output_file, result_queue, request_time):
    """Worker thread to test a single endpoint"""
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {
                "seed": 1,
                "simplify": 0.95,
                "texture_size": 1024
            }
            
            start = time.time()
            response = requests.post(url, files=files, data=data)
            elapsed = time.time() - start
        
        if response.status_code == 200:
            # Save the GLB file
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            # Extract timing from headers
            headers = response.headers
            total_time = float(headers.get("X-Total-Time", 0))
            gpu_time = float(headers.get("X-GPU-Time", 0))
            postproc_time = float(headers.get("X-PostProc-Time", 0))
            method = headers.get("X-Method", "unknown")
            
            print(f"  [{image_path.name}] ✓ {total_time:.2f}s (GPU: {gpu_time:.2f}s, PostProc: {postproc_time:.2f}s, Request: {elapsed:.2f}s)")
            
            result_queue.put({
                "success": True,
                "total": total_time,
                "gpu": gpu_time,
                "postproc": postproc_time,
                "request": elapsed,
                "method": method,
                "image": image_path.name,
                "request_time": request_time
            })
        else:
            print(f"  [{image_path.name}] ✗ FAILED (Status: {response.status_code})")
            result_queue.put({"success": False, "image": image_path.name})
    except Exception as e:
        print(f"  [{image_path.name}] ✗ ERROR: {e}")
        result_queue.put({"success": False, "image": image_path.name, "error": str(e)})

def test_endpoint(endpoint_name, endpoint_url):
    """Test a single endpoint with all images"""
    print(f"\n{'='*70}")
    print(f"{endpoint_name.upper()} ENDPOINT - Sending requests...")
    print("="*70)
    
    result_queue = Queue()
    threads = []
    start_time = time.time()
    
    for i, img_path in enumerate(IMAGE_PATHS):
        output_name = f"{img_path.stem}_{endpoint_name.lower()}.glb"
        request_time = time.time()
        
        thread = Thread(
            target=test_endpoint_worker,
            args=(
                endpoint_url,
                img_path,
                OUTPUT_DIR / output_name,
                result_queue,
                request_time
            ),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        
        print(f"  [{img_path.name}] Request sent at T+{time.time() - start_time:.1f}s")
        
        # Wait 1 second before sending next request (except for last one)
        if i < len(IMAGE_PATHS) - 1:
            time.sleep(1.0)
    
    print(f"\nAll {endpoint_name.lower()} requests sent. Waiting for completion...")
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    print(f"{endpoint_name} completed in {total_time:.2f}s")
    
    return results, total_time

def main():
    parser = argparse.ArgumentParser(
        description="Test TRELLIS API endpoints with multiple images"
    )
    parser.add_argument(
        "-e", "--endpoint",
        choices=["baseline", "optimized", "both"],
        default="both",
        help="Which endpoint to test (default: both)"
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    if args.endpoint == "both":
        print("TRELLIS API Endpoint Comparison - Testing Both Endpoints")
    else:
        print(f"TRELLIS API Endpoint Test - {args.endpoint.upper()}")
    print("="*70)
    print(f"Testing {len(IMAGE_PATHS)} images - sending with 1s intervals")
    print("Requests sent without waiting for completion (concurrent)")
    
    baseline_results = None
    baseline_total_time = None
    optimized_results = None
    optimized_total_time = None
    
    # Test endpoints based on user selection
    if args.endpoint in ["baseline", "both"]:
        baseline_results, baseline_total_time = test_endpoint(
            "baseline", 
            f"{API_BASE}/convert/baseline"
        )
    
    if args.endpoint in ["optimized", "both"]:
        optimized_results, optimized_total_time = test_endpoint(
            "optimized",
            f"{API_BASE}/convert/optimized"
        )
    
    # Statistics
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print("="*70)
    
    baseline_success = [r for r in baseline_results if r["success"]] if baseline_results else []
    optimized_success = [r for r in optimized_results if r["success"]] if optimized_results else []
    
    # Show individual endpoint stats
    if baseline_results:
        print(f"\nBaseline endpoint:")
        print(f"  Successful conversions: {len(baseline_success)}/{len(baseline_results)}")
        if baseline_success:
            baseline_avg = sum(r["total"] for r in baseline_success) / len(baseline_success)
            baseline_gpu_avg = sum(r["gpu"] for r in baseline_success) / len(baseline_success)
            baseline_postproc_avg = sum(r["postproc"] for r in baseline_success) / len(baseline_success)
            print(f"  Average time per image: {baseline_avg:.2f}s")
            print(f"    GPU: {baseline_gpu_avg:.2f}s, PostProc: {baseline_postproc_avg:.2f}s")
            print(f"  Total wall-clock time: {baseline_total_time:.2f}s")
    
    if optimized_results:
        print(f"\nOptimized endpoint:")
        print(f"  Successful conversions: {len(optimized_success)}/{len(optimized_results)}")
        if optimized_success:
            optimized_avg = sum(r["total"] for r in optimized_success) / len(optimized_success)
            optimized_gpu_avg = sum(r["gpu"] for r in optimized_success) / len(optimized_success)
            optimized_postproc_avg = sum(r["postproc"] for r in optimized_success) / len(optimized_success)
            print(f"  Average time per image: {optimized_avg:.2f}s")
            print(f"    GPU: {optimized_gpu_avg:.2f}s, PostProc: {optimized_postproc_avg:.2f}s")
            print(f"  Total wall-clock time: {optimized_total_time:.2f}s")
    
    # Show comparison only if both were tested
    if baseline_success and optimized_success:
        # Comparison
        print(f"\n{'='*70}")
        print("COMPARISON")
        print("="*70)
        
        # Recalculate averages for comparison
        baseline_avg = sum(r["total"] for r in baseline_success) / len(baseline_success)
        optimized_avg = sum(r["total"] for r in optimized_success) / len(optimized_success)
        
        diff = baseline_avg - optimized_avg
        speedup = baseline_avg / optimized_avg if optimized_avg > 0 else 0
        
        print(f"\nPer-image average:")
        print(f"  Difference: {diff:.2f}s")
        print(f"  Speedup:    {speedup:.2f}x")
        
        if diff > 0.1:
            print(f"\n✓ Optimized is {diff:.2f}s faster per image on average!")
        elif diff < -0.1:
            print(f"\n✗ Baseline is {abs(diff):.2f}s faster per image on average")
        else:
            print(f"\n≈ Similar performance (difference < 0.1s)")
        
        total_diff = baseline_total_time - optimized_total_time
        print(f"\nTotal time saved: {total_diff:.2f}s for {len(IMAGE_PATHS)} images")
        
        if total_diff > 0:
            print(f"✓ Optimized handles concurrent load {total_diff:.2f}s faster overall!")
        elif total_diff < 0:
            print(f"✗ Baseline handles concurrent load {abs(total_diff):.2f}s faster overall")
        else:
            print(f"≈ Similar concurrent performance")
    elif baseline_results and optimized_results:
        # Both were tested but one failed
        print("\n✗ Cannot compare - some conversions failed")
        if baseline_results:
            print(f"  Baseline successful: {len(baseline_success)}/{len(baseline_results)}")
        if optimized_results:
            print(f"  Optimized successful: {len(optimized_success)}/{len(optimized_results)}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()

