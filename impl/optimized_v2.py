import os
os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import torch
import json
from datetime import datetime
from threading import Thread
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

IMAGE_PATHS = [
    "assets/example_image/T.png",
    "assets/example_image/typical_building_building.png",
    "assets/example_image/typical_vehicle_biplane.png",
    "assets/example_image/typical_vehicle_helicopter.png",
    "assets/example_image/typical_vehicle_pirate_ship.png",
    "assets/example_image/typical_misc_gate.png",
]

OUTPUT_DIR = "outputs"
TIME_LOG_FILE = "time_log.json"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# Clear everything in the output directory
for file in os.listdir(OUTPUT_DIR):
    os.remove(os.path.join(OUTPUT_DIR, file))

if not os.path.exists(TIME_LOG_FILE):
    with open(TIME_LOG_FILE, "w") as f:
        json.dump({}, f)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(message)s')


def postprocessing_worker(postproc_queue, output_queue, worker_id):
    """
    Worker thread that handles post-processing.
    Uses threading to allow GPU and CPU work to overlap.
    """
    logging.info(f"Post-processing worker {worker_id} started")
    
    while True:
        try:
            item = postproc_queue.get(timeout=1)
            if item is None:
                postproc_queue.task_done()
                break
            
            image_path, gaussian, mesh = item
            logging.info(f"Worker {worker_id} post-processing: {image_path}")
            
            start_time = time.time()
            
            # Post-processing: mesh simplification, UV mapping, texture baking
            # This is CPU-bound work that can run while GPU processes next image
            glb = postprocessing_utils.to_glb(
                gaussian,
                mesh,
                simplify=0.95,
                texture_size=1024,
                verbose=False,
            )
            
            output_path = os.path.join(OUTPUT_DIR, f"{image_path.split('/')[-1].split('.')[0]}.glb")
            glb.export(output_path)
            
            elapsed = time.time() - start_time
            logging.info(f"Worker {worker_id} completed {image_path} in {elapsed:.2f}s")
            
            output_queue.put((image_path, output_path, elapsed))
            postproc_queue.task_done()
            
            # Cleanup
            del gaussian, mesh, glb
            torch.cuda.empty_cache()
            
        except Empty:
            continue
        except Exception as e:
            logging.error(f"Worker {worker_id} error: {e}")
            import traceback
            traceback.print_exc()
            postproc_queue.task_done()
    
    logging.info(f"Post-processing worker {worker_id} stopped")


def gpu_generation_worker(image_queue, postproc_queue, pipeline):
    """
    Worker that handles GPU generation (stages 1-5).
    Sends results to post-processing queue for parallel processing.
    """
    logging.info("GPU generation worker started")
    
    while True:
        item = image_queue.get()
        if item is None:
            image_queue.task_done()
            break
        
        image_path = item
        logging.info(f"GPU worker processing: {image_path}")
        
        try:
            start_time = time.time()
            
            image = Image.open(image_path)
            
            # Run core pipeline (GPU-bound stages 1-5)
            outputs = pipeline.run(
                image,
                seed=1,
            )
            
            elapsed = time.time() - start_time
            logging.info(f"GPU generation for {image_path} completed in {elapsed:.2f}s")
            
            # Send to post-processing queue
            # While post-processing happens, GPU can work on next image
            gaussian = outputs['gaussian'][0]
            mesh = outputs['mesh'][0]
            
            postproc_queue.put((image_path, gaussian, mesh))
            
            # Cleanup GPU memory for next image
            del outputs
            torch.cuda.empty_cache()
            
            image_queue.task_done()
            
        except Exception as e:
            logging.error(f"GPU worker error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            image_queue.task_done()
    
    logging.info("GPU generation worker stopped")


def main():
    """Main function implementing pipeline parallelism with threading"""
    
    logging.info("=" * 80)
    logging.info("OPTIMIZED PIPELINE V2: Pipeline Parallelism (GPU + CPU overlap)")
    logging.info("=" * 80)
    
    # Load pipeline
    logging.info("Loading pipeline...")
    load_start = time.time()
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    load_time = time.time() - load_start
    logging.info(f"Pipeline loaded in {load_time:.2f}s")
    
    # Setup queues
    image_queue = Queue()
    postproc_queue = Queue()
    output_queue = Queue()
    
    # Populate image queue
    for image_path in IMAGE_PATHS:
        image_queue.put(image_path)
    image_queue.put(None)  # Sentinel to stop GPU worker
    
    # Start post-processing workers (run in separate threads)
    # While GIL limits CPU parallelism, the key benefit is GPU/CPU overlap
    NUM_POSTPROC_WORKERS = 2  # Multiple workers to handle post-processing queue
    postproc_workers = []
    for i in range(NUM_POSTPROC_WORKERS):
        worker = Thread(
            target=postprocessing_worker,
            args=(postproc_queue, output_queue, i + 1),
            name=f"PostProc-{i+1}"
        )
        worker.daemon = True
        worker.start()
        postproc_workers.append(worker)
    
    # Start GPU generation worker in separate thread
    logging.info("Starting GPU generation and post-processing pipeline...")
    start_time = time.time()
    
    gpu_worker = Thread(
        target=gpu_generation_worker,
        args=(image_queue, postproc_queue, pipeline),
        name="GPU-Worker"
    )
    gpu_worker.daemon = True
    gpu_worker.start()
    
    # Wait for GPU generation to complete
    image_queue.join()
    logging.info("GPU generation completed, waiting for post-processing...")
    
    # Signal post-processing workers to stop after all items are processed
    postproc_queue.join()
    for _ in range(NUM_POSTPROC_WORKERS):
        postproc_queue.put(None)
    
    # Collect results
    results = []
    while len(results) < len(IMAGE_PATHS):
        try:
            result = output_queue.get(timeout=1)
            results.append(result)
        except Empty:
            continue
    
    # Wait for workers to finish
    for worker in postproc_workers:
        worker.join(timeout=2)
    gpu_worker.join(timeout=2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logging.info("=" * 80)
    logging.info(f"COMPLETED: Total time: {total_time:.2f}s")
    logging.info(f"Average per image: {total_time / len(IMAGE_PATHS):.2f}s")
    logging.info("=" * 80)
    
    # Log results
    for image_path, output_path, postproc_time in results:
        logging.info(f"  {image_path} -> {output_path} (postproc: {postproc_time:.2f}s)")
    
    # Save to time log
    run_metadata = {
        "time_taken": total_time,
        "image_paths": IMAGE_PATHS,
        "timestamp": datetime.now().isoformat(),
        "method": "optimized_v2_pipeline_parallelism_threading",
        "num_postproc_workers": NUM_POSTPROC_WORKERS,
    }
    
    with open(TIME_LOG_FILE, "r") as f:
        time_log = json.load(f)
    
    if not isinstance(time_log, list):
        time_log = []
    
    time_log.append(run_metadata)
    
    with open(TIME_LOG_FILE, "w") as f:
        json.dump(time_log, f, indent=2)
    
    logging.info(f"Results saved to {TIME_LOG_FILE}")


if __name__ == "__main__":
    main()

