import os
os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'

# Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.
os.environ['SPCONV_ALGO'] = 'native'        
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Internal dependencies
import logging
import time
import torch
import json
from datetime import datetime
from threading import Thread
from queue import Queue

# External dependencies
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

NUM_WORKERS = 3  # Number of parallel workers

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# Clear everything in the output directory
for file in os.listdir(OUTPUT_DIR):
    os.remove(os.path.join(OUTPUT_DIR, file))

if not os.path.exists(TIME_LOG_FILE):
    with open(TIME_LOG_FILE, "w") as f:
        json.dump({}, f)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def process_images(pipeline, image_queue, worker_id):
    """Worker function to process images from a queue"""
    while True:
        item = image_queue.get()
        if item is None:  # Sentinel value to stop the worker
            image_queue.task_done()
            break
        
        image_path = item
        logging.info(f"Worker {worker_id} processing: {image_path}")
        
        try:
            image = Image.open(image_path)
            
            # Run the pipeline
            outputs = pipeline.run(
                image,
                seed=1,
            )
            
            # GLB files can be extracted from the outputs
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                # Optional parameters
                simplify=0.95,          # Ratio of triangles to remove in the simplification process
                texture_size=1024,      # Size of the texture used for the GLB
            )
            glb.export(os.path.join(OUTPUT_DIR, f"{image_path.split('/')[-1].split('.')[0]}.glb"))
            
            del outputs
            del glb
            torch.cuda.empty_cache()
            
            logging.info(f"Worker {worker_id} completed: {image_path}")
        except Exception as e:
            logging.error(f"Worker {worker_id} error processing {image_path}: {e}")
        finally:
            image_queue.task_done()

def main():
    """main function to run the pipeline"""
    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    logging.info(f"Starting to run the pipeline with {NUM_WORKERS} parallel workers")
    start_time = time.time()

    # Create a queue and populate it with image paths
    image_queue = Queue()
    for image_path in IMAGE_PATHS:
        image_queue.put(image_path)
    
    # Add sentinel values to stop workers (one per worker)
    for _ in range(NUM_WORKERS):
        image_queue.put(None)
    
    # Create and start worker threads
    workers = []
    for i in range(NUM_WORKERS):
        worker = Thread(target=process_images, args=(pipeline, image_queue, i + 1))
        worker.start()
        workers.append(worker)
    
    # Wait for all tasks to complete
    image_queue.join()
    
    # Wait for all worker threads to finish
    for worker in workers:
        worker.join()

    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time} seconds")

    run_metadata = {
        "time_taken": end_time - start_time,
        "image_paths": IMAGE_PATHS,
        "timestamp": datetime.now().isoformat(),
    }

    # Load the best time from the log file
    with open(TIME_LOG_FILE, "r") as f:
        time_log = json.load(f)
    
    # If we beat the best time, append the new record to the log file
    if len(time_log) == 0 or run_metadata["time_taken"] < time_log[-1]["time_taken"]:
        time_log.append(run_metadata)
        with open(TIME_LOG_FILE, "w") as f:
            json.dump(time_log, f, indent=2)
    else:
        print(f"Did not beat the best time. Best time: {time_log[-1]['time_taken']} seconds. New time: {run_metadata['time_taken']} seconds.")
    

if __name__ == "__main__":
    main()