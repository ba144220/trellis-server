import os
os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'

# Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.
os.environ['SPCONV_ALGO'] = 'native'        
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal dependencies
import time
import torch
import json
from datetime import datetime

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

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(TIME_LOG_FILE):
    with open(TIME_LOG_FILE, "w") as f:
        json.dump({}, f)

def main():
    """main function to run the pipeline"""
    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    print(f"Starting to run the pipeline")
    start_time = time.time()

    for image_path in IMAGE_PATHS:
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

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

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