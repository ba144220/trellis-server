import os
os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import io
import tempfile
import time
import torch
import uuid
from contextlib import asynccontextmanager
from threading import Thread, Lock
from queue import Queue, Empty
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
import uvicorn

# Global pipeline instance and worker infrastructure
pipeline = None
inference_queue = None
postproc_queue = None
result_dict = {}  # Maps request_id -> result
result_dict_lock = Lock()  # Thread-safe access to result_dict
workers_started = False

# Start workers
NUM_INFERENCE_WORKER = 2  # Usually 1 is best since GPU is serialized, but configurable for testing
NUM_POSTPROC_WORKERS = 5

def postprocessing_worker(postproc_queue, worker_id):
    """
    Persistent worker thread that handles post-processing.
    Processes items from the queue and stores results.
    """
    print(f"[PostProc-{worker_id}] Worker started")
    
    while True:
        try:
            item = postproc_queue.get(timeout=1)
            if item is None:
                postproc_queue.task_done()
                break
            
            request_id, gaussian, mesh, simplify, texture_size, inference_time = item
            print(f"[PostProc-{worker_id}] Processing request {request_id}")
            
            try:
                postproc_start = time.time()
                glb = postprocessing_utils.to_glb(
                    gaussian,
                    mesh,
                    simplify=simplify,
                    texture_size=texture_size,
                    verbose=False,
                )
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tmp_file:
                    tmp_path = tmp_file.name
                    glb.export(tmp_path)
                
                postproc_time = time.time() - postproc_start
                
                # Store result (thread-safe)
                with result_dict_lock:
                    result_dict[request_id] = {
                        'status': 'success',
                        'tmp_path': tmp_path,
                        'inference_time': inference_time,
                        'postproc_time': postproc_time
                    }
                
                print(f"[PostProc-{worker_id}] Completed request {request_id} in {postproc_time:.2f}s")
                
                # del gaussian, mesh, glb
                # torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"[PostProc-{worker_id}] Error processing request {request_id}: {e}")
                with result_dict_lock:
                    result_dict[request_id] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            postproc_queue.task_done()
            
        except Empty:
            continue
        except Exception as e:
            print(f"[PostProc-{worker_id}] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[PostProc-{worker_id}] Worker stopped")


def inference_worker(inference_queue, postproc_queue, pipeline, worker_id=1):
    """
    Persistent worker that handles GPU generation.
    Sends results to post-processing queue for parallel processing.
    """
    print(f"[Inference-Worker-{worker_id}] Worker started")
    
    while True:
        try:
            item = inference_queue.get(timeout=1)
            if item is None:
                inference_queue.task_done()
                break
            
            request_id, pil_image, seed, simplify, texture_size = item
            print(f"[Inference-Worker-{worker_id}] Processing request {request_id}")
            
            try:
                inference_start = time.time()
                outputs = pipeline.run(pil_image, seed=seed)
                inference_time = time.time() - inference_start
                
                print(f"[Inference-Worker-{worker_id}] Inference for {request_id} completed in {inference_time:.2f}s")
                
                # Send to post-processing queue
                gaussian = outputs['gaussian'][0]
                mesh = outputs['mesh'][0]
                
                postproc_queue.put((request_id, gaussian, mesh, simplify, texture_size, inference_time))
                
                # Cleanup GPU memory
                del outputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"[Inference-Worker-{worker_id}] Error processing request {request_id}: {e}")
                with result_dict_lock:
                    result_dict[request_id] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            inference_queue.task_done()
            
        except Empty:
            continue
        except Exception as e:
            print(f"[Inference-Worker-{worker_id}] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[Inference-Worker-{worker_id}] Worker stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, start workers, and cleanup on shutdown"""
    global pipeline, inference_queue, postproc_queue, workers_started
    
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    print("Pipeline loaded successfully!")
    
    # Setup queues
    inference_queue = Queue()
    postproc_queue = Queue()
    
    
    inference_workers = []
    postproc_workers = []
    
    # Start GPU workers
    for i in range(NUM_INFERENCE_WORKER):
        worker = Thread(
            target=inference_worker,
            args=(inference_queue, postproc_queue, pipeline, i + 1),
            daemon=True,
            name=f"Inference-Worker-{i+1}"
        )
        worker.start()
        inference_workers.append(worker)
    
    # Start post-processing workers
    for i in range(NUM_POSTPROC_WORKERS):
        worker = Thread(
            target=postprocessing_worker,
            args=(postproc_queue, i + 1),
            daemon=True,
            name=f"PostProc-{i+1}"
        )
        worker.start()
        postproc_workers.append(worker)
    
    workers_started = True
    print(f"Workers started successfully! ({NUM_INFERENCE_WORKER} Inference, {NUM_POSTPROC_WORKERS} PostProc)")
    
    yield
    
    # Cleanup
    print("Shutting down workers...")
    for _ in range(NUM_INFERENCE_WORKER):
        inference_queue.put(None)
    for _ in range(NUM_POSTPROC_WORKERS):
        postproc_queue.put(None)
    
    if pipeline is not None:
        del pipeline
        torch.cuda.empty_cache()

app = FastAPI(title="TRELLIS Image to 3D API", lifespan=lifespan)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "TRELLIS Image to 3D API is running",
        "endpoints": {
            "POST /convert/baseline": "Convert using baseline approach",
            "POST /convert/optimized": "Convert using optimized approach with parallel post-processing"
        }
    }

@app.post("/convert/baseline")
async def convert_baseline(
    image: UploadFile = File(...),
    seed: int = 1,
    simplify: float = 0.95,
    texture_size: int = 1024
):
    """
    Convert using baseline approach (impl/baseline.py logic).
    Sequential: GPU generation -> CPU post-processing
    
    Parameters:
    - image: Image file to convert
    - seed: Random seed for reproducibility (default: 1)
    - simplify: Ratio of triangles to remove in simplification (default: 0.95)
    - texture_size: Size of texture for the GLB (default: 1024)
    
    Returns:
    - GLB file with timing metadata in headers
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        start_time = time.time()
        
        # Read the uploaded image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        print(f"[BASELINE] Processing image: {image.filename}")
        
        # GPU generation
        inference_start = time.time()
        outputs = pipeline.run(pil_image, seed=seed)
        inference_time = time.time() - inference_start
        
        # CPU post-processing
        postproc_start = time.time()
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=simplify,
            texture_size=texture_size,
        )
        postproc_time = time.time() - postproc_start
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tmp_file:
            tmp_path = tmp_file.name
            glb.export(tmp_path)
        
        total_time = time.time() - start_time
        
        # Cleanup
        del outputs, glb
        torch.cuda.empty_cache()
        
        print(f"[BASELINE] Total: {total_time:.2f}s (Inference: {inference_time:.2f}s, PostProc: {postproc_time:.2f}s)")
        
        # Return with timing headers
        return FileResponse(
            tmp_path,
            media_type="model/gltf-binary",
            filename=f"{os.path.splitext(image.filename)[0]}_baseline.glb",
            headers={
                "X-Total-Time": str(total_time),
                "X-Inference-Time": str(inference_time),
                "X-PostProc-Time": str(postproc_time),
                "X-Method": "baseline"
            }
        )
        
    except Exception as e:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/convert/optimized")
async def convert_optimized(
    image: UploadFile = File(...),
    seed: int = 1,
    simplify: float = 0.95,
    texture_size: int = 1024
):
    """
    Convert using optimized approach (impl/optimized_v2.py logic).
    Uses persistent worker threads to process requests through queues.
    GPU and post-processing workers run in parallel for better throughput.
    
    Parameters:
    - image: Image file to convert
    - seed: Random seed for reproducibility (default: 1)
    - simplify: Ratio of triangles to remove in simplification (default: 0.95)
    - texture_size: Size of texture for the GLB (default: 1024)
    
    Returns:
    - GLB file with timing metadata in headers
    """
    if not workers_started or pipeline is None:
        raise HTTPException(status_code=503, detail="Workers not ready yet")
    
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        start_time = time.time()
        
        # Read the uploaded image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        print(f"[OPTIMIZED] Request {request_id}: Processing image {image.filename}")
        
        # Add to inference queue
        inference_queue.put((request_id, pil_image, seed, simplify, texture_size))
        
        # Wait for result (poll result_dict)
        timeout = 120  # 2 minute timeout
        poll_interval = 0.1  # Check every 100ms
        elapsed = 0
        
        while elapsed < timeout:
            # Check for result (thread-safe)
            result = None
            with result_dict_lock:
                if request_id in result_dict:
                    result = result_dict.pop(request_id)
            
            if result is not None:
                if result['status'] == 'error':
                    raise Exception(result['error'])
                
                total_time = time.time() - start_time
                inference_time = result['inference_time']
                postproc_time = result['postproc_time']
                
                print(f"[OPTIMIZED] Request {request_id}: Total {total_time:.2f}s (Inference: {inference_time:.2f}s, PostProc: {postproc_time:.2f}s)")
                
                # Return with timing headers
                return FileResponse(
                    result['tmp_path'],
                    media_type="model/gltf-binary",
                    filename=f"{os.path.splitext(image.filename)[0]}_optimized.glb",
                    headers={
                        "X-Total-Time": str(total_time),
                        "X-Inference-Time": str(inference_time),
                        "X-PostProc-Time": str(postproc_time),
                        "X-Method": "optimized",
                        "X-Request-ID": request_id
                    }
                )
            
            # Use async sleep to allow other requests to be processed concurrently
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        # Timeout
        raise HTTPException(status_code=504, detail=f"Request {request_id} timed out")
        
    except HTTPException:
        raise
    except Exception as e:
        try:
            torch.cuda.empty_cache()
        except:
            pass  # Ignore errors when trying to clear cache after an error
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

