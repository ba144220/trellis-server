import os
os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import tempfile
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
import uvicorn

# Global pipeline instance (loaded once at startup)
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown"""
    global pipeline
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    print("Pipeline loaded successfully!")
    yield
    # Cleanup
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
            "POST /convert": "Convert PNG image to GLB 3D model"
        }
    }

@app.post("/convert")
async def convert_image_to_3d(
    image: UploadFile = File(...),
    seed: int = 1,
    simplify: float = 0.95,
    texture_size: int = 1024
):
    """
    Convert a PNG image to a 3D GLB model.
    
    Parameters:
    - image: PNG image file to convert
    - seed: Random seed for reproducibility (default: 1)
    - simplify: Ratio of triangles to remove in simplification (default: 0.95)
    - texture_size: Size of texture for the GLB (default: 1024)
    
    Returns:
    - GLB file
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read the uploaded image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Run the pipeline
        print(f"Processing image: {image.filename}")
        outputs = pipeline.run(
            pil_image,
            seed=seed,
        )
        
        # Convert to GLB
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=simplify,
            texture_size=texture_size,
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tmp_file:
            tmp_path = tmp_file.name
            glb.export(tmp_path)
        
        # Cleanup
        del outputs
        del glb
        torch.cuda.empty_cache()
        
        # Return the GLB file
        return FileResponse(
            tmp_path,
            media_type="model/gltf-binary",
            filename=f"{os.path.splitext(image.filename)[0]}.glb",
            background=None
        )
        
    except Exception as e:
        # Cleanup on error
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

