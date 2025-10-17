"""
Detailed profiling script for the baseline image-to-3D pipeline.

This script profiles each stage of the pipeline including:
- Image preprocessing
- Sparse structure sampling
- SLAT sampling
- Decoding
- Mesh postprocessing
- Texture baking
"""

import os
import sys

# Environment setup
os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
from PIL import Image
from datetime import datetime

# Import pipeline components
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from profiling.profiler import Profiler


def profile_pipeline_run(profiler: Profiler, pipeline: TrellisImageTo3DPipeline, image: Image.Image, seed: int = 1):
    """
    Profile a single run of the pipeline with detailed stage breakdown.
    
    Args:
        profiler: Profiler instance
        pipeline: The TrellisImageTo3DPipeline instance
        image: Input PIL image
        seed: Random seed
    
    Returns:
        outputs: Dictionary containing generated 3D assets
    """
    
    # Stage 1: Preprocess image
    with profiler.profile_stage("1. Preprocess Image"):
        preprocessed_image = pipeline.preprocess_image(image)
    
    # Stage 2: Encode image (get conditioning)
    with profiler.profile_stage("2. Encode Image (get_cond)"):
        cond = pipeline.get_cond([preprocessed_image])
    
    # Set random seed
    torch.manual_seed(seed)
    
    # Stage 3: Sample sparse structure
    with profiler.profile_stage("3. Sample Sparse Structure"):
        coords = pipeline.sample_sparse_structure(cond, num_samples=1, sampler_params={})
    
    # Stage 4: Sample SLAT
    with profiler.profile_stage("4. Sample SLAT"):
        slat = pipeline.sample_slat(cond, coords, sampler_params={})
    
    # Stage 5: Decode SLAT
    with profiler.profile_stage("5. Decode SLAT"):
        outputs = pipeline.decode_slat(slat, formats=['mesh', 'gaussian', 'radiance_field'])
    
    return outputs


def profile_to_glb(profiler: Profiler, gaussian, mesh, simplify: float = 0.95, texture_size: int = 1024):
    """
    Profile the to_glb conversion with detailed stage breakdown.
    
    Args:
        profiler: Profiler instance
        gaussian: Gaussian representation
        mesh: Mesh representation
        simplify: Simplification ratio
        texture_size: Texture size
    
    Returns:
        glb: The final GLB mesh
    """
    
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    
    # Stage 6: Postprocess mesh (simplify + fill holes)
    with profiler.profile_stage("6. Postprocess Mesh"):
        vertices, faces = postprocessing_utils.postprocess_mesh(
            vertices, faces,
            simplify=simplify > 0,
            simplify_ratio=simplify,
            fill_holes=True,
            fill_holes_max_hole_size=0.04,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)),
            fill_holes_resolution=1024,
            fill_holes_num_views=1000,
            debug=False,
            verbose=True,
        )
    
    # Stage 7: Parametrize mesh (UV mapping)
    with profiler.profile_stage("7. Parametrize Mesh (UV mapping)"):
        vertices, faces, uvs = postprocessing_utils.parametrize_mesh(vertices, faces)
    
    # Stage 8: Render multiview for texture baking
    with profiler.profile_stage("8. Render Multiview"):
        from trellis.utils.render_utils import render_multiview
        observations, extrinsics, intrinsics = render_multiview(gaussian, resolution=1024, nviews=100)
        masks = [np.any(observation > 0, axis=-1) for observation in observations]
        extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
        intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
    
    # Stage 9: Bake texture
    with profiler.profile_stage("9. Bake Texture"):
        texture = postprocessing_utils.bake_texture(
            vertices, faces, uvs,
            observations, masks, extrinsics, intrinsics,
            texture_size=texture_size, mode='opt',
            lambda_tv=0.01,
            verbose=True
        )
        texture = Image.fromarray(texture)
    
    # Stage 10: Create final GLB
    with profiler.profile_stage("10. Create Final GLB"):
        import trimesh
        import trimesh.visual
        
        # rotate mesh (from z-up to y-up)
        vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        material = trimesh.visual.material.PBRMaterial(
            roughnessFactor=1.0,
            baseColorTexture=texture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
        )
        glb = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    
    return glb


def main():
    """Main profiling function"""
    
    print(f"\n{'#'*100}")
    print(f"# DETAILED PROFILING OF IMAGE-TO-3D PIPELINE")
    print(f"# Timestamp: {datetime.now().isoformat()}")
    print(f"{'#'*100}\n")
    
    # Initialize profiler 
    # device_id=0 because CUDA_VISIBLE_DEVICES=1 makes GPU 1 appear as GPU 0
    # nvidia_smi_device_id=1 because we want to query the actual GPU 1 from nvidia-smi
    profiler = Profiler(device_id=0, nvidia_smi_device_id=1)
    
    # Test image - using just one image for detailed profiling
    image_path = "assets/example_image/typical_vehicle_biplane.png"
    output_dir = "outputs"
    profiling_dir = "profiling"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(profiling_dir):
        os.makedirs(profiling_dir)
    
    # Load pipeline
    print("Loading pipeline...")
    with profiler.profile_stage("0. Load Pipeline"):
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline.cuda()
    
    print(f"\nProcessing image: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    
    # Profile pipeline run
    print("\n" + "="*100)
    print("PROFILING PIPELINE STAGES")
    print("="*100)
    outputs = profile_pipeline_run(profiler, pipeline, image, seed=1)
    
    # Profile to_glb conversion
    print("\n" + "="*100)
    print("PROFILING GLB CONVERSION STAGES")
    print("="*100)
    glb = profile_to_glb(
        profiler,
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024
    )
    
    # Save output
    output_filename = os.path.join(output_dir, f"{image_path.split('/')[-1].split('.')[0]}_profiled.glb")
    glb.export(output_filename)
    print(f"\nOutput saved to: {output_filename}")
    
    # Print summary
    profiler.print_summary()
    
    # Save detailed results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = os.path.join(profiling_dir, f"profile_baseline_{timestamp}.json")
    profiler.save_to_json(json_filename)
    
    # Clean up
    del outputs
    del glb
    torch.cuda.empty_cache()
    
    print("\nProfiling complete!")


if __name__ == "__main__":
    main()

