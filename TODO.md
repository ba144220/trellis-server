# Build an image to 3D model pipeline

# Goal
We're building a image to 3D model pipeline. Currently (take a look at `example.py`) it takes a single image and generates a 3D model (glb file). We want to increase the throughput of the pipeline.

For each request, the input is a single PNG image, and the output is a glb file. No need to generate the video or the gaussian files.


# Environment
```bash
conda activate trellis
```

# Run a simple example
```bash
CUDA_VISIBLE_DEVICES=1 python example.py
```
I'm assigned to use GPU 1.

# Plans

## Increase batch size
My colleague said that they were able to run two processes in parallel but not three. However, these two processes both copy the same model weights to the GPU, so I think we can instead increase the batch size of the pipeline.

But before we do that, we need to make sure that
1. Is the pipeline compute-bound or memory-bound?


## Quantization
My colleague said they tried to quantize the model, but it only improved accuracy by a small amount.


# Development Logs

For every feature or improvement, please add a log here. 
- Keep the log in chronological order.
- Keep the log concise and to the point.


## 2025-10-17 - Baseline
Process one image at a time.
Time taken: 20.22 s/image

## 2025-10-17 - Process multiple images in parallel
First attempt is to simply run two pipelines in parallel.
Time taken: 98.50 seconds

The problem is that the two pipelines are copying the same model weights to the GPU, which is not efficient.

The next attempt is just use one pipeline but spawn multiple workers to process the images.
For 2 workers, time taken: 16.08 seconds
For 3 workers, time taken: 15.16 seconds
For 4 workers, time taken: 15.24 seconds

But both GPU memory and utilization are not fully utilized.

## 2025-10-17 - Detailed Profiling
Ran detailed profiling on baseline with a single image to understand bottlenecks.


Key findings:
- **Texture Baking** is the biggest bottleneck: 6.23s (15.6% of total time)
  - GPU utilization: 36% (not fully utilized)
  - Running 2,500 optimization iterations
- **Postprocess Mesh** is second: 3.33s (8.4% of total time)
  - GPU utilization: 13% (CPU-bound)
  - Includes simplification and hole filling
- **Core Pipeline** (sampling + encoding): 5.58s (14% of total time)
  - Sample Sparse Structure: 1.90s with 82% GPU utilization (excellent!)
  - Sample SLAT: 3.17s with 41% GPU utilization (can be improved)
  - Decode SLAT: 0.34s with 51% GPU utilization

**Batching potential:**
- Pipeline stages (encode → decode): Can be batched, estimated 1.5-2x speedup for batch_size=2
- Post-processing stages: Mostly per-image operations, cannot be batched

**Memory usage:**
- Peak GPU memory: 24.7 GB (for 1 image)
- Large jump at decode stage: 5.7GB → 20.9GB
- Should be able to fit 2 images in 48GB GPU

**Recommendations:**
1. Implement batching for core pipeline (stages 1-5)
2. Optimize texture baking (reduce iterations or use 'fast' mode)
3. Reduce postprocessing overhead (fewer views for hole filling)

See `profiling/PROFILING_REPORT.md` for detailed analysis.

## 2025-10-17 - Use Fast Mode for Texture Baking
Texture baking is the biggest bottleneck. We can use fast mode to reduce the time taken.

We get 25% speedup by using fast mode.


## 2025-10-17 - Separating Inference and Postprocessing

Based on the profiling, the inference is compute-bound, but the postprocessing is memory-bound. 
So we can separate the inference and postprocessing into two separate workers, hiding the latency.

Another key finding is that postprocessing takes more time, so I run 2-3 postprocessing workers to increase the throughput. (the increase tops out at 3 workers)

With fast mode texture baking and 3 postprocessing workers, the speed is 9.92 s/image.

Separating inference and postprocessing gives us another 25% speedup.

## Discussion - Can we increase the inference worker as well?

I ran 2 inference workers and 4 postprocessing workers, it's almost the same as 1 inference worker and 4 postprocessing workers.



# How to run the server?
In one terminal, run the server:
```bash
CUDA_VISIBLE_DEVICES=1 # Set the GPU to use
conda activate trellis
python server/app.py
```

Wait until the server is ready.

In another terminal, run the test script:
```bash
conda activate trellis
python scripts/compare_endpoints.py
```

This will send several requests to the server with 1 second interval.