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