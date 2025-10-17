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
python example.py
```

