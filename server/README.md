# TRELLIS FastAPI Server

FastAPI server with two endpoints to compare baseline vs optimized approaches for TRELLIS image-to-3D conversion.

## Setup

Install dependencies:
```bash
pip install fastapi uvicorn python-multipart
```

## Running the Server

```bash
cd server
python app.py
```

Or from the project root:
```bash
python server/app.py
```

Server runs on `http://0.0.0.0:8000`

## API Endpoints

### GET `/`
Health check and available endpoints information.

### POST `/convert/baseline`
Convert image using baseline approach (from `impl/baseline.py`).
- Sequential execution: GPU generation → CPU post-processing
- Straightforward implementation

### POST `/convert/optimized`
Convert image using optimized approach (from `impl/optimized_v2.py`).
- Uses threading to potentially overlap GPU and CPU work
- Frees GPU memory faster for subsequent requests
- Main benefit visible with concurrent requests

### Parameters (both endpoints)
- `image` (file, required): Image file to convert
- `seed` (int, optional): Random seed (default: 1)
- `simplify` (float, optional): Triangle reduction ratio (default: 0.95)
- `texture_size` (int, optional): Texture size (default: 1024)

### Response Headers
Both endpoints return timing information in response headers:
- `X-Total-Time`: Total processing time
- `X-GPU-Time`: GPU generation time
- `X-PostProc-Time`: Post-processing time
- `X-Method`: Which method was used

## Testing

### Using the shell script:
```bash
./scripts/one_request.sh
```

### Using the Python script (recommended):
```bash
python scripts/compare_endpoints.py
```

### Using cURL:

**Baseline:**
```bash
curl -X POST "http://localhost:8000/convert/baseline" \
  -F "image=@assets/example_image/T.png" \
  -o outputs/T_baseline.glb \
  -v 2>&1 | grep "X-"
```

**Optimized:**
```bash
curl -X POST "http://localhost:8000/convert/optimized" \
  -F "image=@assets/example_image/T.png" \
  -o outputs/T_optimized.glb \
  -v 2>&1 | grep "X-"
```

### Using Python:
```python
import requests

# Test baseline
with open("assets/example_image/T.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/convert/baseline",
        files={"image": f},
        data={"seed": 1}
    )

print(f"Total Time: {response.headers['X-Total-Time']}s")
print(f"GPU Time: {response.headers['X-GPU-Time']}s")
print(f"PostProc Time: {response.headers['X-PostProc-Time']}s")

with open("output_baseline.glb", "wb") as f:
    f.write(response.content)
```

## Performance Notes

- **Baseline**: Simple sequential approach, easier to understand
- **Optimized**: Uses threading to free GPU faster, beneficial when:
  - Processing multiple concurrent requests
  - GPU memory is bottleneck
  - Want faster GPU availability for next request

For single isolated requests, the performance difference may be minimal. The optimized approach shines with concurrent workloads or batch processing.

## Architecture

```
Request → FastAPI → Load Image → Pipeline
                                    ↓
                    [BASELINE]      ↓
                    GPU Gen → CPU PostProc → Return GLB
                    
                    [OPTIMIZED]     ↓
                    GPU Gen → Thread(CPU PostProc) → Return GLB
                    (GPU freed immediately after generation)
```

