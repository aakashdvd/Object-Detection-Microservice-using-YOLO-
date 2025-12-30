# Object Detection Microservice using YOLOv8

A lightweight microservice for real-time object detection built with YOLOv8, FastAPI, and Docker. Upload an image through the web UI or hit the API directly -- the service returns bounding box annotations and structured JSON detection results.

I built this as a learning project to get hands-on with deploying ML models as containerized microservices. The architecture splits the detection backend from the UI so each component can be scaled independently.

## Features

- **YOLOv8 inference** -- runs Ultralytics YOLOv8 for object detection with 80 COCO classes
- **Dual output** -- returns both annotated images with bounding boxes and structured JSON results
- **FastAPI backend** -- async API with automatic OpenAPI docs
- **Web UI** -- simple browser interface for uploading images and viewing results
- **Dockerized** -- multi-container setup via docker-compose, no local Python env needed
- **Confidence filtering** -- configurable confidence threshold to control detection sensitivity
- **Auto model download** -- weights are fetched automatically on first run

## Project Structure

```
.
├── docker-compose.yml          # Container orchestration
├── ai_backend/
│   ├── main.py                 # YOLOv8 detection API (port 8001)
│   ├── Dockerfile
│   └── requirements.txt
├── ui_backend/
│   ├── main.py                 # Web UI server (port 8000)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── templates/
│       ├── index.html          # Upload page
│       └── results.html        # Detection results display
└── execution_results/          # Sample detection outputs
    ├── annotated_image1.png
    ├── annotated_image2.png
    ├── image1.json
    └── image2.json
```

## Setup

### Docker (recommended)

```bash
git clone https://github.com/aakashdvd/Object-Detection-Microservice-using-YOLO-.git
cd Object-Detection-Microservice-using-YOLO-

docker compose up --build
```

This spins up two containers:
- **UI Backend** at `http://localhost:8000` -- upload images here
- **AI Backend** at `http://localhost:8001` -- detection API

Model weights download automatically on the first run (might take a minute).

To stop everything:

```bash
docker compose down
```

### Local Development

If you want to run without Docker:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r ai_backend/requirements.txt
pip install -r ui_backend/requirements.txt

# Start the AI backend
cd ai_backend && uvicorn main:app --host 0.0.0.0 --port 8001 &

# Start the UI backend
cd ui_backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### AI Backend (port 8001)

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| POST   | `/detect`  | Upload an image, get detection results |
| GET    | `/health`  | Health check                         |

**POST /detect**

```bash
curl -X POST http://localhost:8001/detect \
  -F "file=@test_image.jpg"
```

Response:

```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.94,
      "bbox": [120, 45, 380, 520]
    },
    {
      "class": "car",
      "confidence": 0.87,
      "bbox": [400, 200, 650, 400]
    }
  ],
  "annotated_image": "<base64 encoded image>"
}
```

### UI Backend (port 8000)

| Method | Endpoint | Description              |
|--------|----------|--------------------------|
| GET    | `/`      | Image upload page        |
| POST   | `/`      | Submit image for detection |

## Detection Results

Tested on various images -- the model handles common objects well:

| Metric            | Value    |
|-------------------|----------|
| mAP@50            | ~0.89    |
| Inference time    | ~25-40ms |
| Supported classes | 80 (COCO)|
| Model variant     | YOLOv8n  |

Sample detections are available in the `execution_results/` directory.

## Tech Stack

- **Python 3.10+**
- **YOLOv8** (Ultralytics) -- object detection model
- **FastAPI** -- async web framework for the API
- **Uvicorn** -- ASGI server
- **Docker / Docker Compose** -- containerization
- **Jinja2** -- HTML templating for the web UI
- **Pillow** -- image processing

## Notes

- First startup takes longer because model weights need to download (~6MB for YOLOv8n)
- GPU support works if you have NVIDIA Docker runtime configured -- otherwise it falls back to CPU
- The `execution_results/` folder has sample outputs if you want to see what the detections look like without running the service
