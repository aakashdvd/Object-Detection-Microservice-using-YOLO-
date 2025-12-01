from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import uvicorn
import base64

app = FastAPI(title="AI Backend - YOLO Object Detection")

# Load lightweight YOLO model once at startup
model = YOLO("yolov8n.pt")  # ultralytics will download if not present


def run_detection(img: np.ndarray):
    """Run YOLO on an image and return (results_object, json_payload_without_image)."""
    results = model(img)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            {
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": conf,
                "box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
            }
        )

    h, w = img.shape[:2]

    payload = {
        "image": {"width": w, "height": h},
        "num_detections": len(detections),
        "detections": detections,
    }

    return results, payload


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Original endpoint: returns ONLY JSON.
    Kept intact for compatibility.
    """
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image. Could not decode."},
        )

    _, payload = run_detection(img)
    return payload


@app.post("/detect_with_image")
async def detect_objects_with_image(file: UploadFile = File(...)):
    """
    New endpoint: returns JSON + base64 encoded annotated image.
    """
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image. Could not decode."},
        )

    results, payload = run_detection(img)

    # Get annotated frame (BGR) and encode to JPEG
    annotated_frame = results.plot()
    success, buffer = cv2.imencode(".jpg", annotated_frame)
    if not success:
        # Fallback: just return JSON if encoding fails
        return payload

    img_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    payload["annotated_image_base64"] = img_b64
    payload["annotated_image_format"] = "jpg"

    return payload


if __name__ == "__main__":
    # For local dev (not used in Docker, but nice to keep)
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
