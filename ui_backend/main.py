from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx
import uvicorn
import json

AI_SERVICE_URL = "http://ai-backend:8001/detect_with_image"

app = FastAPI(title="UI Backend - Image Uploader")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render upload page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    """Handle upload, call AI backend, show JSON + annotated image."""
    file_bytes = await file.read()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            AI_SERVICE_URL,
            files={"file": (file.filename, file_bytes, file.content_type)},
            timeout=120.0,
        )
        response.raise_for_status()

    result_json = response.json()
    formatted_json = json.dumps(result_json, indent=2)

    annotated_b64 = result_json.get("annotated_image_base64")
    image_format = result_json.get("annotated_image_format", "jpg")
    annotated_image_url = None
    if annotated_b64:
        annotated_image_url = f"data:image/{image_format};base64,{annotated_b64}"

    image_info = result_json.get("image") or {}
    img_width = image_info.get("width", "?")
    img_height = image_info.get("height", "?")
    num_detections = result_json.get("num_detections", 0)

    context = {
        "request": request,
        "file_name": file.filename,
        "img_width": img_width,
        "img_height": img_height,
        "num_detections": num_detections,
        "formatted_json": formatted_json,
        "annotated_image_url": annotated_image_url,
    }

    return templates.TemplateResponse("results.html", context)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
