# Object Detection Using YOLOv8 (FastAPI + Docker Microservices)

This project implements a simple microservice-based object detection system using FastAPI, Ultralytics YOLOv8, and Docker.
Users can upload an image through a web UI, and the system returns:

JSON detection results

An annotated image with bounding boxes

## 🚀 Features

YOLOv8 object detection (Ultralytics)

FastAPI-based UI + AI backend

Docker containerized (no environment conflicts)

Clean browser interface for uploading images

JSON output + annotated image preview

## 📁 Project Structure
```
Object-Detection-YOLO/
│
├── docker-compose.yml
│
├── ui_backend/
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── templates/
│       ├── index.html
│       └── results.html
│
└── ai_backend/
    ├── main.py
    ├── Dockerfile
    └── requirements.txt
```

## 🛠️ Requirements

Before running the project, install:

Docker Desktop → https://www.docker.com/products/docker-desktop

(Optional) Git → https://git-scm.com/

## 📥 Setup & Run
### 1️⃣ Clone the Repository
` git clone https://github.com/aakashdvd/Object-Detection-YOLO.git`<br>
`cd Object-Detection-YOLO`

### 2️⃣ Build and Run the Services
``docker compose up --build``


This starts:

UI Backend → http://localhost:8000

AI Backend → http://localhost:8001

Model weights download automatically on first run.

### 3️⃣ Use the Application

Open your browser and visit:

http://localhost:8000


<ins>Upload an image</ins>

View:

JSON detection output

Annotated image

### 4️⃣ Stop the Application

Press `CTRL + C`, then run:

``docker compose down``
