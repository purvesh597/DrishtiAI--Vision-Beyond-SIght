"""
DrishtiAI — FastAPI WebSocket Backend
Deployed on HuggingFace Spaces (Docker)
"""

import base64
import json
import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

# ── Load model ──────────────────────────────────────────
# Model file must be uploaded to HF Space as best.pt
MODEL_PATH = "best.pt"

print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Please upload best.pt to your HuggingFace Space."
    )

model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names
print(f"✅ Model loaded. Classes: {CLASS_NAMES}")

# ── FastAPI app ──────────────────────────────────────────
app = FastAPI(title="DrishtiAI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Color map for bounding boxes ─────────────────────────
COLOR_MAP = {
    "Red Light":   "#ef4444",
    "Green Light": "#22c55e",
    "Stop":        "#f97316",
}

# ── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":  "DrishtiAI running ✅",
        "model":   MODEL_PATH,
        "classes": CLASS_NAMES,
        "ws_url":  "/ws"
    }

@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = websocket.client
    print(f"🔌 Client connected: {client}")

    try:
        while True:
            # ── 1. Receive frame ──────────────────────────
            raw     = await websocket.receive_text()
            payload = json.loads(raw)
            b64     = payload.get("frame", "")

            if not b64:
                await websocket.send_text(json.dumps({"detections": []}))
                continue

            # ── 2. Decode JPEG → numpy ────────────────────
            img_bytes = base64.b64decode(b64)
            arr       = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({"detections": []}))
                continue

            # ── 3. YOLOv8 inference ───────────────────────
            results = model.predict(
                source  = frame,
                conf    = 0.45,
                iou     = 0.5,
                verbose = False,
            )[0]

            # ── 4. Build response ─────────────────────────
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
                label = CLASS_NAMES[int(box.cls)]
                conf  = round(float(box.conf), 3)
                detections.append({
                    "label": label,
                    "conf":  conf,
                    "bbox":  [x1, y1, x2, y2],
                    "color": COLOR_MAP.get(label, "#4f6ef7"),
                })

            await websocket.send_text(json.dumps({
                "detections": detections,
                "count":      len(detections),
            }))

            if detections:
                print(f"  → {[(d['label'], d['conf']) for d in detections]}")

    except WebSocketDisconnect:
        print(f"🔌 Client disconnected: {client}")
    except Exception as e:
        print(f"⚠️  Error: {type(e).__name__}: {e}")
        try:
            await websocket.send_text(json.dumps({"detections": [], "error": str(e)}))
        except Exception:
            pass
