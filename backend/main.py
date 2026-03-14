import asyncio
import base64
import json
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# Allow browser (your HTML file) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    MODEL_PATH = "best.pt"
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded:", MODEL_PATH)
    print("📋 Classes:", model.names)
except Exception as e:
    print(f"⚠️ Error loading model at {MODEL_PATH}: {e}")
    print("⚠️ Please make sure you have downloaded best.pt and placed it in the backend folder.")
    model = None


@app.get("/")
def health():
    return {"status": "DrishtiAI backend running ✅"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🔌 Browser connected!")

    try:
        while True:
            # ── Receive frame from browser ──
            raw = await ws.receive_text()
            payload = json.loads(raw)
            frame_b64 = payload.get("frame", "")

            # ── Decode base64 → OpenCV image ──
            img_bytes = base64.b64decode(frame_b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None or model is None:
                await ws.send_text(json.dumps({"detections": []}))
                continue

            # ── Run YOLOv8 inference ──
            results = model(frame, conf=0.4, verbose=False)[0]

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
                label = model.names[int(box.cls)]
                detections.append({
                    "label": label,
                    "conf":  round(float(box.conf), 3),
                    "bbox":  [x1, y1, x2, y2],   # pixel coords
                    "color": get_color(label)
                })

            # ── Send detections back to browser ──
            await ws.send_text(json.dumps({"detections": detections}))

    except WebSocketDisconnect:
        print("🔌 Browser disconnected")
    except Exception as e:
        print(f"⚠️ Error: {e}")


def get_color(label):
    """Return a hex color per class for bounding boxes"""
    color_map = {
        "Red Light":   "#ef4444",
        "Green Light": "#22c55e",
        "Stop":        "#f97316",
        "Speed 20":    "#4f6ef7",
        "Speed 30":    "#4f6ef7",
        "Speed 40":    "#4f6ef7",
        "Speed 50":    "#4f6ef7",
        "Speed 60":    "#4f6ef7",
        "Speed 70":    "#4f6ef7",
        "Speed 80":    "#4f6ef7",
        "Speed 90":    "#4f6ef7",
        "Speed 100":   "#4f6ef7",
        "Speed 110":   "#4f6ef7",
        "Speed 120":   "#4f6ef7",
    }
    return color_map.get(label, "#d832f5")

print("✅ FastAPI app ready")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
