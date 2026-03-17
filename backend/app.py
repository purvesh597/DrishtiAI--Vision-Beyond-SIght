import asyncio
import base64
import json
import cv2
import numpy as np
import uvicorn
import io
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import os

app = FastAPI(title="Drishti AI API")

# Allow browser to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING ---
MODELS_DIR = "models"
TRAFFIC_MODEL_PATH = os.path.join(MODELS_DIR, "traffic.pt")
CURRENCY_MODEL_PATH = os.path.join(MODELS_DIR, "currency.pt")

traffic_model = None
currency_model = None

print("Loading models...")

# Load Traffic Model
try:
    if os.path.exists(TRAFFIC_MODEL_PATH):
        traffic_model = YOLO(TRAFFIC_MODEL_PATH)
        # Warmup with 800px
        dummy = np.zeros((800, 800, 3), dtype=np.uint8)
        traffic_model(dummy, verbose=False)
        print("✅ Traffic model loaded")
    else:
        print("⚠️ traffic.pt not found in models/")
except Exception as e:
    print(f"⚠️ Error loading traffic model: {e}")

# Load Currency Model
try:
    if os.path.exists(CURRENCY_MODEL_PATH):
        currency_model = YOLO(CURRENCY_MODEL_PATH)
        # Warmup with 800px
        dummy = np.zeros((800, 800, 3), dtype=np.uint8)
        currency_model(dummy, verbose=False)
        print("✅ Currency model loaded")
    else:
        print("⚠️ currency.pt not found in models/")
except Exception as e:
    print(f"⚠️ Error loading currency model: {e}")

if not traffic_model and not currency_model:
    print("❌ NO MODELS LOADED. Detection will not work.")

# --- REST ENDPOINTS (from update_backend.py) ---

@app.get("/")
def health():
    return {
        "status": "DrishtiAI backend running ✅", 
        "models_loaded": {
            "traffic": traffic_model is not None,
            "currency": currency_model is not None
        }
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...), model: str = "traffic"):
    try:
        t0 = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        mdl = traffic_model if model == "traffic" else currency_model
        if mdl is None:
            return JSONResponse({"success": False, "error": "Model not loaded"}, status_code=500)
            
        results = mdl(img_array, conf=0.45, verbose=False, imgsz=640)[0]

        detections = []
        for box in results.boxes:
            detections.append({
                "label": mdl.names[int(box.cls)].replace("_", " "),
                "confidence": round(float(box.conf), 3),
                "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                "model": model
            })

        detections.sort(key=lambda x: x["confidence"], reverse=True)
        elapsed = round(time.time() - t0, 3)

        return JSONResponse({
            "success": True,
            "model": model,
            "count": len(detections),
            "detections": detections,
            "inference_ms": elapsed
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/detect/both")
async def detect_both(file: UploadFile = File(...)):
    try:
        t0 = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        all_detections = []
        for model_name, mdl in [("traffic", traffic_model), ("currency", currency_model)]:
            if mdl is None: continue
            results = mdl(img_array, conf=0.45, verbose=False, imgsz=640)[0]
            for box in results.boxes:
                all_detections.append({
                    "label": mdl.names[int(box.cls)].replace("_", " "),
                    "confidence": round(float(box.conf), 3),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                    "model": model_name
                })

        all_detections.sort(key=lambda x: x["confidence"], reverse=True)
        elapsed = round(time.time() - t0, 3)

        return JSONResponse({
            "success": True,
            "count": len(all_detections),
            "detections": all_detections,
            "inference_ms": elapsed
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# --- WEBSOCKET ENDPOINT (Compatibility Layer) ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🔌 Browser connected via WebSocket!")

    try:
        while True:
            # Receive frame from browser
            raw = await ws.receive_text()
            payload = json.loads(raw)
            frame_b64 = payload.get("frame", "")

            # Decode base64 → OpenCV image
            img_bytes = base64.b64decode(frame_b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None or (traffic_model is None and currency_model is None):
                await ws.send_text(json.dumps({"detections": []}))
                continue

            all_detections = []
            
            # Run both models for comprehensive output
            for model_name, mdl in [("traffic", traffic_model), ("currency", currency_model)]:
                if mdl is None: continue
                # Higher threshold (0.60) to avoid false positives with random blue/red objects
                c_thr = 0.60 if model_name == "traffic" else 0.65
                results = mdl(frame, conf=c_thr, iou=0.45, imgsz=800, verbose=False)[0]
                
                for box in results.boxes:
                    label = mdl.names[int(box.cls)].replace("_", " ")
                    bbox = [round(v) for v in box.xyxy[0].tolist()]
                    
                    all_detections.append({
                        "label": label,
                        "conf": round(float(box.conf), 3),
                        "bbox": bbox,
                        "color": get_color(label),
                        "model": model_name
                    })

            # Send combined detections back to browser
            await ws.send_text(json.dumps({"detections": all_detections}))

    except WebSocketDisconnect:
        print("🔌 Browser disconnected")
    except Exception as e:
        print(f"⚠️ WebSocket Error: {e}")

def get_color(label):
    """Return a hex color per class for bounding boxes based on sign category"""
    # Prohibitory (Red)
    red_signs = {
        "stop", "no entry", "all motor vehicle prohibited", "horn prohibited", 
        "left turn prohibited", "overtaking prohibited", "right turn prohibited", 
        "straight prohibited", "u turn prohibited", "no stopping or standing",
        "no parking", "restriction ends"
    }
    # Warning (Orange)
    orange_signs = {
        "roundabout", "pedestrian crossing", "slippery road", "cross road", 
        "dangerous dip", "falling rocks", "gap in median", "give way", 
        "guarded level crossing", "hump or rough road", "left hand curve", 
        "left reverse bend", "loose gravel", "men at work", "narrow bridge ahead", 
        "narrow road ahead", "quay side or river bank", "right hand curve", 
        "right reverse bend", "road widens ahead", "school ahead", "side road left", 
        "side road right", "staggered intersection", "steep ascent", "steep descent", 
        "t intersection", "u turn", "unguarded level crossing", "y intersection"
    }
    # Mandatory / Info (Blue)
    blue_signs = {
        "compulsary ahead", "compulsary keep left", "compulsary keep right", 
        "compulsary turn left ahead", "compulsary turn right ahead", "pass either side",
        "speed limit 100", "speed limit 30", "speed limit 50", "petrol pump ahead",
        "hospital ahead", "axle load limit", "height limit", "width limit"
    }

    if label in red_signs:
        return "#ff0033" # Neon Red
    elif label in orange_signs:
        return "#ffcc00" # Neon Gold/Yellow-Orange
    elif label in blue_signs:
        return "#00ffff" # Neon Cyan/Electric Blue
    return "#ff00ff" # Neon Magenta / Pink

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
