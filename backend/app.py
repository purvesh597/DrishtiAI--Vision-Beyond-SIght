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
                # Optimized for 1.5m distance (imgsz=800)
                # Traffic is more permissive (0.4) for distant objects
                c_thr = 0.40 if model_name == "traffic" else 0.55
                results = mdl(frame, conf=c_thr, iou=0.4, imgsz=800, verbose=False)[0]
                
                for box in results.boxes:
                    label = mdl.names[int(box.cls)].replace("_", " ")
                    bbox = [round(v) for v in box.xyxy[0].tolist()]
                    pos = get_spatial_position(bbox)
                    
                    all_detections.append({
                        "label": label,
                        "conf": round(float(box.conf), 3),
                        "bbox": bbox,
                        "position": pos,
                        "color": get_color(label),
                        "model": model_name
                    })

            # Non-blocking Grok integration with caching
            if all_detections:
                top = all_detections[0]
                cache_key = (top["label"], top["position"])
                
                if cache_key in grok_cache:
                    top["voice_msg"] = grok_cache[cache_key]
                else:
                    # Fire and forget Grok fetch in background
                    asyncio.create_task(fetch_grok_msg(top["label"], top["position"]))
                    # No voice_msg yet, frontend will use fallback for speed
                    top["voice_msg"] = None

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

# --- GROK AI & CACHING ---
from openai import OpenAI
import asyncio

XAI_API_KEY = os.getenv("XAI_API_KEY", "your_fallback_key_here")
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

# Global cache for Grok messages: { (label, position): "msg" }
grok_cache = {}
active_grok_tasks = set()

def get_spatial_position(bbox, frame_width=1280):
    """
    Determine if sign is on the left, right, or center.
    """
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2
    
    if center_x < frame_width / 3:
        return "on your left"
    elif center_x > (2 * frame_width) / 3:
        return "on your right"
    else:
        return "dead ahead"

async def fetch_grok_msg(label, position):
    """Background task to fetch and cache a human-like message from Grok"""
    cache_key = (label, position)
    if cache_key in active_grok_tasks:
        return
        
    active_grok_tasks.add(cache_key)
    try:
        # Prompt for ultra-human, casual, and friendly speech
        response = await asyncio.to_thread(client.chat.completions.create,
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are a friendly human friend guiding a blind person. Speak CASUALLY and NATURALLY. No robotic 'detected' phrases. Use things like 'Hey', 'Watch out', 'Mind the...', 'Look, there's a...'. MAX 10 words total."},
                {"role": "user", "content": f"You see a '{label}' sign {position}. Tell your friend what to do naturally."}
            ],
            temperature=0.9, # Higher temperature for more variety
            max_tokens=25
        )
        msg = response.choices[0].message.content.strip()
        # Remove common "AI markers"
        msg = msg.replace('Detected: ', '').replace('Attention: ', '').strip('"')
        grok_cache[cache_key] = msg
    except Exception as e:
        print(f"Grok Error: {e}")
    finally:
        active_grok_tasks.remove(cache_key)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
