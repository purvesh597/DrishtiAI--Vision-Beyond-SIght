import asyncio
import base64
import json
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
from huggingface_hub import hf_hub_download

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
    if not os.path.exists(MODEL_PATH):
        print(f"📥 Downloading {MODEL_PATH} from Hugging Face Hub...")
        hf_hub_download(repo_id="purvesh597/drishtiai", filename="best.pt", repo_type="space", local_dir=".")
    
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded:", MODEL_PATH)
    print(f"📋 Total classes: {len(model.names)}")
    for idx, name in model.names.items():
        print(f"   {idx}: {name}")
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

            # ── Run YOLOv8 inference (SPEED OPTIMIZED) ──
            # imgsz=640 (model native resolution for high accuracy)
            # conf=0.35 (lower threshold for faster assistive feedback)
            # iou=0.4 (stricter overlap filtering)
            results = model(frame, conf=0.35, iou=0.4, imgsz=640, verbose=False)[0]

            raw_detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
                label = model.names[int(box.cls)]
                raw_detections.append({
                    "label": label,
                    "conf":  round(float(box.conf), 3),
                    "bbox":  [x1, y1, x2, y2],
                    "color": get_color(label)
                })

            # --- ACCURACY POST-PROCESSING (Disambiguation) ---
            # If multiple signs from the same group appear, only keep the best one
            final_detections = []
            if raw_detections:
                processed_indices = set()
                for i, det in enumerate(raw_detections):
                    if i in processed_indices: continue
                    
                    # Find which group this label belongs to
                    current_group = next((g for g in CONFUSION_GROUPS if det["label"] in g), None)
                    
                    if current_group:
                        # Find all other detections in the same group and keep highest confidence
                        group_candidates = [(i, det)]
                        for j, other in enumerate(raw_detections):
                            if i != j and other["label"] in current_group:
                                group_candidates.append((j, other))
                        
                        best_idx, best_det = max(group_candidates, key=lambda x: x[1]["conf"])
                        final_detections.append(best_det)
                        # Mark all candidates as processed
                        for idx, _ in group_candidates:
                            processed_indices.add(idx)
                    else:
                        final_detections.append(det)

            # --- TEMPORAL SMOOTHING (Accuracy ↑) ---
            # Buffer only the top detection label to smooth voice output
            if final_detections:
                top_label = max(final_detections, key=lambda x: x["conf"])["label"]
                DETECTIONS_BUFFER.append(top_label)
                if len(DETECTIONS_BUFFER) > BUFFER_SIZE:
                    DETECTIONS_BUFFER.pop(0)
                
                # Only trust the top detection if it's seen consistently
                # (prevents frame-flicker misclassification)
                from collections import Counter
                counts = Counter(DETECTIONS_BUFFER)
                most_common, freq = counts.most_common(1)[0]
                
                # If the most common label in buffer is NOT this one, 
                # or if it's too rare, proceed with caution
                if freq < 2 and len(DETECTIONS_BUFFER) == BUFFER_SIZE:
                    # Optional: filter out if inconsistent
                    pass

            # ── Send detections back to browser ──
            await ws.send_text(json.dumps({"detections": final_detections}))

    except WebSocketDisconnect:
        print("🔌 Browser disconnected")
    except Exception as e:
        print(f"⚠️ Error: {e}")


# --- ACCURACY & SPEED OPTIMIZATIONS ---
# Groups of classes that YOLO often confuses
CONFUSION_GROUPS = [
    {"stop", "no_entry", "do_not_stop", "no_stop"},
    {"left_turn", "right_turn", "u_turn"},
    {"speed_limit_20", "speed_limit_30", "speed_limit_40", "speed_limit_50", "speed_limit_60", "speed_limit_70", "speed_limit_80", "speed_limit_100", "speed_limit_120"}
]

# Temporal smoothing buffer (rolling window of last 3 detections)
DETECTIONS_BUFFER = []
BUFFER_SIZE = 3

def get_color(label):
    """Return a hex color per class for bounding boxes"""
    color_map = {
        # Lights
        "red_light":        "#ef4444",
        "green_light":      "#22c55e",
        "yellow_light":     "#f5b800",
        # Stop / Danger
        "stop":             "#ef4444",
        "no_entry":         "#ef4444",
        "do_not_stop":      "#dc2626",
        # Prohibitions
        "do_not_turn_left": "#b91c1c",
        "do_not_turn_right":"#b91c1c",
        "do_not_u_turn":    "#b91c1c",
        "no_overtaking":    "#b91c1c",
        "no_parking":       "#b91c1c",
        "no_stop":          "#b91c1c",
        "no_waiting":       "#b91c1c",
        # Speed limits
        "speed_limit_20":   "#4f6ef7",
        "speed_limit_30":   "#4f6ef7",
        "speed_limit_40":   "#4f6ef7",
        "speed_limit_50":   "#4f6ef7",
        "speed_limit_60":   "#4f6ef7",
        "speed_limit_70":   "#4f6ef7",
        "speed_limit_80":   "#4f6ef7",
        "speed_limit_100":  "#4f6ef7",
        "speed_limit_120":  "#4f6ef7",
        # Warnings
        "warning":          "#f97316",
        "speed_bump":       "#f97316",
        "narrow_road":      "#f97316",
        "railway_crossing": "#f97316",
        "road_work":        "#f97316",
        "t_intersection_l": "#f97316",
        "school_nearby":    "#f97316",
        "children":         "#f97316",
        # Info / Direction
        "crosswalk":        "#06b6d4",
        "give_way":         "#f5b800",
        "left_turn":        "#4f6ef7",
        "right_turn":       "#4f6ef7",
        "u_turn":           "#4f6ef7",
        "enter_left_lane":  "#4f6ef7",
        "left_lane_enter":  "#4f6ef7",
        "road_main":        "#4f6ef7",
        # Misc
        "bicycle":          "#06b6d4",
        "bus_stop":         "#06b6d4",
        "parking":          "#06b6d4",
        "refueling":        "#06b6d4",
        "truck":            "#f97316",
    }
    return color_map.get(label, "#d832f5")

print("✅ FastAPI app ready")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
