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
import random

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

# --- PHRASES DICTIONARY ---

SIGN_PHRASES = {
    # Prohibitory
    "stop": {
        "en-US": ["Stop sign ahead. Please bring the vehicle to a full halt.", "Safe driving requires a full stop here.", "Prepare to stop completely for the sign ahead."],
        "hi-IN": ["आगे रुकने का संकेत है। कृपया पूरी तरह से रुकें।", "सुरक्षित ड्राइविंग के लिए यहाँ रुकना ज़रूरी है।", "सावधान, आगे रुकने का बोर्ड है।"],
        "mr-IN": ["पुढं थांबण्याचा इशारा आहे. कृपया पूर्णपणे थांबा.", "सुरक्षित प्रवासासाठी इथं थांबणं गरजेचं आहे.", "सावधान, पुढं स्टॉप साईन आहे."]
    },
    "no entry": {
        "en-US": ["No entry ahead. Please find an alternative route.", "Entry is restricted here. You should turn back.", "Forbidden path! This road is closed for entry."],
        "hi-IN": ["आगे प्रवेश निषेध है। कृपया कोई और रास्ता चुनें।", "यहाँ प्रवेश वर्जित है। आपको वापस मुड़ना चाहिए।", "यह रास्ता बंद है, आगे प्रवेश मना है।"],
        "mr-IN": ["पुढं प्रवेश बंदी आहे. कृपया दुसरा मार्ग निवडा.", "इथं प्रवेशासाठी बंदी आहे. कृपया परत फिरा.", "हा रस्ता पुढं बंद आहे."]
    },
    "all motor vehicle prohibited": {
        "en-US": ["All motor vehicles are prohibited on this road.", "No motor vehicles allowed beyond this point.", "Motorized traffic is not permitted here."],
        "hi-IN": ["इस सड़क पर सभी मोटर वाहनों का प्रवेश वर्जित है।", "यहाँ से आगे मोटर वाहनों की अनुमति नहीं है।", "यहाँ वाहनों का आना मना है।"],
        "mr-IN": ["या रस्त्यावर सर्व मोटर वाहनांना बंदी आहे.", "पुढं मोटर वाहनांना परवानगी नाही.", "इथं वाहनांना येण्यास मनाई आहे."]
    },
    "horn prohibited": {
        "en-US": ["Silence zone! Please avoid using the horn.", "Honking is not allowed here.", "Keep it quiet, horn prohibited ahead."],
        "hi-IN": ["शांति क्षेत्र! कृपया हॉर्न न बजाएं।", "यहाँ हॉर्न बजाना मना है।", "शांत रहें, आगे हॉर्न बजाना वर्जित है।"],
        "mr-IN": ["शांतता क्षेत्र! कृपया हॉर्न वाजवू नका.", "इथं हॉर्न वाजवण्यास मनाई आहे.", "पुढं हॉर्न वाजवू नका."]
    },
    "left turn prohibited": {
        "en-US": ["Left turn is not allowed here.", "No left turns permitted at this junction.", "Please do not turn left."],
        "hi-IN": ["यहाँ बाएं मुड़ना मना है।", "इस मोड़ पर बाएं न मुड़ें।", "सावधान, बायां मोड़ वर्जित है।"],
        "mr-IN": ["इथं डावीकडे वळण्यास मनाई आहे.", "या ठिकाणी डावीकडे वळू नका.", "सावधान, डावे वळण निषिद्ध आहे."]
    },
    "overtaking prohibited": {
        "en-US": ["No overtaking allowed on this stretch.", "Please keep to your lane, no overtaking.", "Stay in line, overtaking is risky and prohibited here."],
        "hi-IN": ["इस सड़क पर ओवरटेक करना मना है।", "कृपया अपनी लेन में रहें, ओवरटेक न करें।", "ओवरटेक करना यहाँ वर्जित और खतरनाक है।"],
        "mr-IN": ["या रस्त्यावर ओव्हरटेक करण्यास मनाई आहे.", "कृपया आपल्या लेनमध्ये रहा, ओव्हरटेक करू नका.", "इथं ओव्हरटेक करणं धोकादायक आणि निषिद्ध आहे."]
    },
    "right turn prohibited": {
        "en-US": ["Right turn is prohibited at this intersection.", "No right turns allowed here.", "Please continue straight or turn left; right turn is forbidden."],
        "hi-IN": ["इस चौराहे पर दाएं मुड़ना मना है।", "यहाँ दाएं मुड़ना वर्जित है।", "दाएं मुड़ना मना है, कृपया सीधे जाएं या बाएं मुड़ें।"],
        "mr-IN": ["या चौकात उजवीकडे वळण्यास मनाई आहे.", "इथं उजवीकडे वळणं वर्जित आहे.", "उजवीकडे वळण्यास मनाई आहे."]
    },
    "straight prohibited": {
        "en-US": ["Going straight is prohibited. Please turn.", "No straight path allowed ahead.", "The road ahead is closed for straight travel."],
        "hi-IN": ["सीधे जाना मना है। कृपया मुड़ें।", "आगे सीधे जाने का रास्ता नहीं है।", "सामने का रास्ता बंद है, कृपया मुड़ें।"],
        "mr-IN": ["सरळ जाण्यास मनाई आहे. कृपया वळा.", "पुढं सरळ जाण्यासाठी रस्ता नाही.", "सरळ जाण्यास बंदी आहे."]
    },
    "u turn prohibited": {
        "en-US": ["U-turns are not permitted here.", "No U-turns allowed on this road.", "Please do not attempt a U-turn here."],
        "hi-IN": ["यहाँ यू-टर्न लेना मना है।", "इस सड़क पर यू-टर्न वर्जित है।", "कृपया यहाँ से यू-टर्न न लें।"],
        "mr-IN": ["इथं यू-टर्न घेण्यास मनाई आहे.", "या रस्त्यावर यू-टर्न घेणं वर्जित आहे.", "कृपया इथून यू-टर्न घेऊ नका."]
    },
    "no stopping or standing": {
        "en-US": ["Stopping or standing is prohibited here.", "Please keep moving, no stopping allowed.", "No waiting or standing on this road."],
        "hi-IN": ["यहाँ रुकना या खड़ा होना मना है।", "कृपया चलते रहें, रुकना वर्जित है।", "इस सड़क पर गाड़ी खड़ी करना मना है।"],
        "mr-IN": ["इथं थांबण्यास किंवा उभं राहण्यास मनाई आहे.", "कृपया पुढे चालत रहा, थांबण्यास मनाई आहे.", "या रस्त्यावर गाडी थांबवू नका."]
    },
    "no parking": {
        "en-US": ["No parking allowed in this area.", "This is a no-parking zone.", "Please find a designated parking spot elsewhere."],
        "hi-IN": ["इस क्षेत्र में पार्किंग मना है।", "यह नो-पार्किंग ज़ोन है।", "कृपया कहीं और पार्किंग की जगह ढूंढें।"],
        "mr-IN": ["या परिसरात पार्किंग करण्यास मनाई आहे.", "हे नो-पार्किंग क्षेत्र आहे.", "कृपया इतर ठिकाणी पार्किंग शोधा."]
    },
    "restriction ends": {
        "en-US": ["The previous restriction has ended.", "You can now resume normal driving rules.", "End of restricted zone."],
        "hi-IN": ["पिछला प्रतिबंध अब समाप्त हो गया है।", "अब आप सामान्य नियमों का पालन कर सकते हैं।", "प्रतिबंधित क्षेत्र समाप्त।"],
        "mr-IN": ["मागील निर्बंध आता संपले आहेत.", "आता तुम्ही सामान्य नियमांचे पालन करू शकता.", "निर्बंध संपले."]
    },
    # Warning
    "roundabout": {
        "en-US": ["Roundabout ahead. Slow down and yield.", "Approaching a circular intersection.", "Ready for the roundabout; follow the traffic flow."],
        "hi-IN": ["आगे गोल चक्कर है। धीरे चलें।", "सर्कुलर चौराहे की ओर बढ़ रहे हैं।", "गोल चक्कर के लिए तैयार रहें।"],
        "mr-IN": ["पुढं गोल रस्ता आहे. वेग कमी करा.", "चौकाकडे जाताना सावध रहा.", "पुढं गोल चौक आहे."]
    },
    "pedestrian crossing": {
        "en-US": ["Pedestrian crossing ahead. Please slow down.", "Watch out for people crossing the road.", "Slow down, pedestrians have the right of way."],
        "hi-IN": ["आगे पैदल यात्री क्रॉसिंग है। कृपया धीरे चलें।", "सड़क पार करने वाले लोगों का ध्यान रखें।", "धीरे चलें, पैदल यात्रियों को रास्ता दें।"],
        "mr-IN": ["पुढं पादचारी क्रॉसिंग आहे. कृपया वेग कमी करा.", "रस्ता ओलांडणाऱ्या लोकांकडे लक्ष द्या.", "सावकाश चालवा, पादचाऱ्यांना वाट द्या."]
    },
    "slippery road": {
        "en-US": ["Caution! Slippery road ahead.", "Drive carefully, the road might be slippery.", "Slow down, risk of skidding on the road ahead."],
        "hi-IN": ["सावधान! आगे फिसलन भरी सड़क है।", "सावधानी से ड्राइव करें, सड़क फिसलन भरी हो सकती है।", "धीरे चलें, आगे सड़क पर फिसलने का खतरा है।"],
        "mr-IN": ["सावधान! पुढं घसरणारा रस्ता आहे.", "काळजीपूर्वक चालवा, रस्ता घसरणारा असू शकतो.", "सावकाश चालवा, पुढं घसरण्याचा धोका आहे."]
    },
    "cross road": {
        "en-US": ["Intersection ahead. Be prepared for crossing traffic.", "Cross road approaching, drive safely.", "Watch for traffic from all sides at the cross road ahead."],
        "hi-IN": ["आगे चौराहा है। आने-जाने वाले ट्रैफिक का ध्यान रखें।", "क्रॉस रोड आ रहा है, सुरक्षित ड्राइव करें।", "आगे चौराहे पर सभी ओर से आने वाले वाहनों पर नज़र रखें।"],
        "mr-IN": ["पुढं चौक आहे. येणाऱ्या-जाणाऱ्या वाहतुकीची काळजी घ्या.", "क्रॉस रोड येत आहे, सुरक्षित चालवा.", "पुढं चौकात सर्व बाजूंनी येणाऱ्या वाहनांकडे लक्ष द्या."]
    },
    "school ahead": {
        "en-US": ["School zone ahead. Slow down and watch for children.", "Be careful, children might be crossing near the school ahead.", "Drive slowly, you are approaching a school area."],
        "hi-IN": ["आगे स्कूल क्षेत्र है। धीरे चलें और बच्चों का ध्यान रखें।", "सावधान रहें, आगे स्कूल के पास बच्चे सड़क पार कर सकते हैं।", "धीरे चलाएं, आप स्कूल क्षेत्र के करीब हैं।"],
        "mr-IN": ["पुढं शाळा आहे. वेग कमी करा आणि मुलांची काळजी घ्या.", "सावध रहा, पुढं शाळेजवळ मुलं रस्ता ओलांडू शकतात.", "सावकाश चालवा, पुढं शाळा क्षेत्र आहे."]
    },
    "petrol pump ahead": {
        "en-US": ["Need fuel? A petrol pump is coming up soon.", "Petrol station ahead.", "Approaching a refueling station."],
        "hi-IN": ["ईंधन चाहिए? आगे पेट्रोल पंप आ रहा है।", "आगे पेट्रोल स्टेशन है।", "ईंधन भरने के स्टेशन के पास पहुँच रहे हैं।"],
        "mr-IN": ["इंधन हवे आहे? पुढं पेट्रोल पंप येत आहे.", "पुढं पेट्रोल स्टेशन आहे.", "पुढं पेट्रोल पंप आहे."]
    },
    "hospital ahead": {
        "en-US": ["Hospital zone! Please maintain silence.", "Approaching a hospital area. Keep the noise down.", "Please drive quietly near the hospital ahead."],
        "hi-IN": ["अस्पताल क्षेत्र! कृपया शांति बनाए रखें।", "अस्पताल के पास पहुँच रहे हैं। हॉर्न न बजाएं।", "आगे अस्पताल है, कृपया शांति से ड्राइव करें।"],
        "mr-IN": ["रुग्णालय क्षेत्र! कृपया शांतता राखा.", "पुढं रुग्णालय आहे. हॉर्न वाजवू नका.", "कृपया रुग्णालयाजवळ शांतता राखा."]
    },
    "speed limit 30": {
        "en-US": ["Speed limit is 30. Better slow down a bit.", "Slow down, the limit is 30 here.", "Please maintain a speed of 30 km per hour."],
        "hi-IN": ["गति सीमा 30 है। कृपया गति कम करें।", "धीरे चलें, यहाँ सीमा 30 की है।", "कृपया 30 किमी प्रति घंटे की गति बनाए रखें।"],
        "mr-IN": ["वेग मर्यादा 30 आहे. कृपया वेग कमी करा.", "सावकाश चालवा, इथं मर्यादा 30 ची आहे.", "कृपया ताशी 30 किमी वेग राखा."]
    }
    # Add more signs if needed, keeping them friendly and advisory.
}

def get_random_phrase(label, lang='en-US'):
    """Returns a random friendly phrase for the given label and language."""
    label = label.lower()
    if label in SIGN_PHRASES and lang in SIGN_PHRASES[label]:
        return random.choice(SIGN_PHRASES[label][lang])
    
    # Fallback if specific label/lang not found
    fallbacks = {
        "en-US": f"{label.replace('_', ' ')} detected.",
        "hi-IN": f"{label.replace('_', ' ')} पहचाना गया।",
        "mr-IN": f"{label.replace('_', ' ')} आढळले आहे।"
    }
    return fallbacks.get(lang, fallbacks["en-US"])

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
async def detect(file: UploadFile = File(...), model: str = "traffic", lang: str = "en-US"):
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
            label = mdl.names[int(box.cls)].replace("_", " ")
            detections.append({
                "label": label,
                "confidence": round(float(box.conf), 3),
                "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                "model": model,
                "phrase": get_random_phrase(label, lang) if model == "traffic" else f"{label} detected"
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
async def detect_both(file: UploadFile = File(...), lang: str = "en-US"):
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
                label = mdl.names[int(box.cls)].replace("_", " ")
                all_detections.append({
                    "label": label,
                    "confidence": round(float(box.conf), 3),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                    "model": model_name,
                    "phrase": get_random_phrase(label, lang) if model_name == "traffic" else f"{label} detected"
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
            lang = payload.get("lang", "en-US")

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
                        "model": model_name,
                        "phrase": get_random_phrase(label, lang) if model_name == "traffic" else f"{label} detected"
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
