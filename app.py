"""VisionX - video analytics website (Flask + OpenCV). Run: python app.py"""
import base64, os, shutil
import cv2, numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
HC = cv2.data.haarcascades
FACE = cv2.CascadeClassifier(HC + "haarcascade_frontalface_default.xml")
SMILE = cv2.CascadeClassifier(HC + "haarcascade_smile.xml")
EYES = cv2.CascadeClassifier(HC + "haarcascade_eye.xml")
HOG = cv2.HOGDescriptor(); HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
BG = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)

try:
    from deepface import DeepFace
except Exception:
    DeepFace = None
try:
    import pytesseract
    HAS_OCR = shutil.which("tesseract") is not None
except Exception:
    HAS_OCR = False
PIT_YOLO = None
if os.path.exists("models/pits.pt"):
    try:
        from ultralytics import YOLO
        PIT_YOLO = YOLO("models/pits.pt")
    except Exception:
        PIT_YOLO = None

def box(x, y, w, h, label):
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "label": label}

def pits(img):
    h, w = img.shape[:2]
    out = []
    if PIT_YOLO is not None:
        for b in PIT_YOLO(img, verbose=False)[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(box(x1, y1, x2 - x1, y2 - y1, f"Pit {float(b.conf[0]):.0%}"))
    else:  # classical fallback: dark, compact blobs on the road area (lower 60%)
        top = int(h * 0.4)
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0)
        m = cv2.adaptiveThreshold(g[top:], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 51, 12)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a = cv2.contourArea(c)
            if not (0.002 * h * w < a < 0.15 * h * w):
                continue
            hull = cv2.contourArea(cv2.convexHull(c))
            x, y, bw, bh = cv2.boundingRect(c)
            sol = a / hull if hull else 0
            if sol > 0.8 and 0.4 < bw / bh < 4:
                out.append(box(x, y + top, bw, bh, f"Pit {sol:.0%}"))
    info = [f"{len(out)} pit(s) detected" + ("" if PIT_YOLO else " (heuristic mode)")]
    return out, info, (f"{len(out)} pit(s) on road" if out else None)

def emotions(img):
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    out, info = [], []
    for (x, y, w, h) in FACE.detectMultiScale(gray, 1.2, 5, minSize=(60, 60)):
        if DeepFace:
            try:
                r = DeepFace.analyze(img[y:y+h, x:x+w], actions=["emotion"],
                                     enforce_detection=False, silent=True)
                r = r[0] if isinstance(r, list) else r
                e = r["dominant_emotion"].capitalize()
            except Exception:
                e = "Unknown"
        else:  # basic mode
            roi = gray[y:y+h, x:x+w]
            sm = SMILE.detectMultiScale(roi[h//2:], 1.7, 22, minSize=(w//4, h//10))
            eyes = EYES.detectMultiScale(roi[:h//2], 1.1, 6)
            e = "Happy" if len(sm) else "Neutral"
        out.append(box(x, y, w, h, e)); info.append(f"Face {len(out)}: {e}")
    return out, info or ["No face found"], None

def ocr(img):
    if not HAS_OCR:
        return [], ["Tesseract not installed - see README"], None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    d = pytesseract.image_to_data(g, output_type=pytesseract.Output.DICT, config="--psm 11")
    out, words = [], []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if t and float(d["conf"][i]) > 50:
            out.append(box(d["left"][i]/1.5, d["top"][i]/1.5, d["width"][i]/1.5, d["height"][i]/1.5, t))
            words.append(t)
    return out, [" ".join(words)] if words else ["No text found"], None

def surveil(img):
    mask = BG.apply(img)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    h, w = img.shape[:2]
    out, moved = [], 0
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) > 0.003 * h * w:
            x, y, bw, bh = cv2.boundingRect(c); moved += bw * bh
            out.append(box(x, y, bw, bh, "Motion"))
    rects, _ = HOG.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    for (x, y, bw, bh) in rects:
        out.append(box(x, y, bw, bh, "Person"))
    info = [f"Motion: {min(100, moved * 100 // (h * w))}%", f"People: {len(rects)}"]
    alert = "Person detected" if len(rects) else ("Motion detected" if out else None)
    return out, info, alert

FEATURES = {"pits": pits, "emotion": emotions, "ocr": ocr, "surveil": surveil}

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/api/reset")
def reset():
    global BG
    BG = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
    return jsonify(ok=True)

@app.post("/api/<feature>")
def analyze(feature):
    fn = FEATURES.get(feature)
    if not fn:
        return jsonify(error="unknown feature"), 404
    try:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(request.json["image"].split(",")[-1]), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("bad image")
        b, info, alert = fn(img)
        return jsonify(boxes=b, info=info, alert=alert)
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
