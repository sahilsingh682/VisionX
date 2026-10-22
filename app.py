"""VisionX - video analytics (Flask + OpenCV DNN + ONNX). Run: python app.py"""
import base64, os, threading
from collections import deque
import cv2, numpy as np
from flask import Flask, jsonify, render_template, request
import get_models as gm

app = Flask(__name__)
LOCK = threading.Lock()          # models/trackers are shared state -> one frame at a time
M = {}                           # lazy model cache
HC = cv2.data.haarcascades

# ---------------------------------------------------------------- helpers
def lazy(key, fn):
    if key not in M:
        try: M[key] = fn()
        except Exception as e: print(f"[warn] {key} unavailable: {e}", flush=True); M[key] = None
    return M[key]

def face_net():
    return lazy("face", lambda: cv2.dnn.readNetFromCaffe(gm.ensure("face.prototxt"), gm.ensure("face.caffemodel")))
def ssd_net():
    return lazy("ssd", lambda: cv2.dnn.readNetFromCaffe(gm.ensure("ssd.prototxt"), gm.ensure("ssd.caffemodel")))
def emo_sess():
    def mk():
        import onnxruntime as ort
        o = ort.SessionOptions(); o.intra_op_num_threads = 2
        return ort.InferenceSession(gm.ensure("emotion.onnx"), o, providers=["CPUExecutionProvider"])
    return lazy("emo", mk)
def haar():
    return lazy("haar", lambda: cv2.CascadeClassifier(HC + "haarcascade_frontalface_default.xml"))
def hog():
    def mk():
        h = cv2.HOGDescriptor(); h.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()); return h
    return lazy("hog", mk)
def pit_yolo():
    def mk():
        from ultralytics import YOLO
        return YOLO("models/pits.pt") if os.path.exists("models/pits.pt") else None
    return lazy("pit", mk)

try:
    import pytesseract, shutil
    HAS_OCR = shutil.which("tesseract") is not None
except Exception:
    HAS_OCR = False

def iou(a, b):
    iw = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    ih = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    i = iw * ih; u = a["w"] * a["h"] + b["w"] * b["h"] - i
    return i / u if u > 0 else 0

class Tracker:
    """IoU tracker: stable IDs, smoothed boxes, and a confirmation delay that removes one-frame false alarms."""
    def __init__(self, min_hits=2, max_age=4, thr=0.25, alpha=0.6, hist=6):
        self.min_hits, self.max_age, self.thr, self.alpha, self.hist = min_hits, max_age, thr, alpha, hist
        self.t, self.nid = [], 1
    def update(self, dets):
        pairs = sorted(((iou(t, d), i, j) for i, t in enumerate(self.t) for j, d in enumerate(dets)), reverse=True)
        ut, ud = set(), set()
        for s, i, j in pairs:
            if s < self.thr: break
            if i in ut or j in ud: continue
            ut.add(i); ud.add(j); t, d = self.t[i], dets[j]
            for k in "xywh": t[k] = self.alpha * d[k] + (1 - self.alpha) * t[k]
            t["hits"] += 1; t["miss"] = 0; t["conf"] = d.get("conf", 1); t["label"] = d.get("label", "")
            if d.get("p") is not None: t["hist"].append(d["p"])
        for i, t in enumerate(self.t):
            if i not in ut: t["miss"] += 1
        for j, d in enumerate(dets):
            if j not in ud:
                self.t.append(dict(x=d["x"], y=d["y"], w=d["w"], h=d["h"], hits=1, miss=0, id=self.nid, conf=d.get("conf", 1),
                                   label=d.get("label", ""), hist=deque([d["p"]] if d.get("p") is not None else [], maxlen=self.hist)))
                self.nid += 1
        self.t = [t for t in self.t if t["miss"] <= self.max_age]
        return [t for t in self.t if t["hits"] >= self.min_hits and t["miss"] <= 1]

def mk_trackers():
    return dict(pits=Tracker(3, 4, 0.25, 0.5), emotion=Tracker(2, 5, 0.3, 0.55), ocr=Tracker(1, 2, 0.3, 0.5),
                obj=Tracker(2, 6, 0.25, 0.6), motion=Tracker(3, 3, 0.2, 0.5))
T = mk_trackers()
BG = None
ANN = set()

def reset_state():
    global T, BG, ANN
    T = mk_trackers(); ANN = set()
    BG = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=32, detectShadows=True)
reset_state()

def mkbox(t, label, color=None, id_off=0):
    return {"x": float(t["x"]), "y": float(t["y"]), "w": float(t["w"]), "h": float(t["h"]), "label": label,
            "id": t["id"] + id_off, "color": color}

def det(x, y, w, h, label="", conf=1.0, p=None):
    return dict(x=float(x), y=float(y), w=float(w), h=float(h), label=label, conf=float(conf), p=p)

# ---------------------------------------------------------------- 1. pits
def pits(img):
    H, W = img.shape[:2]; dets = []
    y = pit_yolo()
    if y is not None:
        for b in y(img, verbose=False)[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist(); dets.append(det(x1, y1, x2 - x1, y2 - y1, "Pit", float(b.conf[0])))
    else:  # darker-than-surroundings, compact, high-contrast regions on the road area
        top = int(H * 0.4)
        road = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 0)[top:]
        local = cv2.GaussianBlur(road, (0, 0), max(H, W) / 12)
        m = (cv2.subtract(local, road) > 22).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        cnts = sorted(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, reverse=True)[:10]
        for c in cnts:
            a = cv2.contourArea(c); hull = cv2.contourArea(cv2.convexHull(c))
            x, yy, bw, bh = cv2.boundingRect(c)
            if not (0.002 * H * W < a < 0.2 * H * W) or not hull or not 0.35 < bw / bh < 4: continue
            sol = a / hull
            mask = np.zeros(road.shape, np.uint8); cv2.drawContours(mask, [c], -1, 255, -1)
            ring = cv2.subtract(cv2.dilate(mask, np.ones((15, 15), np.uint8)), mask)
            gap = cv2.mean(road, ring)[0] - cv2.mean(road, mask)[0]
            score = 0.6 * min(1, gap / 60) + 0.4 * sol
            if sol > 0.75 and gap > 20 and score > 0.55:
                dets.append(det(x, yy + top, bw, bh, "Pit", score))
    out = [mkbox(t, f"Pit {t['conf']:.0%}", "#ff9f43") for t in T["pits"].update([(d) for d in dets])]
    info = [f"{len(out)} pit(s) detected" if out else "No pits detected"]
    if y is None: info.append("Mode: image analysis (add models/pits.pt for a trained model)")
    return out, info, [f"{len(out)} pit(s) on road"] if out else [], ""

# ---------------------------------------------------------------- 2. emotion
EMO = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
ECOL = dict(Anger="#ef4444", Contempt="#78716c", Disgust="#84cc16", Fear="#8b5cf6", Happiness="#22c55e",
            Neutral="#14b8a6", Sadness="#3b82f6", Surprise="#f59e0b")
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

def faces(img):
    H, W = img.shape[:2]; net = face_net(); out = []
    if net is not None:
        net.setInput(cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 177, 123)))
        for r in net.forward()[0, 0]:
            if r[2] >= 0.6:
                x1, y1 = max(0, r[3] * W), max(0, r[4] * H); x2, y2 = min(W, r[5] * W), min(H, r[6] * H)
                if x2 - x1 >= 40 and y2 - y1 >= 40: out.append((x1, y1, x2 - x1, y2 - y1, float(r[2])))
    elif haar() is not None:
        g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        out = [(x, y, w, h, .8) for x, y, w, h in haar().detectMultiScale(g, 1.1, 5, minSize=(60, 60))]
    return sorted(out, key=lambda f: -f[2] * f[3])[:6]

def emo_probs(face):
    s = emo_sess()
    if s is None: return None
    x = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    lg = s.run(None, {s.get_inputs()[0].name: x})[0][0]
    p = np.exp(lg - lg.max()); p[5] *= 0.85            # slight de-bias: model over-predicts Neutral
    return p / p.sum()

def emotions(img):
    dets = []
    for x, y, w, h, c in faces(img):
        m = int(0.12 * w); crop = img[int(max(0, y - m)):int(y + h + m), int(max(0, x - m)):int(x + w + m)]
        dets.append((det(x, y, w, h, "Face", c, emo_probs(crop))))
    out, info = [], []
    for t in T["emotion"].update(dets):
        if len(t["hist"]):
            p = np.mean(t["hist"], axis=0); o = np.argsort(p)[::-1]; lab = EMO[o[0]]
            out.append(mkbox(t, f"{lab} {p[o[0]]:.0%}", ECOL[lab]))
            info.append(f"Face {t['id']}: {lab} {p[o[0]]:.0%} · {EMO[o[1]]} {p[o[1]]:.0%}")
        else:
            out.append(mkbox(t, "Face", "#14b8a6")); info.append(f"Face {t['id']} (emotion model unavailable)")
    return out, info or ["No face found"], [], ""

# ---------------------------------------------------------------- 3. OCR
def ocr(img):
    if not HAS_OCR: return [], ["Tesseract not installed - see README"], [], ""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.resize(g, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC))
    def run(im):
        d = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6")
        lines = {}
        for i, t in enumerate(d["text"]):
            t = t.strip(); c = float(d["conf"][i])
            if c >= 60 and any(ch.isalnum() for ch in t):
                lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append((t, c, d["left"][i], d["top"][i], d["width"][i], d["height"][i]))
        return lines
    lines = run(g) or run(cv2.bitwise_not(g))      # retry inverted for light-on-dark text
    dets, texts = [], []
    for ws in lines.values():
        txt = " ".join(w[0] for w in ws)
        if len(txt) < 2 and not txt.isdigit(): continue
        x1 = min(w[2] for w in ws); y1 = min(w[3] for w in ws); x2 = max(w[2] + w[4] for w in ws); y2 = max(w[3] + w[5] for w in ws)
        dets.append((det(x1 / 1.6, y1 / 1.6, (x2 - x1) / 1.6, (y2 - y1) / 1.6, txt, np.mean([w[1] for w in ws]) / 100)))
    dets.sort(key=lambda d: (round(d["y"] / 20), d["x"]))
    out = [mkbox(t, t["label"], "#5b6cff") for t in T["ocr"].update(dets)]
    texts = [b["label"] for b in sorted(out, key=lambda b: (round(b["y"] / 20), b["x"]))]
    return out, texts or ["No text found"], [], "\n".join(texts)

# ---------------------------------------------------------------- 4. surveillance
CLS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
       "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
WATCH = {"person": "#ff5470", "car": "#f59e0b", "bus": "#f59e0b", "motorbike": "#f59e0b", "bicycle": "#f59e0b",
         "cat": "#8b5cf6", "dog": "#8b5cf6", "horse": "#8b5cf6", "cow": "#8b5cf6", "sheep": "#8b5cf6", "bird": "#8b5cf6"}

def objects(img):
    H, W = img.shape[:2]; net = ssd_net(); out = []
    if net is not None:
        net.setInput(cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5))
        for r in net.forward()[0, 0]:
            n = CLS[int(r[1])] if 0 <= int(r[1]) < len(CLS) else ""
            if n in WATCH and r[2] >= 0.5:
                x1, y1, x2, y2 = max(0, r[3] * W), max(0, r[4] * H), min(W, r[5] * W), min(H, r[6] * H)
                out.append((det(x1, y1, x2 - x1, y2 - y1, n, r[2])))
    elif hog() is not None:
        for x, y, w, h in hog().detectMultiScale(img, winStride=(8, 8), scale=1.05)[0]:
            out.append((det(x, y, w, h, "person", .6)))
    return out

def surveil(img):
    H, W = img.shape[:2]; out, alerts = [], []
    objs = T["obj"].update(objects(img)); people = 0; other = 0
    for t in objs:
        n = t["label"]; people += n == "person"; other += n != "person"
        out.append(mkbox(t, f"{n.capitalize()} #{t['id']} {t['conf']:.0%}", WATCH.get(n, "#ff5470")))
        if t["id"] not in ANN and t["hits"] >= 2:
            ANN.add(t["id"]); alerts.append(f"{n.capitalize()} #{t['id']} detected")
    fg = BG.apply(img)
    fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]       # drop shadows (127)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); fg = cv2.dilate(fg, np.ones((9, 9), np.uint8), iterations=2)
    pct, mdets = 0, []
    if cv2.countNonZero(fg) < 0.5 * H * W:                        # ignore global lighting/exposure jumps
        pct = min(100, cv2.countNonZero(fg) * 100 // (H * W))
        for c in sorted(cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(c) > 0.008 * H * W:
                x, y, w, h = cv2.boundingRect(c); d = (det(x, y, w, h, "Motion", .7))
                if not any(iou(d, o) > 0 and (min(d["x"] + d["w"], o["x"] + o["w"]) - max(d["x"], o["x"])) *
                           (min(d["y"] + d["h"], o["y"] + o["h"]) - max(d["y"], o["y"])) > 0.3 * d["w"] * d["h"] for o in objs):
                    mdets.append(d)
    motion = T["motion"].update(mdets)
    for t in motion: out.append(mkbox(t, "Motion", "#38bdf8", 100000))
    if motion and not objs: alerts.append("Motion detected")
    return out, [f"People: {people}", f"Other objects: {other}", f"Motion: {pct}%",
                 "Status: " + ("ALERT" if people else "Motion" if motion else "Clear")], alerts, ""

FEATURES = {"pits": pits, "emotion": emotions, "ocr": ocr, "surveil": surveil}

@app.route("/")
def index(): return render_template("index.html")

@app.post("/api/reset")
def reset():
    with LOCK: reset_state()
    return jsonify(ok=True)

@app.post("/api/<feature>")
def analyze(feature):
    fn = FEATURES.get(feature)
    if not fn: return jsonify(error="unknown feature"), 404
    try:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(request.json["image"].split(",")[-1]), np.uint8), cv2.IMREAD_COLOR)
        if img is None: raise ValueError("bad image")
        with LOCK: b, info, alerts, text = fn(img)
        return jsonify(boxes=b, info=info, alerts=alerts, text=text)
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
