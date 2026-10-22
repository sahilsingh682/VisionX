"""Downloads the AI models (run automatically on first start / during Docker build)."""
import os, urllib.request
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
R = "https://raw.githubusercontent.com/"
FILES = {
    "face.prototxt": R + "opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "face.caffemodel": R + "opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    "ssd.prototxt": R + "chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "ssd.caffemodel": R + "chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel",
    "emotion.onnx": "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx?raw=true",
}

def ensure(name):
    os.makedirs(D, exist_ok=True)
    path = os.path.join(D, name)
    if not (os.path.exists(path) and os.path.getsize(path) > 5000):
        print("downloading", name, flush=True)
        urllib.request.urlretrieve(FILES[name], path + ".part")
        os.replace(path + ".part", path)
    return path

if __name__ == "__main__":
    for n in FILES:
        try: ensure(n)
        except Exception as e: print("FAILED", n, e)
