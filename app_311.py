import base64
import bz2
import os
import threading
import time
from datetime import datetime

import cv2
import dlib
import numpy as np
import requests
from flask import Flask, jsonify, request

JAVA_API_BASE = "http://localhost:8080/api"
PYTHON_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PYTHON_BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PYTHON_BASE_DIR, "models")

DIRS = {
    "captures": os.path.join(DATA_DIR, "captures"),
    "faces": os.path.join(DATA_DIR, "faces"),
    "videos": os.path.join(DATA_DIR, "videos"),
}

MODEL_URLS = {
    "shape_predictor": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "face_recognition": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
}

MODEL_PATHS = {
    "shape_predictor": os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat"),
    "face_recognition": os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat"),
}

for d in [MODELS_DIR] + list(DIRS.values()):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)

detector = None
sp = None
facerec = None
opencv_detector = None
known_faces_cache = []
last_update_time = 0.0
last_attendance = {}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return response


@app.route("/extract_feature", methods=["POST", "OPTIONS"])
def api_extract_feature():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"code": 400, "msg": "No image provided"}), 400
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"code": 400, "msg": "Invalid image"}), 400
        dets = detect_faces(img_bgr)
        if len(dets) == 0:
            return jsonify({"code": 400, "msg": "No face detected"}), 400
        det = max(dets, key=lambda r: r.width() * r.height())
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        feature = get_face_feature(img_rgb, det)
        if feature is None:
            return jsonify({"code": 500, "msg": "Feature extraction failed"}), 500
        feature_str = ",".join(map(str, feature.tolist()))
        return jsonify({"feature": feature_str})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


def download_and_extract_model(url, save_path):
    if os.path.exists(save_path):
        return
    compressed_path = os.path.join(MODELS_DIR, url.split("/")[-1])
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(compressed_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    with bz2.BZ2File(compressed_path) as fr, open(save_path, "wb") as fw:
        fw.write(fr.read())
    os.remove(compressed_path)


def load_models():
    global detector, sp, facerec, opencv_detector
    download_and_extract_model(MODEL_URLS["shape_predictor"], MODEL_PATHS["shape_predictor"])
    download_and_extract_model(MODEL_URLS["face_recognition"], MODEL_PATHS["face_recognition"])
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(MODEL_PATHS["shape_predictor"])
    facerec = dlib.face_recognition_model_v1(MODEL_PATHS["face_recognition"])
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    opencv_detector = cv2.CascadeClassifier(cascade_path)
    if opencv_detector.empty():
        raise RuntimeError("OpenCV cascade load failed")


def get_face_feature(img_rgb, det):
    try:
        shape = sp(img_rgb, det)
        descriptor = facerec.compute_face_descriptor(img_rgb, shape)
        return np.array(descriptor, dtype=np.float64)
    except Exception:
        try:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)
            shape = sp(gray, det)
            descriptor = facerec.compute_face_descriptor(img_rgb, shape)
            return np.array(descriptor, dtype=np.float64)
        except Exception:
            return None


def detect_faces(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    boxes = opencv_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    dets = []
    for x, y, w, h in boxes:
        dets.append(dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
    return dets


def parse_feature(feature_value):
    if feature_value is None:
        return None
    if isinstance(feature_value, list):
        arr = np.array(feature_value, dtype=np.float64)
    else:
        text = str(feature_value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        arr = np.fromstring(text, sep=",", dtype=np.float64)
    if arr.shape[0] != 128:
        return None
    return arr


def update_known_faces():
    global known_faces_cache, last_update_time
    if time.time() - last_update_time < 15:
        return
    try:
        resp = requests.get(f"{JAVA_API_BASE}/employee/features", timeout=3)
        if resp.status_code != 200:
            return
        body = resp.json()
        if body.get("code") != 200:
            return
        employees = body.get("data") or []
        parsed = []
        for emp in employees:
            vec = parse_feature(emp.get("feature"))
            if vec is None:
                continue
            parsed.append({"id": emp.get("id"), "name": emp.get("name", "未知"), "vector": vec})
        known_faces_cache = parsed
        last_update_time = time.time()
        print(f"Updated known faces: {len(known_faces_cache)} employees")
    except Exception as e:
        print(f"Failed to update known faces: {e}")


def recognize_face(face_vector):
    best_emp = None
    best_dist = 10.0
    for emp in known_faces_cache:
        dist = np.linalg.norm(face_vector - emp["vector"])
        if dist < best_dist:
            best_dist = dist
            best_emp = emp
    if best_emp is not None and best_dist < 0.6:
        return best_emp, best_dist
    return None, best_dist


def post_attendance(employee_id):
    payload = {"employeeId": employee_id, "status": 0, "punchTime": datetime.now().isoformat()}
    try:
        requests.post(f"{JAVA_API_BASE}/attendance/record", json=payload, timeout=2)
    except Exception as e:
        print(f"API Error (attendance): {e}")


def post_capture(employee_id, image_url, score):
    payload = {"employeeId": employee_id, "imageUrl": image_url, "score": float(score)}
    try:
        requests.post(f"{JAVA_API_BASE}/capture/save", json=payload, timeout=2)
    except Exception as e:
        print(f"API Error (capture): {e}")


def run_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    recording = False
    video_writer = None
    no_face_since = None
    print("Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        timestamp = datetime.now()
        ts = timestamp.strftime("%Y%m%d_%H%M%S")
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

        dets = detect_faces(frame)

        update_known_faces()

        if len(dets) > 0:
            no_face_since = None
            if not recording:
                video_path = os.path.join(DIRS["videos"], f"{ts}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                recording = True
                print(f"Started recording: {video_path}")
        else:
            if recording:
                if no_face_since is None:
                    no_face_since = time.time()
                elif time.time() - no_face_since >= 5:
                    recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print("Stopped recording (no face for 5s)")

        if recording and video_writer is not None:
            video_writer.write(frame)

        for det in dets:
            l = max(det.left(), 0)
            t = max(det.top(), 0)
            r = min(det.right(), frame.shape[1] - 1)
            b = min(det.bottom(), frame.shape[0] - 1)
            w = max(r - l, 1)
            h = max(b - t, 1)

            feature = get_face_feature(img_rgb, det)
            name = "未知人员"
            color = (0, 0, 255)

            if feature is not None:
                emp, dist = recognize_face(feature)
                if emp is not None and emp.get("id") is not None:
                    emp_id = emp["id"]
                    name = emp.get("name") or "未知"
                    color = (0, 255, 0)
                    last_ts = last_attendance.get(emp_id, 0.0)
                    if time.time() - last_ts >= 3:
                        cap_name = f"{ts}_{emp_id}.jpg"
                        cap_path = os.path.join(DIRS["captures"], cap_name)
                        cv2.imwrite(cap_path, frame)
                        face_name = f"{ts}_{emp_id}_face.jpg"
                        face_path = os.path.join(DIRS["faces"], face_name)
                        face_img = frame[t:b, l:r]
                        if face_img.size > 0:
                            cv2.imwrite(face_path, face_img)
                        post_attendance(emp_id)
                        post_capture(emp_id, f"data/captures/{cap_name}", 1.0 - min(dist, 1.0))
                        last_attendance[emp_id] = time.time()
                        cv2.putText(frame, f"打卡成功：{name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.rectangle(frame, (l, t), (l + w, t + h), color, 2)
            cv2.putText(frame, name, (l, max(20, t - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


def start_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    load_models()
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    run_camera()
