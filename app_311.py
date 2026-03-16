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
from PIL import Image, ImageDraw, ImageFont

try:
    import pymysql
except Exception:
    pymysql = None

JAVA_API_BASE = os.getenv("JAVA_API_BASE", "http://localhost:8080/api")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DB = os.getenv("MYSQL_DB", "face_attendance")
FEATURE_BACKFILL_INTERVAL = int(os.getenv("FEATURE_BACKFILL_INTERVAL", "30"))
ATTENDANCE_INTERVAL = int(os.getenv("ATTENDANCE_INTERVAL", "300"))
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.48"))
FACE_MATCH_MARGIN = float(os.getenv("FACE_MATCH_MARGIN", "0.05"))
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
last_java_api_error_time = 0.0
last_backfill_error_time = 0.0
font_cache = {}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return response


def get_cn_font(size):
    if size in font_cache:
        return font_cache[size]
    candidates = [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "msyh.ttc"),
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "simhei.ttf"),
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "simsun.ttc"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size=size, encoding="utf-8")
                font_cache[size] = font
                return font
            except Exception:
                pass
    font = ImageFont.load_default()
    font_cache[size] = font
    return font


def draw_text(frame_bgr, text, org, color, size):
    x, y = int(org[0]), int(org[1])
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), str(text), font=get_cn_font(size), fill=(int(color[2]), int(color[1]), int(color[0])))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


@app.route("/extract_feature", methods=["POST", "OPTIONS"])
def api_extract_feature():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        img_bgr = None
        upload_file = request.files.get("file")
        if upload_file is not None:
            img_bytes = upload_file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
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
        return jsonify({"code": 200, "msg": "success", "data": feature_str, "feature": feature_str})
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
    h, w = img_rgb.shape[:2]
    det = dlib.rectangle(
        max(0, min(det.left(), w - 1)),
        max(0, min(det.top(), h - 1)),
        max(1, min(det.right(), w - 1)),
        max(1, min(det.bottom(), h - 1)),
    )
    if det.right() <= det.left() or det.bottom() <= det.top():
        return None
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


def detect_faces(frame_bgr, min_side=60):
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return []
    if len(frame_bgr.shape) < 2:
        return []
    h, w = frame_bgr.shape[:2]
    if h < min_side or w < min_side:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    try:
        boxes = opencv_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
    except cv2.error:
        return []
    dets = []
    for x, y, w, h in boxes:
        l = int(x)
        t = int(y)
        r = int(x + w - 1)
        b = int(y + h - 1)
        l = max(0, min(l, gray.shape[1] - 1))
        t = max(0, min(t, gray.shape[0] - 1))
        r = max(l + 1, min(r, gray.shape[1] - 1))
        b = max(t + 1, min(b, gray.shape[0] - 1))
        dets.append(dlib.rectangle(l, t, r, b))
    return dets


def extract_feature_from_bgr(img_bgr, require_single_face=False):
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

    dets = detect_faces(img_bgr, 60)
    if len(dets) == 0:
        dets = detect_faces(img_bgr, 30)
    if len(dets) == 0:
        scale = 1.6
        up_w = max(int(w * scale), 1)
        up_h = max(int(h * scale), 1)
        upscaled = cv2.resize(img_bgr, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
        upscaled_dets = detect_faces(upscaled, 60)
        converted = []
        for det in upscaled_dets:
            l = int(det.left() / scale)
            t = int(det.top() / scale)
            r = int(det.right() / scale)
            b = int(det.bottom() / scale)
            l = max(0, min(l, w - 1))
            t = max(0, min(t, h - 1))
            r = max(l + 1, min(r, w - 1))
            b = max(t + 1, min(b, h - 1))
            converted.append(dlib.rectangle(l, t, r, b))
        dets = converted
    if len(dets) == 0 and detector is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)
            dets = detector(gray, 1)
        except Exception:
            dets = []
    if len(dets) == 0 and detector is not None:
        try:
            dets = detector(img_rgb, 1)
        except Exception:
            dets = []
    if len(dets) == 0:
        return None
    if require_single_face and len(dets) != 1:
        return None
    det = max(dets, key=lambda r: r.width() * r.height())
    return get_face_feature(img_rgb, det)


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
    global known_faces_cache, last_update_time, last_java_api_error_time
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
        last_java_api_error_time = 0.0
        print(f"Updated known faces: {len(known_faces_cache)} employees")
    except Exception as e:
        now = time.time()
        if now - last_java_api_error_time >= 30:
            last_java_api_error_time = now
            print(f"Java API unavailable: {JAVA_API_BASE}. Start Java backend on port 8080. detail={e}")


def recognize_face(face_vector):
    best_emp = None
    best_dist = 10.0
    second_dist = 10.0
    for emp in known_faces_cache:
        dist = np.linalg.norm(face_vector - emp["vector"])
        if dist < best_dist:
            second_dist = best_dist
            best_dist = dist
            best_emp = emp
        elif dist < second_dist:
            second_dist = dist
    if best_emp is not None and best_dist < FACE_MATCH_THRESHOLD and (second_dist - best_dist) >= FACE_MATCH_MARGIN:
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


def resolve_photo_path(photo_url):
    if not photo_url:
        return None
    normalized = str(photo_url).replace("/", os.sep).replace("\\", os.sep)
    normalized = os.path.normpath(normalized)
    if os.path.isabs(normalized) and os.path.exists(normalized):
        return normalized
    if normalized.lower().startswith(("data" + os.sep).lower()):
        project_relative = os.path.join(PROJECT_ROOT, normalized)
        if os.path.exists(project_relative):
            return project_relative
    candidates = [
        normalized,
        os.path.join(PROJECT_ROOT, normalized),
        os.path.join(PYTHON_BASE_DIR, normalized),
        os.path.join(DATA_DIR, normalized),
        os.path.join(DATA_DIR, "faces", os.path.basename(normalized)),
        os.path.join(DATA_DIR, os.path.basename(normalized)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def create_mysql_conn():
    if pymysql is None:
        return None
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def backfill_features_once(conn):
    updated = 0
    skipped = 0
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT id, photo_url FROM employee WHERE feature IS NULL OR TRIM(feature) = ''"
        )
        rows = cursor.fetchall()
        for row in rows:
            emp_id = row.get("id")
            photo_url = row.get("photo_url")
            photo_path = resolve_photo_path(photo_url)
            if not photo_path:
                print(f"Backfill skip employee={emp_id}: photo not found, photo_url={photo_url}")
                skipped += 1
                continue
            img_bgr = cv2.imread(photo_path)
            feature = extract_feature_from_bgr(img_bgr, require_single_face=True)
            if feature is None:
                print(f"Backfill skip employee={emp_id}: no valid face, photo_path={photo_path}")
                skipped += 1
                continue
            feature_str = ",".join(map(str, feature.tolist()))
            cursor.execute("UPDATE employee SET feature=%s WHERE id=%s", (feature_str, emp_id))
            updated += 1
    conn.commit()
    return updated, skipped


def feature_backfill_worker():
    global last_backfill_error_time
    if pymysql is None:
        print("PyMySQL not installed. Feature backfill worker disabled.")
        return
    conn = None
    while True:
        try:
            if conn is None:
                conn = create_mysql_conn()
            updated, skipped = backfill_features_once(conn)
            if updated > 0 or skipped > 0:
                print(f"Feature backfill: updated={updated}, skipped={skipped}")
        except Exception as e:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
            now = time.time()
            if now - last_backfill_error_time >= 30:
                last_backfill_error_time = now
                print(f"Feature backfill worker error: {e}")
        time.sleep(max(FEATURE_BACKFILL_INTERVAL, 5))


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
                    if time.time() - last_ts >= ATTENDANCE_INTERVAL:
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
                        frame = draw_text(frame, f"打卡成功：{name}", (10, 10), (0, 255, 0), 32)
                    else:
                        # 已经打过卡，仅显示姓名，不重复记录和截图
                        name = f"{name} (已打卡)"

            cv2.rectangle(frame, (l, t), (l + w, t + h), color, 2)
            frame = draw_text(frame, name, (l, max(0, t - 36)), color, 28)

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
    backfill_thread = threading.Thread(target=feature_backfill_worker, daemon=True)
    backfill_thread.start()
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    run_camera()
