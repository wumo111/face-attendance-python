import cv2
import dlib
import numpy as np
import requests
import os
import time
import threading
import base64
import json
import bz2
from flask import Flask, request, jsonify
from datetime import datetime

# --- Configuration ---
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
    "face_recognition": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
}

MODEL_PATHS = {
    "shape_predictor": os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat"),
    "face_recognition": os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")
}

# Ensure directories exist
for d in [MODELS_DIR] + list(DIRS.values()):
    os.makedirs(d, exist_ok=True)

# --- Flask App ---
app = Flask(__name__)

# --- Global State ---
detector = None
sp = None
facerec = None
opencv_detector = None
known_faces_cache = [] # List of {id, name, feature_vector}
last_update_time = 0
last_attendance = {} # {employee_id: timestamp}
dlib_feature_available = True
dlib_feature_error = ""

# --- Helper Functions ---

def download_and_extract_model(url, save_path):
    if os.path.exists(save_path):
        return
    
    print(f"Downloading model from {url}...")
    filename = url.split('/')[-1]
    compressed_path = os.path.join(MODELS_DIR, filename)
    
    # Download
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(compressed_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
        
        # Extract
        print(f"Extracting {filename}...")
        with bz2.BZ2File(compressed_path) as fr, open(save_path, 'wb') as fw:
            fw.write(fr.read())
        print(f"Extracted to {save_path}")
        
        # Cleanup
        os.remove(compressed_path)
        
    except Exception as e:
        print(f"Error downloading/extracting model: {e}")
        # If download fails, we can't proceed with that model
        pass

def load_models():
    global detector, sp, facerec, opencv_detector
    
    # Check and download
    download_and_extract_model(MODEL_URLS["shape_predictor"], MODEL_PATHS["shape_predictor"])
    download_and_extract_model(MODEL_URLS["face_recognition"], MODEL_PATHS["face_recognition"])
    
    if not os.path.exists(MODEL_PATHS["shape_predictor"]) or not os.path.exists(MODEL_PATHS["face_recognition"]):
        print("CRITICAL ERROR: Models not found and download failed.")
        return False

    print("Loading models...")
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(MODEL_PATHS["shape_predictor"])
    facerec = dlib.face_recognition_model_v1(MODEL_PATHS["face_recognition"])
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    opencv_detector = cv2.CascadeClassifier(cascade_path)
    if opencv_detector.empty():
        print("CRITICAL ERROR: OpenCV face detector load failed.")
        return False
    print("Models loaded.")
    return True

def get_face_feature(img_rgb, det):
    global dlib_feature_available, dlib_feature_error
    if not dlib_feature_available:
        return None
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    try:
        shape = sp(img_rgb, det)
        face_chip = dlib.get_face_chip(img_rgb, shape)
        face_descriptor = facerec.compute_face_descriptor(face_chip)
        return np.array(face_descriptor)
    except Exception as e:
        dlib_feature_available = False
        dlib_feature_error = str(e)
        print(f"dlib feature extractor disabled: {dlib_feature_error}")
        return None

def update_known_faces():
    """Fetch employee features from Java backend"""
    global known_faces_cache, last_update_time
    # Update every 60 seconds or if empty
    if time.time() - last_update_time < 60 and known_faces_cache:
        return

    try:
        resp = requests.get(f"{JAVA_API_BASE}/employee/features", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == 200:
                employees = data.get("data", [])
                new_cache = []
                for emp in employees:
                    try:
                        # feature string "0.1,0.2,..." to numpy array
                        feat_str = emp.get("feature")
                        if feat_str:
                            feat_vec = np.fromstring(feat_str, sep=',')
                            if feat_vec.shape[0] == 128:
                                new_cache.append({
                                    "id": emp["id"],
                                    "name": emp["name"],
                                    "vector": feat_vec
                                })
                    except Exception as e:
                        print(f"Error parsing feature for emp {emp.get('id')}: {e}")
                known_faces_cache = new_cache
                last_update_time = time.time()
                print(f"Updated known faces: {len(known_faces_cache)} employees")
    except Exception as e:
        print(f"Failed to update known faces: {e}")

def recognize_face(face_vector):
    """Compare face vector with known faces"""
    min_dist = 1.0
    match_emp = None
    
    for emp in known_faces_cache:
        dist = np.linalg.norm(face_vector - emp["vector"])
        if dist < min_dist:
            min_dist = dist
            match_emp = emp
            
    if min_dist < 0.6: # Threshold
        return match_emp, min_dist
    return None, min_dist

def detect_faces(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    boxes = opencv_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    dets = []
    for x, y, w, h in boxes:
        dets.append(dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
    return dets

# --- Flask Routes ---

@app.route('/extract_feature', methods=['POST'])
def api_extract_feature():
    try:
        data = request.json
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"code": 400, "msg": "No image provided"}), 400
            
        # Decode image
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"code": 400, "msg": "Invalid image"}), 400
            
        dets = detect_faces(img)
        
        if len(dets) == 0:
            return jsonify({"code": 400, "msg": "No face detected"}), 400
            
        # Extract feature of the largest face
        det = max(dets, key=lambda r: r.width() * r.height())
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feature = get_face_feature(img_rgb, det)
        if feature is None:
            return jsonify({"code": 503, "msg": f"dlib unavailable: {dlib_feature_error}"}), 503
        
        # Convert to comma-separated string
        feature_str = ",".join(map(str, feature.tolist()))
        
        return jsonify({
            "code": 200, 
            "msg": "success", 
            "data": {"feature": feature_str}
        })
        
    except Exception as e:
        print(f"Error in extract_feature: {e}")
        return jsonify({"code": 500, "msg": str(e)}), 500

# --- Main Camera Loop ---

def run_camera():
    global last_attendance
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Video recording state
    recording = False
    video_writer = None
    no_face_start_time = None
    dlib_error_notified = False
    
    print("Camera started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Ensure frame is valid and has correct type (uint8)
        if frame is None or frame.size == 0:
            continue
            
        # Convert to standard format if needed (OpenCV usually returns BGR uint8)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        dets = detect_faces(frame)
        
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        
        # Update known faces periodically
        update_known_faces()
        
        has_face = len(dets) > 0
        
        # Video Recording Logic
        if has_face:
            no_face_start_time = None
            if not recording:
                video_path = os.path.join(DIRS["videos"], f"{timestamp_str}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                recording = True
                print(f"Started recording: {video_path}")
        else:
            if recording:
                if no_face_start_time is None:
                    no_face_start_time = time.time()
                elif time.time() - no_face_start_time > 5:
                    recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    print("Stopped recording (no face for 5s)")
        
        if recording and video_writer:
            video_writer.write(frame)
            
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

        # Processing Faces
        for det in dets:
            x, y, w, h = det.left(), det.top(), det.width(), det.height()
            
            feature = get_face_feature(img_rgb, det)
            if feature is None:
                if not dlib_error_notified:
                    print(f"dlib unavailable, recognition skipped: {dlib_feature_error}")
                    dlib_error_notified = True
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, "DlibError", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                continue
            
            # Recognition
            match_emp, dist = recognize_face(feature)
            
            name = "Unknown"
            color = (0, 0, 255) # Red
            
            if match_emp:
                name = match_emp["name"]
                color = (0, 255, 0) # Green
                emp_id = match_emp["id"]
                
                # Debounce Attendance (3 seconds)
                last_time = last_attendance.get(emp_id, 0)
                if time.time() - last_time > 3:
                    print(f"Match: {name} (ID: {emp_id}, Dist: {dist:.2f})")
                    
                    # 1. Save Capture
                    cap_filename = f"{timestamp_str}_{emp_id}.jpg"
                    cap_path = os.path.join(DIRS["captures"], cap_filename)
                    cv2.imwrite(cap_path, frame)
                    
                    # 2. Save Face Crop
                    face_filename = f"{timestamp_str}_{emp_id}_face.jpg"
                    face_path = os.path.join(DIRS["faces"], face_filename)
                    try:
                        face_img = frame[max(0,y):min(frame.shape[0],y+h), max(0,x):min(frame.shape[1],x+w)]
                        cv2.imwrite(face_path, face_img)
                    except:
                        pass

                    # 3. Call Java API: Attendance Record
                    try:
                        requests.post(f"{JAVA_API_BASE}/attendance/record", json={
                            "employeeId": emp_id,
                            "status": 0 # Normal
                        }, timeout=1)
                    except Exception as e:
                        print(f"API Error (attendance): {e}")

                    # 4. Call Java API: Save Capture
                    try:
                        requests.post(f"{JAVA_API_BASE}/capture/save", json={
                            "employeeId": emp_id,
                            "imageUrl": f"data/captures/{cap_filename}", # Relative path
                            "score": float(1 - dist) # Rough confidence score
                        }, timeout=1)
                    except Exception as e:
                        print(f"API Error (capture): {e}")
                        
                    last_attendance[emp_id] = time.time()
                    
                    # Show Message
                    cv2.putText(frame, f"Success: {name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show Frame
        if not dlib_feature_available:
            cv2.putText(frame, "dlib unavailable", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Face Attendance System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

def start_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    if load_models():
        # Start Flask in a separate thread
        t = threading.Thread(target=start_flask, daemon=True)
        t.start()
        
        # Run Camera in Main Thread
        try:
            run_camera()
        except KeyboardInterrupt:
            pass
