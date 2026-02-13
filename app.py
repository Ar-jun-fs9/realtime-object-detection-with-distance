# from flask import Flask, Response, render_template, request, jsonify
# import cv2
# from ultralytics import YOLO
# import base64
# import io
# import numpy as np
# import pyttsx3
# import threading
# import time
# from deep_sort_realtime.deepsort_tracker import DeepSort

# app = Flask(__name__)
# model = YOLO("yolov10s.pt")
# tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# # Real-world heights in meters for common objects
# REAL_HEIGHTS = {
#     "person": 1.65,
#     "car": 1.5,
#     "truck": 2.5,
#     "bus": 3.0,
#     "bicycle": 1.0,
#     "motorcycle": 1.2,
#     "dog": 0.5,
#     "cat": 0.3,
#     "chair": 0.9,
#     "dining table": 0.75,
#     "couch": 0.8,
#     "tv": 0.6,
#     "laptop": 0.3,
#     "mouse": 0.05,
#     "keyboard": 0.15,
#     "cell phone": 0.15,
#     "book": 0.2,
#     "bottle": 0.25,
#     "cup": 0.1,
#     "fork": 0.15,
#     "knife": 0.2,
#     "spoon": 0.15,
#     "bowl": 0.1,
#     "banana": 0.2,
#     "apple": 0.08,
#     "sandwich": 0.1,
#     "orange": 0.08,
#     "broccoli": 0.2,
#     "carrot": 0.15,
#     "hot dog": 0.15,
#     "pizza": 0.05,
#     "donut": 0.08,
#     "cake": 0.1,
#     "potted plant": 0.3,
#     "bed": 0.6,
#     "toilet": 0.4,
#     "sink": 0.9,
#     "refrigerator": 1.8,
#     "oven": 0.9,
#     "microwave": 0.3,
#     "toaster": 0.2,
#     "hair drier": 0.2,
#     "toothbrush": 0.15,
# }

# FOCAL_LENGTH = 550  # Approximate focal length in pixels for 640x480 resolution webcam (calibrate for accuracy)


# def speak(text):
#     def _speak():
#         engine = pyttsx3.init()
#         engine.say(text)
#         engine.runAndWait()

#     threading.Thread(target=_speak, daemon=True).start()


# def iou(box1, box2):
#     """Calculate Intersection over Union (IoU) of two bounding boxes."""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
#     inter = max(0, x2 - x1) * max(0, y2 - y1)
#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union = area1 + area2 - inter
#     return inter / union if union > 0 else 0


# def calculate_distance(class_name, pixel_height):
#     real_height = REAL_HEIGHTS.get(class_name.lower(), 1.0)  # Default to 1m if unknown
#     if pixel_height > 0:
#         distance = (real_height * FOCAL_LENGTH) / pixel_height
#         return round(float(distance), 2)
#     return None


# def generate_frames():
#     global current_detections, track_distances, track_prev_bbox
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model(frame)
#         annotated_frame = results[0].plot()

#         # Prepare detections for DeepSORT
#         detections = []
#         yolo_detections = []
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             conf = float(box.conf)
#             class_id = int(box.cls)
#             detections.append(
#                 [[x1, y1, x2, y2], conf, class_id]
#             )  # bbox as [left, top, right, bottom], conf, class_id
#             yolo_detections.append(
#                 {
#                     "bbox": [x1, y1, x2, y2],
#                     "class_id": class_id,
#                     "class_name": model.names[class_id],
#                     "conf": conf,
#                 }
#             )

#         # Update tracker
#         tracks = tracker.update_tracks(detections, frame=frame)

#         # Associate tracks with classes and calculate distances
#         current_detections = []
#         current_track_ids = set()
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
#             track_id = track.track_id
#             current_track_ids.add(track_id)
#             ltrb = track.to_ltrb()
#             track_bbox = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
#             # Find best matching yolo detection
#             best_iou = 0
#             best_det = None
#             for det in yolo_detections:
#                 iou_val = iou(track_bbox, det["bbox"])
#                 if iou_val > best_iou:
#                     best_iou = iou_val
#                     best_det = det
#             if best_det:
#                 class_name = best_det["class_name"]
#                 pixel_height = best_det["bbox"][3] - best_det["bbox"][1]
#                 # Check movement to stabilize distance
#                 if track_id in track_prev_bbox:
#                     prev_bbox = track_prev_bbox[track_id]
#                     center_current = (
#                         (track_bbox[0] + track_bbox[2]) / 2,
#                         (track_bbox[1] + track_bbox[3]) / 2,
#                     )
#                     center_prev = (
#                         (prev_bbox[0] + prev_bbox[2]) / 2,
#                         (prev_bbox[1] + prev_bbox[3]) / 2,
#                     )
#                     movement = (
#                         (center_current[0] - center_prev[0]) ** 2
#                         + (center_current[1] - center_prev[1]) ** 2
#                     ) ** 0.5
#                     if movement < 10:  # threshold for stability
#                         distance = track_distances.get(
#                             track_id, calculate_distance(class_name, pixel_height)
#                         )
#                     else:
#                         distance = calculate_distance(class_name, pixel_height)
#                         track_distances[track_id] = distance
#                 else:
#                     distance = calculate_distance(class_name, pixel_height)
#                     track_distances[track_id] = distance
#                 track_prev_bbox[track_id] = track_bbox
#                 current_detections.append(
#                     {
#                         "class": class_name,
#                         "confidence": best_det["conf"],
#                         "distance": distance,
#                         "id": track_id,
#                     }
#                 )
#         # Clean up old tracks
#         track_distances = {
#             k: v for k, v in track_distances.items() if k in current_track_ids
#         }
#         track_prev_bbox = {
#             k: v for k, v in track_prev_bbox.items() if k in current_track_ids
#         }

#         ret, buffer = cv2.imencode(".jpg", annotated_frame)
#         frame_bytes = buffer.tobytes()
#         yield (
#             b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
#         )


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/video_feed")
# def video_feed():
#     return Response(
#         generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
#     )


# current_detections = []
# last_spoken = 0
# track_distances = {}
# track_prev_bbox = {}


# @app.route("/detections")
# def get_detections():
#     global last_spoken
#     if current_detections and time.time() - last_spoken > 5:
#         speak_text = ", ".join(
#             [
#                 f"{d['class']} #{d['id']} approximately {d['distance']} meters away"
#                 for d in current_detections
#                 if d["distance"]
#             ]
#         )
#         if speak_text:
#             speak(speak_text)
#         last_spoken = time.time()
#     return jsonify(current_detections)


# @app.route("/upload", methods=["POST"])
# def upload():
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#     if file:
#         # Read image
#         img_bytes = file.read()
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         nparr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         # Run detection
#         results = model(nparr)
#         annotated_frame = results[0].plot()
#         # Encode to base64
#         _, buffer = cv2.imencode(".jpg", annotated_frame)
#         img_base64 = base64.b64encode(buffer).decode("utf-8")
#         # Get detections
#         detections = []
#         for box in results[0].boxes:
#             class_name = model.names[int(box.cls)]
#             pixel_height = float(box.xyxy[0][3] - box.xyxy[0][1])  # y2 - y1
#             distance = calculate_distance(class_name, pixel_height)
#             detections.append(
#                 {
#                     "class": class_name,
#                     "confidence": float(box.conf),
#                     "distance": distance,
#                 }
#             )

#         return jsonify(
#             {"image": f"data:image/jpeg;base64,{img_base64}", "detections": detections}
#         )


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

# from flask import Flask, Response, render_template, request, jsonify
# import cv2
# from ultralytics import YOLO
# import base64
# import numpy as np
# import pyttsx3
# import threading
# import time
# import os

# app = Flask(__name__)
# model = YOLO("yolov10s.pt")

# # Real-world heights in meters for common objects
# REAL_HEIGHTS = {
#     "person": 1.65,
#     "car": 1.5,
#     "truck": 2.5,
#     "bus": 3.0,
#     "bicycle": 1.0,
#     "motorcycle": 1.2,
#     "dog": 0.5,
#     "cat": 0.3,
#     "chair": 0.9,
#     "dining table": 0.75,
#     "couch": 0.8,
#     "tv": 0.6,
#     "laptop": 0.3,
#     "mouse": 0.05,
#     "keyboard": 0.15,
#     "cell phone": 0.15,
#     "book": 0.2,
#     "bottle": 0.25,
#     "cup": 0.1,
#     "fork": 0.15,
#     "knife": 0.2,
#     "spoon": 0.15,
#     "bowl": 0.1,
#     "banana": 0.2,
#     "apple": 0.08,
#     "sandwich": 0.1,
#     "orange": 0.08,
#     "broccoli": 0.2,
#     "carrot": 0.15,
#     "hot dog": 0.15,
#     "pizza": 0.05,
#     "donut": 0.08,
#     "cake": 0.1,
#     "potted plant": 0.3,
#     "bed": 0.6,
#     "toilet": 0.4,
#     "sink": 0.9,
#     "refrigerator": 1.8,
#     "oven": 0.9,
#     "microwave": 0.3,
#     "toaster": 0.2,
#     "hair drier": 0.2,
#     "toothbrush": 0.15,
# }

# FOCAL_LENGTH = 550  # Approximate focal length in pixels for 640x480 webcam

# # Globals for tracking
# current_detections = []
# last_spoken = 0
# track_distances = {}
# track_prev_bbox = {}
# tracker = None  # Initialize tracker later to avoid pkg_resources error


# def speak(text):
#     def _speak():
#         engine = pyttsx3.init()
#         engine.say(text)
#         engine.runAndWait()

#     threading.Thread(target=_speak, daemon=True).start()


# def iou(box1, box2):
#     """Calculate Intersection over Union (IoU) of two bounding boxes."""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
#     inter = max(0, x2 - x1) * max(0, y2 - y1)
#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union = area1 + area2 - inter
#     return inter / union if union > 0 else 0


# def calculate_distance(class_name, pixel_height):
#     real_height = REAL_HEIGHTS.get(class_name.lower(), 1.0)  # default 1m
#     if pixel_height > 0:
#         distance = (real_height * FOCAL_LENGTH) / pixel_height
#         return round(float(distance), 2)
#     return None


# def init_tracker():
#     """Initialize DeepSort tracker (import delayed to avoid pkg_resources error)."""
#     global tracker
#     from deep_sort_realtime.deepsort_tracker import DeepSort
#     tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)


# def generate_frames():
#     global current_detections, track_distances, track_prev_bbox, tracker

#     # Initialize tracker once
#     if tracker is None:
#         init_tracker()

#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame)
#         annotated_frame = results[0].plot()

#         # Prepare detections for DeepSORT
#         detections = []
#         yolo_detections = []
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             conf = float(box.conf)
#             class_id = int(box.cls)
#             detections.append([[x1, y1, x2, y2], conf, class_id])
#             yolo_detections.append({
#                 "bbox": [x1, y1, x2, y2],
#                 "class_id": class_id,
#                 "class_name": model.names[class_id],
#                 "conf": conf,
#             })

#         # Update tracker
#         tracks = tracker.update_tracks(detections, frame=frame)

#         # Associate tracks with classes and calculate distances
#         current_detections = []
#         current_track_ids = set()
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
#             track_id = track.track_id
#             current_track_ids.add(track_id)
#             ltrb = track.to_ltrb()
#             track_bbox = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]

#             # Match with YOLO detections
#             best_iou_val = 0
#             best_det = None
#             for det in yolo_detections:
#                 iou_val = iou(track_bbox, det["bbox"])
#                 if iou_val > best_iou_val:
#                     best_iou_val = iou_val
#                     best_det = det

#             if best_det:
#                 class_name = best_det["class_name"]
#                 pixel_height = best_det["bbox"][3] - best_det["bbox"][1]

#                 # Stable distance calculation
#                 if track_id in track_prev_bbox:
#                     prev_bbox = track_prev_bbox[track_id]
#                     center_current = (
#                         (track_bbox[0] + track_bbox[2]) / 2,
#                         (track_bbox[1] + track_bbox[3]) / 2,
#                     )
#                     center_prev = (
#                         (prev_bbox[0] + prev_bbox[2]) / 2,
#                         (prev_bbox[1] + prev_bbox[3]) / 2,
#                     )
#                     movement = ((center_current[0] - center_prev[0])**2 +
#                                 (center_current[1] - center_prev[1])**2)**0.5
#                     if movement < 10:
#                         distance = track_distances.get(
#                             track_id, calculate_distance(class_name, pixel_height)
#                         )
#                     else:
#                         distance = calculate_distance(class_name, pixel_height)
#                         track_distances[track_id] = distance
#                 else:
#                     distance = calculate_distance(class_name, pixel_height)
#                     track_distances[track_id] = distance

#                 track_prev_bbox[track_id] = track_bbox
#                 current_detections.append({
#                     "class": class_name,
#                     "confidence": best_det["conf"],
#                     "distance": distance,
#                     "id": track_id,
#                 })

#         # Clean up old tracks
#         track_distances = {k: v for k, v in track_distances.items() if k in current_track_ids}
#         track_prev_bbox = {k: v for k, v in track_prev_bbox.items() if k in current_track_ids}

#         # Encode frame
#         ret, buffer = cv2.imencode(".jpg", annotated_frame)
#         frame_bytes = buffer.tobytes()
#         yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/video_feed")
# def video_feed():
#     return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# @app.route("/detections")
# def get_detections():
#     global last_spoken
#     if current_detections and time.time() - last_spoken > 5:
#         speak_text = ", ".join([
#             f"{d['class']} #{d['id']} approximately {d['distance']} meters away"
#             for d in current_detections if d["distance"]
#         ])
#         if speak_text:
#             speak(speak_text)
#         last_spoken = time.time()
#     return jsonify(current_detections)


# @app.route("/upload", methods=["POST"])
# def upload():
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     img_bytes = file.read()
#     nparr = np.frombuffer(img_bytes, np.uint8)
#     nparr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     results = model(nparr)
#     annotated_frame = results[0].plot()

#     _, buffer = cv2.imencode(".jpg", annotated_frame)
#     img_base64 = base64.b64encode(buffer).decode("utf-8")

#     detections = []
#     for box in results[0].boxes:
#         class_name = model.names[int(box.cls)]
#         pixel_height = float(box.xyxy[0][3] - box.xyxy[0][1])
#         distance = calculate_distance(class_name, pixel_height)
#         detections.append({
#             "class": class_name,
#             "confidence": float(box.conf),
#             "distance": distance
#         })

#     return jsonify({"image": f"data:image/jpeg;base64,{img_base64}", "detections": detections})


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))  # Use Render-assigned PORT
#     app.run(host="0.0.0.0", port=port, debug=True)


from flask import Flask, Response, render_template, request, jsonify
import cv2
import base64
import numpy as np
import pyttsx3
import threading
import time
import os

app = Flask(__name__)

# Globals
model = None  # YOLO model will be lazy-loaded
tracker = None  # DeepSort tracker will be lazy-loaded
current_detections = []
last_spoken = 0
track_distances = {}
track_prev_bbox = {}

# Real-world heights in meters for common objects
REAL_HEIGHTS = {
    "person": 1.65,
    "car": 1.5,
    "truck": 2.5,
    "bus": 3.0,
    "bicycle": 1.0,
    "motorcycle": 1.2,
    "dog": 0.5,
    "cat": 0.3,
    "chair": 0.9,
    "dining table": 0.75,
    "couch": 0.8,
    "tv": 0.6,
    "laptop": 0.3,
    "mouse": 0.05,
    "keyboard": 0.15,
    "cell phone": 0.15,
    "book": 0.2,
    "bottle": 0.25,
    "cup": 0.1,
    "fork": 0.15,
    "knife": 0.2,
    "spoon": 0.15,
    "bowl": 0.1,
    "banana": 0.2,
    "apple": 0.08,
    "sandwich": 0.1,
    "orange": 0.08,
    "broccoli": 0.2,
    "carrot": 0.15,
    "hot dog": 0.15,
    "pizza": 0.05,
    "donut": 0.08,
    "cake": 0.1,
    "potted plant": 0.3,
    "bed": 0.6,
    "toilet": 0.4,
    "sink": 0.9,
    "refrigerator": 1.8,
    "oven": 0.9,
    "microwave": 0.3,
    "toaster": 0.2,
    "hair drier": 0.2,
    "toothbrush": 0.15,
}

FOCAL_LENGTH = 550  # Approximate focal length in pixels

# -------------------------
# Helper functions
# -------------------------


def get_model():
    """Lazy-load YOLO model to avoid memory issues."""
    global model
    if model is None:
        from ultralytics import YOLO

        model = YOLO("yolov10n.pt")  # Use nano model for CPU
    return model


def init_tracker():
    """Lazy-load DeepSort tracker."""
    global tracker
    if tracker is None:
        from deep_sort_realtime.deepsort_tracker import DeepSort

        tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    return tracker


def speak(text):
    """Speak text asynchronously."""

    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=_speak, daemon=True).start()


def iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def calculate_distance(class_name, pixel_height):
    real_height = REAL_HEIGHTS.get(class_name.lower(), 1.0)
    if pixel_height > 0:
        distance = (real_height * FOCAL_LENGTH) / pixel_height
        return round(float(distance), 2)
    return None


# -------------------------
# Flask Routes
# -------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Image upload endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = get_model()(img)  # Lazy-loaded YOLO
    annotated_frame = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated_frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    detections = []
    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls)]
        pixel_height = float(box.xyxy[0][3] - box.xyxy[0][1])
        distance = calculate_distance(class_name, pixel_height)
        detections.append(
            {"class": class_name, "confidence": float(box.conf), "distance": distance}
        )

    return jsonify(
        {"image": f"data:image/jpeg;base64,{img_base64}", "detections": detections}
    )


# -------------------------
# Run Flask
# -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render-assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
