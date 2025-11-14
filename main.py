from ultralytics import YOLO
import cv2


def run_yolo_detection():
    # Load YOLOv8 model (pretrained on COCO dataset)
    model = YOLO("yolov10s.pt")

    # Start webcam or load video
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or path to video file

    frame_count = 0
    max_frames = 100  # Limit to 100 frames for demo

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)

        # Get detected class names
        names = model.names
        detected_classes = [names[int(box.cls)] for box in results[0].boxes]

        # Print detections
        print(
            f"Frame {frame_count}: Detected {len(detected_classes)} objects: {detected_classes}"
        )

        frame_count += 1

    cap.release()


if __name__ == "__main__":
    run_yolo_detection()
