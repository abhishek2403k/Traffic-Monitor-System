import cv2
from ultralytics import YOLO

# Load fastest YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("input.mp4")

# Vehicle classes
vehicle_classes = [2, 3, 5, 7]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Tracking instead of pure detection
    results = model.track(
        frame,
        persist=True,
        conf=0.3,
        iou=0.5,
        classes=vehicle_classes,
        device="cpu"  # Using CPU since GPU is not available
    )

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            cls = int(box.cls[0])
            label = f"{model.names[cls]} ID:{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Real-Time Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
