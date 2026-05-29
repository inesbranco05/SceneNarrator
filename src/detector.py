from ultralytics import YOLO
import cv2
from narrator import get_position, describe_scene

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0]
        x_center = (x1 + x2) / 2
        position = get_position(x_center, frame.shape[1])

        detections.append({
            "label": label,
            "position": position
        })

    scene_description = describe_scene(detections)
    print(scene_description)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()