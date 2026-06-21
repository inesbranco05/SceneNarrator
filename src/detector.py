from ultralytics import YOLO
import cv2
from narrator import get_position, describe_scene, normalize_scene
from llm import generate_narration
from utils import scene_changed
from speech import speak
from vision_context import generate_visual_context

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

ret, first_frame = cap.read()

if not ret:
    print("Could not capture initial frame.")
    exit()

cv2.imwrite("initial_scene.jpg", first_frame)

environment_context = generate_visual_context("initial_scene.jpg")

print("\nENVIRONMENT MEMORY:")
print(environment_context)
print("-" * 50)

frame_count = 0
previous_scene = ""
last_narration = ""
previous_detections = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1 

    results = model(frame, verbose=False)

    detections = []
    
    for box in results[0].boxes:

        confidence = float(box.conf[0])

        if confidence < 0.65:
            continue

        cls_id = int(box.cls[0])

        label = model.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0]

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        position = get_position(
            x_center,
            y_center,
            frame.shape[1],
            frame.shape[0]
        )

        area = (x2 - x1) * (y2 - y1)

        if area > 100000:
            distance = "near"

        elif area > 30000:
            distance = "medium"

        else:
            distance = "far"

        detections.append({
            "label": label,
            "position": position,
            "distance": distance
        })
        
    scene_description = describe_scene(detections)
    normalized_scene = normalize_scene(detections)
    current_set = set(normalized_scene)
    previous_set = set(previous_detections)

    added = current_set - previous_set
    removed = previous_set - current_set
    
    if frame_count % 60 == 0:

        if scene_changed(normalized_scene, previous_scene):

            event_description = ""

            for item in added:
                label, position, distance = item.split(":")

                event_description += (
                    f"A {label} appeared at "
                    f"{position} and is {distance}. "
                )

            for item in removed:
                label, position, distance = item.split(":")

                event_description += (
                    f"A {label} disappeared from "
                    f"{position}. "
                )

            if event_description.strip() == "":
                event_description = scene_description

            print("\nSCENE DESCRIPTION:")
            print(scene_description)

            print("\nEVENT DESCRIPTION:")
            print(event_description)

            print("-" * 50)

            narration = generate_narration(
                scene_description,
                environment_context
            )

            if narration != last_narration:
                print("\nNarration:")
                print(narration)
                print("-" * 50)

                print("SPEAKING:", narration)
                speak(narration)
                
                last_narration = narration
            previous_scene = normalized_scene
            previous_detections = normalized_scene.copy()


    annotated_frame = results[0].plot()

    cv2.imshow("Scene Narrator", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()