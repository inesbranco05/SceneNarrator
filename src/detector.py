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

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1 

    results = model(frame, verbose=False)

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
    normalized_scene = normalize_scene(detections)
    
    if frame_count % 60 == 0:

        if scene_changed(normalized_scene, previous_scene):

            narration = generate_narration(scene_description, environment_context)

            if narration != last_narration:
                print("\nNarration:")
                print(narration)
                print("-" * 50)

                speak(narration)
                
                last_narration = narration
            previous_scene = normalized_scene

    annotated_frame = results[0].plot()

    cv2.imshow("Scene Narrator", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()