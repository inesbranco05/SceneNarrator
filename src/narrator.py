from collections import Counter

def get_position(x_center, y_center, frame_width, frame_height):

    if x_center < frame_width / 3:
        horizontal = "left"

    elif x_center < 2 * frame_width / 3:
        horizontal = "center"

    else:
        horizontal = "right"

    if y_center < frame_height / 3:
        vertical = "top"

    elif y_center < 2 * frame_height / 3:
        vertical = "middle"

    else:
        vertical = "bottom"

    return f"{vertical}-{horizontal}"
    
def describe_scene(detections):

    if not detections:
        return "No objects detected."

    counts = Counter()

    for obj in detections:
        key = (
            obj["label"],
            obj["position"],
            obj["distance"]
        )
        counts[key] += 1

    descriptions = []

    for (label, position, distance), count in counts.items():

        if count == 1:
            descriptions.append(
                f"a {label} at {position}"
            )

        else:
            descriptions.append(
                f"{count} {label}s at {position}"
            )

    return "There is " + ", ".join(descriptions) + "."

def normalize_scene(detections):

    normalized = []

    for obj in detections:

        normalized.append(
            f"{obj['label']}:{obj['position']}:{obj['distance']}"
        )

    return sorted(normalized)