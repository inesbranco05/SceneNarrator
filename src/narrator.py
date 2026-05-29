from collections import Counter

def get_position(x_center, frame_width):

    if x_center < frame_width / 3:
        return "left"

    elif x_center < 2 * frame_width / 3:
        return "center"

    else:
        return "right"
    
def describe_scene(detections):

    if not detections:
        return "No objects detected."

    counts = Counter()

    for obj in detections:
        key = (obj["label"], obj["position"])
        counts[key] += 1

    descriptions = []

    for (label, position), count in counts.items():

        if count == 1:
            descriptions.append(
                f"a {label} on the {position}"
            )

        else:
            descriptions.append(
                f"{count} {label}s on the {position}"
            )

    return "There is " + ", ".join(descriptions) + "."

def normalize_scene(detections):

    labels = sorted([obj["label"] for obj in detections])

    return ", ".join(labels)