import math


def convert_bbox(box, width, height, origin):
    x1, y1, x2, y2 = map(float, box)
    if origin == "bottom-left":
        y1, y2 = height - y2, height - y1
    elif origin != "top-left":
        raise ValueError(f"Unknown origin: {origin}")
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {box}")
    return [x1, y1, x2, y2]


def crop_box(image, box, margin=4):
    x1, y1, x2, y2 = box
    return image.crop((max(0, math.floor(x1-margin)), max(0, math.floor(y1-margin)),
                       min(image.width, math.ceil(x2+margin)), min(image.height, math.ceil(y2+margin))))
