import numpy as np

GRAY_CODES = [
    25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
    75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
    125, 130, 135, 140, 145, 150, 155, 160, 165, 170,
    175, 180, 185, 190, 195, 200, 205, 210, 215
]

NUM_CLASSES = len(GRAY_CODES) + 1

# === class_id -> gray ===
class2gray = np.zeros(NUM_CLASSES, dtype=np.uint8)
class2gray[0] = 0  # фон = 0
for cls_id, gray in enumerate(GRAY_CODES, start=1):
    class2gray[cls_id] = gray

# === gray -> class_id ===
gray2class = np.zeros(256, dtype=np.int64) - 1
gray2class[0] = 0  # фон
for cls_id, gray in enumerate(GRAY_CODES, start=1):
    gray2class[gray] = cls_id