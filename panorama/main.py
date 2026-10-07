import os
import cv2

base_dir = os.path.dirname(os.path.abspath(__file__))

images = []
for name in ["part_1.jpg", "part_2.jpg", "part_3.jpg", "part_4.jpg", "part_5.jpg"]:
    img = cv2.imread(os.path.join(base_dir, name))
    if img is None:
        raise FileNotFoundError(f"{name} not found next to main.py")
    images.append(img)

stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)  # ou PANORAMA / SCAN
stitcher.setPanoConfidenceThresh(0.5)
status, pano = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    cv2.imwrite(os.path.join(base_dir, "panorama.jpg"), pano)
else:
    print("Échec du stitching, code:", status)