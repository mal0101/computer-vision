import os
import cv2
import matplotlib.pyplot as plt

image = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.jpg"))
if image is None:
    raise FileNotFoundError("image.jpg not found next to exo5.py")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(
    gray,
    None
)

print("Nombre de keypoints :", len(keypoints))
print("Taille des descripteurs :", descriptors.shape)

result = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10,8))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "keypoints with SIFT.jpg"), result)
plt.axis("off")
plt.show()