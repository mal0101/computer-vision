import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic.jpg"))
if image is None:
    raise FileNotFoundError("synthetic.jpg not found next to exo2.py")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

corners = cv2.cornerHarris(
    gray,
    blockSize=2,
    ksize=3,
    k=0.04
)

corners = cv2.dilate(corners, None)

result = image.copy()

result[corners > 0.01 * corners.max()] = [0, 0, 255]

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()