import os
import cv2
import matplotlib.pyplot as plt

image = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess.jpg"))
if image is None:
    raise FileNotFoundError("chess.jpg not found next to exo3.py")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=100,
    qualityLevel=0.01,
    minDistance=10
)

corners = corners.astype(int)

result = image.copy()

for corner in corners:
    x, y = corner.ravel()

    cv2.circle(
        result,
        (x, y),
        4,
        (0, 0, 255),
        -1
    )

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()