import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((500,500), dtype=np.uint8)

cv2.rectangle(image, (100,100), (400,400), 255, 3)
cv2.line(image, (100,250), (400,250), 255, 3)
cv2.circle(image, (250,250), 50, 255, 3)

plt.imshow(image, cmap="gray")
plt.show()