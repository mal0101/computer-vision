import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("coins.jpg", cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(image, cv2.CV_64F)

laplacian_abs = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title("Originale")

plt.subplot(1,2,2)
plt.imshow(laplacian_abs, cmap="gray")
plt.title("Laplacien")

plt.show()
