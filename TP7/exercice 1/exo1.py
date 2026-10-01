import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image_bruitee_1.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

"""
# Construire le filtre manuellement
kernel = np.ones((3, 3), np.float32) / 9

result = cv2.filter2D(image, -1, kernel)

plt.imshow(result)
plt.axis("off")
plt.show()
"""

blur3 = cv2.blur(image, (3, 3))
blur5 = cv2.blur(image, (5, 5))
blur8 = cv2.blur(image, (8, 8))
blur12 = cv2.blur(image, (12, 12))
blur15 = cv2.blur(image, (15, 15))

plt.figure(figsize=(12, 12))

images = [image, blur3, blur5, blur8,blur12, blur15]
titles = ["Originale.jpg", "3x3.jpg", "5x5.jpg","8x8.jpg","12x12.jpg", "15x15.jpg"]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")
    image = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
    cv2.imwrite(titles[i],image)

plt.show()