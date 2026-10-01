import cv2
import matplotlib.pyplot as plt

image = cv2.imread("image_bruitee_1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

"""
g3 = cv2.GaussianBlur(image, (3, 3), 0)
g7 = cv2.GaussianBlur(image, (7, 7), 0)
g15 = cv2.GaussianBlur(image, (15, 15), 0)
"""
g3 = cv2.GaussianBlur(image, (7, 7), 1)
g7 = cv2.GaussianBlur(image, (7, 7), 4)
g15 = cv2.GaussianBlur(image, (7, 7), 5)

plt.figure(figsize=(12, 8))

for i, (img, title) in enumerate([
    (image, "Originale.jpg"),
    (g3, "Gaussian 3x3.jpg"),
    (g7, "Gaussian 7x7.jpg"),
    (g15, "Gaussian 15x15.jpg")
]):

    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(title,image)

plt.show()