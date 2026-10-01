import cv2
import numpy as np
import matplotlib.pyplot as plt

iamge = cv2.imread("exercice 2/image_bruitee_1.jpg")
image = cv2.cvtColor(iamge, cv2.COLOR_BGR2RGB)

noisy = image.copy()
prob = 0.5


random = np.random.rand(*noisy.shape[:2])

noisy[random < prob/2] = 0
noisy[random > 1 - prob/2] = 255

median = cv2.medianBlur(noisy, 5)
gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
mean = cv2.blur(noisy, (5,5))

plt.figure(figsize=(12,8))

images = [
    image,
    noisy,
    mean,
    gaussian,
    median
]

for idx, i in enumerate(images):
    plt.subplot(2, 3, idx+1)
    plt.imshow(i)
    plt.axis("off")
    plt.title(f"{idx}.jpg")
    cv2.imwrite(f"{idx}.jpg", cv2.cvtColor(i, cv2.COLOR_RGB2BGR))
    

plt.show()
