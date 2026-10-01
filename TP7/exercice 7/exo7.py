import cv2
import matplotlib.pyplot as plt

image = cv2.imread("cells.jpg", cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(image, (5,5), 0)

edges1 = cv2.Canny(blur, 50, 150)
edges2 = cv2.Canny(blur, 100, 200)
edges3 = cv2.Canny(blur, 150, 250)

plt.figure(figsize=(12,8))

for i, (img, title) in enumerate([
    (image, "Originale"),
    (edges1, "50-150"),
    (edges2, "100-200"),
    (edges3, "150-250")
]):

    plt.subplot(2,2,i+1)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.show()