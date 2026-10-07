import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("object.jpg", 0)
img2 = cv2.imread("scene.jpg", 0)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=True
)

matches = bf.match(des1, des2)

matches = sorted(
    matches,
    key=lambda x: x.distance
)

result = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:50],
    None,
    flags=2
)

plt.figure(figsize=(15,8))
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()
cv2.imwrite("restltat de matching.jpg", result)