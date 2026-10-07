import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("object.jpg", 0)
img2 = cv2.imread("scene.jpg", 0)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(
    des1,
    des2,
    k=2
)

# Lowe ratio test
good = []

for m, n in matches:

    if m.distance < 0.7 * n.distance:
        good.append(m)

print("Good matches :", len(good))
# Homogtaphie
if len(good) >= 4:

    src_pts = np.float32([
        kp1[m.queryIdx].pt
        for m in good
    ]).reshape(-1,1,2)

    dst_pts = np.float32([
        kp2[m.trainIdx].pt
        for m in good
    ]).reshape(-1,1,2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    print("Homographie :")
    print(H)

    print("Inliers :", mask.sum())

    #Localisation

    h, w = img1.shape

corners = np.float32([
    [0,0],
    [w,0],
    [w,h],
    [0,h]
]).reshape(-1,1,2)

transformed = cv2.perspectiveTransform(
    corners,
    H
)

scene_color = cv2.cvtColor(
    img2,
    cv2.COLOR_GRAY2BGR
)

cv2.polylines(
    scene_color,
    [np.int32(transformed)],
    True,
    (0,255,0),
    3
)

plt.figure(figsize=(12,8))

plt.imshow(
    cv2.cvtColor(
        scene_color,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.show()
cv2.imwrite("image cadre.jpg", scene_color)
