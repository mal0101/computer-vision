import cv2
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(
    nfeatures=500
)

keypoints, descriptors = orb.detectAndCompute(
    gray,
    None
)

print("Nombre de points :", len(keypoints))
print("Dimensions :", descriptors.shape)

result = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
cv2.imwrite("keypoints with ORB.jpg",result)