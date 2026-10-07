import cv2

image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(
    hessianThreshold=400
)

keypoints, descriptors = surf.detectAndCompute(
    gray,
    None
)

print("Keypoints :", len(keypoints))
result = cv2.drawKeypoints(
    image,
    keypoints,
    None
)

cv2.imshow("SURF", result)
cv2.waitKey(0)
cv2.destroyAllWindows()