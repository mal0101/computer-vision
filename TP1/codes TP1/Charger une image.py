import cv2
img = cv2.imread('ressources/image1.jpg')
cv2.imshow("Image", img)
print(img[10,30])
cv2.waitKey(0)
cv2.destroyAllWindows()
