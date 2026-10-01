import cv2
img = cv2.imread('ressources/image1.jpg')
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



cv2.imshow("Image couleur", img)
cv2.imshow("Image niveaux de gris", gris)
cv2.imwrite('ressources/image1_gris.bmp',gris)
cv2.waitKey(0)
cv2.destroyAllWindows()