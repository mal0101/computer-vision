import cv2
import numpy as numpy
img = cv2.imread('ressources/image1.jpg')

img=cv2.resize(img,(0,0),fx=1,fy=1.5)
print("valeur pixel =",img[100,100])
img[40:80,40:80]=(0,0,0)
cv2.imshow("affichage",img)
bleu = img.copy()
bleu[:,:,1]=0
bleu[:,:,2]=0
cv2.imshow("bleu",bleu)
rouge = img.copy()
rouge[:,:,0]=0
rouge[:,:,1]=0
cv2.imshow("rouge",rouge)

vert = img.copy()
vert[:,:,0]=0
vert[:,:,2]=0
cv2.imshow("vert",vert)
cv2.waitKey(0)
cv2.destroyAllWindows()

hsv =cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(hsv)
cv2.imshow("Hue",h)
cv2.imshow("Saturation",s)
cv2.imshow("Luminosité",v)
cv2.waitKey(0)
cv2.destroyAllWindows()