import cv2
import numpy as np

ca=np.zeros((380,480,3),np.uint8)

ca[:]=(160,160,160)
cv2.imshow("canvas vide",ca)

cv2.line(ca,(20,20),(200,250),(0,255,0),6,cv2.LINE_8)
cv2.rectangle(ca,(20,20),(200,250),(255,0,0),6,cv2.LINE_8)
cv2.circle(ca,(200,250),90,(0,0,255),6,cv2.LINE_8)

cv2.ellipse(ca,(200,250),(45,110),0,0,179,(255,255,0),6,cv2.LINE_8)
cv2.putText(ca,"LES FORMES",(240,300),cv2.FONT_HERSHEY_COMPLEX,6,(120,240,198), 4,cv2.LINE_AA)

cv2.imshow("canvas vide",ca)
cv2.waitKey(0)
cv2.destroyAllWindows()