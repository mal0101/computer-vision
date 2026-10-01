import cv2
import numpy as np
def trackbar_ev(val):
    print("Valeur Trackbar :", val)

cv2.namedWindow("Fenetre_Trackbar")
cv2.createTrackbar("Trackbar", "Fenetre_Trackbar", 200, 800, trackbar_ev)
canevas = np.zeros((50, 800, 3), np.uint8)
canevas[:] = 235,235,235

while True:
    cv2.imshow("Fenetre_Trackbar", canevas)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()