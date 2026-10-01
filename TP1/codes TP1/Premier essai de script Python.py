import cv2
import numpy as np
img = np.zeros((320,1000,3), np.uint8)#initiation de la canvas 
img[:] = 125,125,125 #couleur de canvas

tx_position = (80,160)

tx_police = cv2.FONT_HERSHEY_COMPLEX #type de caractéres
tx_couleur = (0, 255, 0)#couleur de caractéres
tx_policeTaille = 2 #taille de caractéres
tx_policeEpaisseur = 2 #épaisseur de caractéres
cv2.putText(img, "Hello OpenCV", tx_position, tx_police,tx_policeTaille, tx_couleur, tx_policeEpaisseur)
cv2.imshow("Bonjour", img)#fenetre d'affichage
cv2.waitKey(5000)
cv2.destroyAllWindows()