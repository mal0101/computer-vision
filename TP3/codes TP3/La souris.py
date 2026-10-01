import numpy as np
import cv2

#
# Liste toutes les evenements supportés par cv2
#
def list_event():
    for event in dir(cv2):
        if "EVENT" in event:
            print(event)

#
# Afficher les coordonnèes et la valeur de la couleur RGB
# quand on click avec la souris 
def souris_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        bleu = img[y, x, 0]
        vert = img[y, x, 1]
        rouge = img[y, x, 2]
        imageCouleur = np.zeros((100, 280, 3), np.uint8)
        imageCouleur[:] = [bleu, vert, rouge]
        strBGR = "(B,G,R) = (" + str(bleu) + ", " + str(vert) + ", " + str(rouge) + ")"
        strXY = "(X,Y) = (" + str(x) + ", " + str(y) + ")"
        txtFont = cv2.FONT_HERSHEY_COMPLEX
        txtCouleur = (255, 255, 255)
        cv2.putText(imageCouleur, strXY, (10, 30), txtFont, .6, txtCouleur, 1)
        cv2.putText(imageCouleur, strBGR, (10, 50), txtFont, .6, txtCouleur, 1)
        cv2.imshow("couleur", imageCouleur)

if __name__ == '__main__':
    list_event()

    img = cv2.imread('ressources/image1.jpg',-1)
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", souris_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


