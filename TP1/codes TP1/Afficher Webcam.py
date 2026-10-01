#
# Afficher la webcam comme video
#
import cv2

capture = cv2.VideoCapture(0)                   # lire a partir de la camera par defaut

# configurer les propriétées de la video
# https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704dab26d2ba37086662261148e9fe93eecad
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)      # la largeur
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)     # la hauteur
capture.set(cv2.CAP_PROP_BRIGHTNESS, 180)       # la brillance
capture.set(cv2.CAP_PROP_CONTRAST, 50)          # le contraste

succes, img = capture.read()
while succes:
    cv2.imshow("Webcam", img)




    # Taper sur la touche ESC pour stopper la boucle
    if cv2.waitKey(10) & 0xFF == 27:
        break
    succes, img = capture.read()

capture.release()
cv2.destroyWindow("Webcam")