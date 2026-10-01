#
# Afficher une video à partir d'un fichier local
#
import cv2

capture = cv2.VideoCapture(1)

succes, img = capture.read()
while succes:
    #Convert the img to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Video", img)
   
    

    # taper la touche ESC pour arrèter la boucle
    if cv2.waitKey(10) & 0xFF == 27:
        break
    succes, img = capture.read()

capture.release()
cv2.destroyWindow("Video")
