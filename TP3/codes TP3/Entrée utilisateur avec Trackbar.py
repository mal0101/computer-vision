import cv2

def change_brillance(valeur):
    global cap
    print("Brillance: " + str(valeur))
    cap.set(cv2.CAP_PROP_BRIGHTNESS, valeur)

def change_contraste(valeur):
    print("Contraste: " + str(valeur))
    cap.set(cv2.CAP_PROP_CONTRAST, valeur)

def change_saturation(valeur):
    print("Saturation: " + str(valeur))
    cap.set(cv2.CAP_PROP_SATURATION, valeur)

def change_hue(valeur):
    print("Hue: " + str(valeur))
    cap.set(cv2.CAP_PROP_HUE, valeur)

def redimension(image, pourcentage):
    largeur= int(image.shape[1] * pourcentage / 100)
    Hauteur = int(image.shape[0] * pourcentage / 100)
    image_redimensionee = cv2.resize(image, (largeur, Hauteur))
    return image_redimensionee


def main():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)      # paramétrer largeur
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)     # paramétrer Hauteur
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # paramétrer brillance
    cap.set(cv2.CAP_PROP_CONTRAST, 50)          # paramétrer Contraste
    cap.set(cv2.CAP_PROP_SATURATION, 90)        # paramétrer Saturation
    cap.set(cv2.CAP_PROP_HUE, 15)               # paramétrer Hue

    cv2.namedWindow('Webcam')
    cv2.createTrackbar('Brillance', 'Webcam', 100,300, change_brillance)
    cv2.createTrackbar('Contraste', 'Webcam', 50, 300, change_contraste)
    cv2.createTrackbar('Saturation', 'Webcam', 90, 100, change_saturation)
    cv2.createTrackbar('Hue', 'Webcam', 15, 360, change_hue)

    succes, img = cap.read()
    while succes:
        cv2.imshow("Webcam", redimension(img, 90))

        
        if cv2.waitKey(10) & 0xFF == 27:
            break
        success, img = cap.read()

    cap.release()
    cv2.destroyWindow("Webcam")

if __name__ == '__main__':
    main()