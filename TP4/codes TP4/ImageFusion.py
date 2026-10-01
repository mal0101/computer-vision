import cv2
import fonctions.TraitementImage as ti


def change_alpha(valeur):
    global alpha
    alpha = float(valeur)/100

def fusionDeuxImages(image1, image2):
    global alpha
    alpha = 0.1
    titre = "fusion"
    itraiter = ti.TraitementImage(titre, image1)
    tofusion = cv2.imread(image2)
    #cv2.createTrackbar("Alpha", "fusion", 50, 100, change_alpha)

    itraiter.affiche("Original Image 1")
    itraiter.affiche("Original Image 2", tofusion)
    print("Press s key to save image, ESC to exit.")
    while True:
        image_fusionner = itraiter.fusion(tofusion, alpha)
        itraiter.affiche("fusion",image_fusionner)
        ch = cv2.waitKey(10)
        if (ch & 0xFF) == 27:
            break
        elif ch == ord('s'):
            # press 's' key to save image
            chemin = 'ressources/fusion.jpg'
            cv2.imwrite(chemin, image_fusionner)
            print("image enregistrer " + chemin)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path_image1 = 'ressources/f1.jpg'
    path_image2 = 'ressources/f2.jpg'
    fusionDeuxImages(path_image1, path_image2)


