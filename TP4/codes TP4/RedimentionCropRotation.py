import cv2
import fonctions.TraitementImage as ti

if __name__ == "__main__":
    # Crèer un objet de la classe TraitementImage
    ti = ti.TraitementImage("Originale", 'ressources/image1.jpg')

    # affiche l'image originale
    ti.affiche(image=ti.redimention(80))

    # Redimentionner l'image Originale et l'afficher
    image_redim = ti.redimention(50)
    ti.affiche("Redimentionner -- 50%", image_redim)

    # rotation de l'image redimentionner et l'afficher
    imge_rotation = ti.rotation(30, image_redim)
    ti.affiche("Rotourner -- 30 degree", imge_rotation)

    # Cropper l'image original et l'afficher
    imge_croppe = ti.crop((100, 100), (530, 400))
    ti.affiche("Cropper", imge_croppe)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

