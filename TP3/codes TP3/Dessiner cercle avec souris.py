import cv2
import numpy as np
import math
import fonctions.dessin as ds

dessin = False
couleur_final = (0, 255, 0)
couleur_dessin = (0, 0, 255)

def souris_ev(event, x, y, flags, param):
    global dessin, centre, rayon, img, img_bk
    if event == cv2.EVENT_LBUTTONDOWN:
        dessin = True
        centre = x, y
        rayon = 0 
        dess_cercle(img, centre, rayon, couleur_dessin)
    elif event == cv2.EVENT_MOUSEMOVE:
        if dessin == True:
            img = img_bk.copy()
            rayon = calc_rayon(centre, (x,y))
            dess_cercle(img, centre, rayon, couleur_dessin)
    elif event == cv2.EVENT_LBUTTONUP:
        dessin = False
        rayon = calc_rayon(centre, (x,y))
        dess_cercle(img, centre, rayon, couleur_final, 2, True)
        img_bk = img.copy()

def calc_rayon(Centre, point_courant):
    cx, cy = point_courant
    tx, ty = Centre
    return int(math.hypot(cx - tx, cy - ty))#calcul de rayon

def dess_cercle(img, Centre, r, couleur, line_scale=1, is_final=False ):
    txtCentre = "centre=(%d,%d)" % Centre
    txtrayon = "r=%d" % rayon
    if is_final == True:
        print("Completer le cercle avec %s et %s" % (txtCentre, txtrayon))
    ds.dess_cercle(img, Centre, 1, couleur, line_scale)   # dessiner le Centre 
    ds.dess_cercle(img, Centre, r, couleur, line_scale)   # dessiner cercle
    ds.dess_text(img, txtCentre, (Centre[0]-60, Centre[1]+20), 0.5, couleur)
    ds.dess_text(img, txtrayon, (Centre[0]-15, Centre[1]+35), 0.5, couleur)

def afficher_instruction(img):
    txtInstruction = "Appuyez et maintenez la touche gauche pour dessiner un cercle. ESC pour sortir."
    ds.dess_text(img,txtInstruction, (10, 20), 0.5, (255, 255, 255))
    print(txtInstruction)

def main():
    global img, img_bk
    fenetreNom = 'Dessin Cercles avec souris'
    img = np.zeros((500, 1000, 3), np.uint8)
    img[:] = 100, 100, 100
    afficher_instruction(img)
    img_bk = img.copy()
    cv2.namedWindow(fenetreNom)
    cv2.setMouseCallback(fenetreNom, souris_ev)
    while (True):
        cv2.imshow(fenetreNom, img)
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()