import numpy as np
import cv2
import fonctions.dessin as ds

dessin = False
couleur_final = (0, 255, 0)
couleur_dessin = (0, 0, 255)
pts = []

def on_mouse(event, x, y, flags, param):
    global pts, dessin, img, img_bk
    if event == cv2.EVENT_LBUTTONDOWN:
        dessin= True
        ajout_point(pts, (x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if dessin== True:
            img = img_bk.copy()
            dessiner_polygone(img, pts, (x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        ajout_point(pts, (x, y), True)
        dessiner_polygone(img, pts, (x, y), True)
        dessin= False
        pts.clear()
        img_bk = img.copy()

def ajout_point(points, pt_courant, est_finale=False):
    print("Ajout de points #%d de position(%d,%d)"
          % (len(points), pt_courant[0], pt_courant[1]))
    points.append(pt_courant)
    if est_finale == True:
        print("Completer polygone avec %d points." % len(pts))

def dessiner_polygone(img, points, pt_courant, est_finale=False):
    if (len(points) > 0):
        if est_finale == False:
            ds.dess_polylignes(img, np.array([points]), False, couleur_final)
            ds.dess_ligne(img, points[-1], pt_courant, couleur_dessin)
        else:
            ds.dess_polylignes(img, np.array([points]), True, couleur_final)
        for point in points:
            ds.dess_cercle(img, point, 2, couleur_final, 2)
            ds.dess_ligne(img, point, point, couleur=couleur_final, epaisseur=2)

def Afficher_instruction(img):
    txtInstruction = "Touche gauche pour ajouter un point, touche droite pour terminer un polygone. ESC pour sortir."
    ds.dess_text(img,txtInstruction, (10, 20), 0.5, (255, 255, 255))
    print(txtInstruction)

def main():
    global img, img_bk
    nomFenetre = 'Dessiner polygone avec souris'
    img = np.zeros((500, 1000, 3), np.uint8)
    img[:]=100,100,100
    Afficher_instruction(img)
    img_bk = img.copy()
    cv2.namedWindow(nomFenetre)
    cv2.setMouseCallback(nomFenetre, on_mouse)
    while (True):
        cv2.imshow(nomFenetre, img)
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()