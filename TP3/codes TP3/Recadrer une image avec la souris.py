import cv2
import fonctions.dessin as ds


dessin = False
final_couleur = (0, 255, 0)
couleur_dessin = (0, 0, 255)
premier_pt = (0, 0)
second_pt = (0, 0)

def souris_ev(event, x, y, flags, param):
    global premier_pt, second_pt, dessin, img, img_bk, img_original
    if event == cv2.EVENT_LBUTTONDOWN:
        dessin = True
        premier_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if dessin == True:
            img = img_bk.copy()
            dessiner_rectangle(img, premier_pt, (x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        dessin = False
        second_pt = (x, y)
        dessiner_rectangle(img, premier_pt, second_pt, True)
        crop_image(img_original, premier_pt, second_pt)

def crop_image(img, premier_pt, second_pt):
    x_tl, y_tl = premier_pt               # top-left point
    x_br, y_br = second_pt              # bottom-right point
    if x_br < x_tl:                     # swap x value if opposite
        x_br, x_tl = x_tl, x_br
    if y_br < y_tl:                     # swap y value if opposite
        y_br, y_tl = y_tl, y_br
    img_recadre = img[y_tl:y_br, x_tl:x_br]
    cv2.imshow("Recadrage", img_recadre)

def dessiner_rectangle(img, point1, point2, is_final=False):
    if is_final == False:
        ds.dess_rectangle(img, point1, point2, couleur_dessin)
        ds.dess_text(img, str(point1), point1, couleur=couleur_dessin, epaisseur=1)
        ds.dess_text(img, str(point2), point2, couleur=couleur_dessin, epaisseur=1)
    else:
        ds.dess_rectangle(img, point1, point2, final_couleur, epaisseur=2)
        ds.dess_text(img, str(point1), point1, couleur=final_couleur, epaisseur=1)
        ds.dess_text(img, str(point2), point2, couleur=final_couleur, epaisseur=1)

def Afficher_instruction(img):
    txtInstruction = "Appuyez sur le bouton gauche pour faire glisser un rectangle, relachez-le pour le recadrer. ESC pour quitter."
    ds.dess_text(img,txtInstruction, (10, 20),0.3, (255, 255, 255))
    print(txtInstruction)

def main():
    global img, img_bk, img_original
    nomFenetre = 'Recadrage d\'image'
    img = cv2.imread('ressources/image1.jpg')
    img_original = img.copy()
    Afficher_instruction(img)
    img_bk = img.copy()
    cv2.namedWindow(nomFenetre)
    cv2.setMouseCallback(nomFenetre, souris_ev)
    while (True):
        cv2.imshow(nomFenetre, img)
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()