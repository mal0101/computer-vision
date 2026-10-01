import cv2
import numpy as np

#Definition de fonctions de dessin
def dess_ligne(image, pt_debut, pt_fin, couleur=(255,255,255), epaisseur=1, type_ligne=cv2.LINE_AA):
    cv2.line(image, pt_debut, pt_fin, couleur, epaisseur, type_ligne)
def dess_rectangle(image, haut_gauche, bas_droit, couleur=(255,255,255), epaisseur=1, type_ligne=cv2.LINE_AA):
    cv2.rectangle(image, haut_gauche, bas_droit, couleur, epaisseur, type_ligne)
def dess_cercle(image, centre, rayon, couleur=(255,255,255), epaisseur=1, type_ligne=cv2.LINE_AA): 
    cv2.circle(image, centre, rayon, couleur, epaisseur, type_ligne)
def dess_ellipse(image, centre, axes, angle, debut_angle, fin_angle, couleur=(255,255,255), epaisseur=1, type_ligne=cv2.LINE_AA):
    cv2.ellipse(image, centre, axes, angle, debut_angle, fin_angle, couleur, epaisseur, type_ligne)
def dess_polylignes(image, points, est_fermer=True, couleur=(255,255,255), epaisseur=1, type_ligne=cv2.LINE_AA ):
    cv2.polylines(image, points, est_fermer, couleur, epaisseur, type_ligne)

def dess_text(image, text, org,
              taille=1,
              couleur=(255,255,255),
              epaisseur=1,
              police=cv2.FONT_HERSHEY_COMPLEX,
              type_ligne=cv2.LINE_AA):
    cv2.putText(image, text, org, police,
                taille, couleur, epaisseur, type_ligne	)


