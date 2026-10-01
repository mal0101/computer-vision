import cv2
import numpy as np
import fonctions.dessin as ds

def dess_opencv_icon(image):
    axes = (100, 100)
    centre_top_cercle = (320, 140)
    centre_basGauche_cercle = (centre_top_cercle[0]-160,
                               centre_top_cercle[1]+240)
    centre_basDroit_cercle = (centre_top_cercle[0]+160,
                                centre_top_cercle[1]+240)
    angle_top_cercle= 90
    angle_basGauche_cercle = -45
    angle_basDroit_cercle = -90
    debut_angle = 40
    fin_angle = 320
    ds.dess_ellipse(image, centre_top_cercle, axes,
                 angle_top_cercle, debut_angle, fin_angle,
                 couleur=(0, 0, 255),
                 epaisseur=80)
    ds.dess_ellipse(image, centre_basGauche_cercle, axes,
                 angle_basGauche_cercle, debut_angle, fin_angle,
                 couleur=(0, 255, 0),
                 epaisseur=80)
    ds.dess_ellipse(image, centre_basDroit_cercle, axes,
                 angle_basDroit_cercle, debut_angle, fin_angle,
                 couleur=(255, 0, 0),
                 epaisseur=80)
    ds.dess_text(image, "OpenCV", (20,660), couleur=(0,0,0),
                 taille=4.8, epaisseur=15)

if __name__ == '__main__':
    # create an image from numpy array
    canevas = np.zeros((480, 980, 3), np.uint8)
    canevas[:] = 235, 235, 235

    ds.dess_text(canevas, "Hello OpenCV", (150,260),
                 couleur=(125,0,0),
                 epaisseur=8, taille=3)
    cv2.imshow("canevas", canevas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    canevas = np.zeros((720,640,3), np.uint8)
    canevas[:] = 235,235,235
    dess_opencv_icon(canevas)
    cv2.imshow("canevas", canevas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()