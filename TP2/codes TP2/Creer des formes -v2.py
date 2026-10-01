import cv2
import numpy as np
import fonctions.dessin as ds
# créer une canevas de table numpy
canevas = np.zeros((380, 480, 3), np.uint8)
cv2.imshow("canevas", canevas)
print("taper sur une touche pour continue...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Dessiner la canevas avec une couleur
canevas[:] = 235, 235, 235
cv2.imshow("canevas", canevas)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Dessiner  ligne
ds.dess_ligne(canevas, pt_debut=(100,100), pt_fin=(canevas.shape[1]-100, canevas.shape[0]-100), couleur=(10,10,10), epaisseur=10 )
cv2.imshow("Canevas", canevas)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Dessiner rectangles
ds.dess_rectangle(canevas, (200,50), (400, 20), couleur=(255, 255, 0), epaisseur=2)
ds.dess_rectangle(canevas, (20,220), (200, 320), couleur=(255, 135, 135), epaisseur=cv2.FILLED)

#Dessiner cercles
ds.dess_cercle(canevas, centre=(280, 120), rayon=50, couleur=(85, 130, 255), epaisseur=cv2.FILLED)
ds.dess_cercle(canevas, centre=(380, 160), rayon=80, couleur=(180, 30, 175), epaisseur=5)
#Dessiner ellipse 
ds.dess_ellipse(canevas, centre=(180, 120),axes=(100,40),angle=100,debut_angle=60,fin_angle=189,couleur=(0,36,180),epaisseur=4,type_ligne=cv2.LINE_AA)

#Dessiner ellipses
pts = np.array([[25, 70], [25, 160],
                [110, 200], [220, 140],
                [200, 70], [110, 20]], np.int32)
pts = pts.reshape((-1,1,2))
ds.dess_polylignes(canevas, [pts], couleur=(23,23,10))

cv2.imshow("canevas", canevas)

cv2.waitKey(0)
cv2.destroyAllWindows()