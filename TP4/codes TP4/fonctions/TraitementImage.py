import cv2

class TraitementImage(object):
    def __init__(self,nom_fenetre, nom_image):
        self.nom_fenetre=nom_fenetre
        self.nom_image=nom_image
        self.image=cv2.imread(self.nom_image)
    def affiche(self,titre=None,image=None):
        if image is None:
            image=self.image
        if titre is None:
            titre=self.nom_fenetre
        cv2.imshow("titre",image)
    def redimention(self,pourcentage,image=None):
        if image is None:
            image=self.image
        largeur=int(image.shape[1]*pourcentage/100)
        hauteur=int(image.shape[0]*pourcentage/100)
        image_redimension=cv2.resize(image,(largeur,hauteur))
        return image_redimension
    def crop(self,premier_pt,second_pt,image=None):
        if image is None:
            image=self.image
        #le point haut-gauche
        x_hg,y_hg=premier_pt
        #le point bas-droit
        x_bd,y_bd=second_pt
        #echangée les x si ils sont oposés
        if x_bd<x_hg:
            x_bd,x_hg=x_hg,x_bd
        #echangée les x si ils sont oposés
        if y_bd<y_hg:
            y_bd,y_hg=y_hg,y_bd 
        image_cropper=image[y_hg:y_bd,x_hg:x_bd]
        return image_cropper
    def rotation(self,angle,image=None,echelle=1.0):
        if image is None:
            image=self.image
        (h,l)=image.shape[:2]
        centre =(l/2,h/2)
        mat_rot=cv2.getRotationMatrix2D(centre,angle,echelle)
        image_rotation=cv2.warpAffine(image,mat_rot,(l,h))
        return image_rotation
    def fusion(self,fusion,alpha,image=None ):
        if image is None:
            image=self.image
        fusion=cv2.resize(fusion,(image.shape[1],image.shape[0]))
        resultat=cv2.addWeighted(image,alpha,fusion,(1.0-alpha),0)
        return resultat
