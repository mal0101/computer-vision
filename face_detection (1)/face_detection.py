from PIL import Image
import face_recognition
import cv2
import numpy as np



image = face_recognition.load_image_file("people4.jpg")
visages_positions = face_recognition.face_locations(image)

# afficher le nombre de visage detécter 
print("J'ai trouvé  {} visage(s) dans cette photo".format(len(visages_positions)))

for visage_position in visages_positions:

	#Afficher la position de chaque visage trouvé dans l'image
	haut, droit, bas, gauche = visage_position
	print("un visage est detecter a la position: haut{}, gauche: {}, bas: {}, droit: {}".format(haut,  bas, gauche, droit))
	#accéder, afficher et enregistrer chaque visage trouvé dans l'image
	image_visage = image[haut:bas, gauche:droit]
	cv2.imshow("Visage detecte",image_visage)
	cv2.imwrite("visage -{}-{}-{}-{}.jpg".format(haut,  bas, gauche, droit),image_visage)
	cv2.waitKey(0) 
	#pil_image = Image.fromarray(image_visage)
	#pil_image.show()
# exemple d'execution : >python3 code_lab.py --image ../materiels/people4.jpg


print(visages_positions)