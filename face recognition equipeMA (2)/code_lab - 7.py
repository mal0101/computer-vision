import face_recognition
from PIL import Image, ImageDraw
#
# Ce programme est un exemple de reconnaissance de visage
# avec encadrement est reconnaissance sur l'image du nom du personnage 

# charger l'image de la personne est apprendre au système a la reconnaitre  .
hakim_image = face_recognition.load_image_file("D:\\cours 2022\\ENSAM\\Cours IAGI 2 Intelligence Artificielle\\Labs\\Lab 2 - Vision intélligentes via OpenCV\\work\\Lab 8\\hakim.jpg")
hakim_visage_signature = face_recognition.face_encodings(hakim_image)[0]


# charger la deuxiéme image de personne est apprendre au système a la reconnaitre  .
mrabet_image = face_recognition.load_image_file("D:\\cours 2022\\ENSAM\\Cours IAGI 2 Intelligence Artificielle\\Labs\\Lab 2 - Vision intélligentes via OpenCV\\work\\Lab 8\\Mrabet.jpg")
mrabet_visage_signature = face_recognition.face_encodings(mrabet_image)[0]

# creer un tableau des signature des visages associés au nom de personnes
reconnu_visage_signatures = [
    hakim_visage_signature,
    mrabet_visage_signature
]
reconnu_visage_noms = [
    "Hakim Ziyach",
    "Soufiane Mrabet"
]

# Charger l'image avec les visages non reconnus
non_reconnu_image = face_recognition.load_image_file("D:\\cours 2022\\ENSAM\\Cours IAGI 2 Intelligence Artificielle\\Labs\\Lab 2 - Vision intélligentes via OpenCV\\work\\Lab 8\\equipeMA.jpg")

# trouver tout les visages et leurs signature dans une image source 
visage_locations = face_recognition.face_locations(non_reconnu_image)
visage_signatures = face_recognition.face_encodings(non_reconnu_image, visage_locations)

# convertire l'image en PIL-format pour qu'on puisse dessiner avec la librairie pillow  
# regarder http://pillow.readthedocs.io/ pour plus d'informations sur PIL/Pillow
pil_image = Image.fromarray(non_reconnu_image)
# creer une image ou nous allons faire des dessin sous format d'instance de Pillow ImageDraw 
img_dessin=ImageDraw.Draw(pil_image)
# boucler sur les visage trouver dans l'image à reconnaite
for (top, right, bottom, left), visage_signature in zip(visage_locations, visage_signatures):
    # voir si le visage extrait correspond a un(des) visage(s) reconnu(s)
    matches = face_recognition.compare_faces(reconnu_visage_signatures, visage_signature)

    nom_personne = "non_reconnu"

    # Si une correspondance est retrouvé dans les reconne_visage_signatures, seulement prendre le premier!
    if True in matches:
        first_match_index = matches.index(True)
        nom_personne = reconnu_visage_noms[first_match_index]

    # dessiner un rectangle atour du visage utilisant les modules de pillow
    img_dessin.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # dessiner un label avec le text du nom de visage reconnus
    text_width, text_height = img_dessin.textsize(nom_personne)
    img_dessin.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 255, 0), outline=(0, 0, 255))
    img_dessin.text((left + 6, bottom - text_height - 5), nom_personne, fill=(255, 0, 0, 255))


# supprimer la library de dessin de la  memoire
del img_dessin

# afficher l'image resultant
pil_image.show()

pil_image.save("D:\\cours 2022\\ENSAM\\Cours IAGI 2 Intelligence Artificielle\\Labs\\Lab 2 - Vision intélligentes via OpenCV\\work\\Lab 8\\image_avec_detection.jpg")
