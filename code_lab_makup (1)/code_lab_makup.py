import face_recognition
from PIL import Image, ImageDraw

# Charger l'image dans une table
image = face_recognition.load_image_file("D:\\cours 2022\\ENSAM\\Cours IAGI 2 Intelligence Artificielle\\Labs\\Lab 2 - Vision intélligentes via OpenCV\\work\\Lab 8\\No Makeup.jpg")

# trouver toute les composantes de visages dans 
face_landmarks_list = face_recognition.face_landmarks(image)

print(face_landmarks_list)

for face_landmarks in face_landmarks_list:
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image, 'RGBA')
    # RGBA - creaer une image o format de couleur alpha composite , pour permettre de mettre de la transparence 

    # sourcis plus foncés
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # ajouter du rouge à lévre
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    # rendre les yeux rouge
    d.polygon(face_landmarks['left_eye'], fill=(255, 0, 0, 30))
    d.polygon(face_landmarks['right_eye'], fill=(255, 0, 0, 30))

    # mettre un peu de masscare au yeux
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
   
    pil_image.show()

    
    pil_image.save("D:\\cours 2022\\ENSAM\\Cours IAGI 2 Intelligence Artificielle\\Labs\\Lab 2 - Vision intélligentes via OpenCV\\work\\Lab 8\\image_makeup.jpg")
