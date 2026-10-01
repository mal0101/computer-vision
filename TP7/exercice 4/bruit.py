import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

DOSSIER = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# 1. Charger l'image
# --------------------------------------------------

image = cv2.imread(os.path.join(DOSSIER, "image_bruitee_4.jpg"))
if image is None:
    raise FileNotFoundError("image_bruitee_4.jpg introuvable dans " + DOSSIER)

# Conversion BGR -> RGB pour l'affichage avec Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# 2. Bruit gaussien
# --------------------------------------------------

def bruit_gaussien(image, moyenne=0, sigma=25):
    """
    Ajoute un bruit gaussien à l'image.
    moyenne : moyenne du bruit
    sigma   : écart-type du bruit
    """

    bruit = np.random.normal(moyenne, sigma, image.shape)

    image_bruitee = image.astype(np.float32) + bruit

    # Limiter les valeurs entre 0 et 255
    image_bruitee = np.clip(image_bruitee, 0, 255)

    return image_bruitee.astype(np.uint8)


# --------------------------------------------------
# 3. Bruit sel-poivre
# --------------------------------------------------

def bruit_sel_poivre(image, proportion=0.05):
    """
    Ajoute un bruit sel-poivre.
    proportion : proportion de pixels bruités.
    """

    image_bruitee = image.copy()

    # Nombre total de pixels
    hauteur, largeur = image.shape[:2]
    nombre_pixels = int(proportion * hauteur * largeur)

    # Pixels "sel" -> blanc
    for _ in range(nombre_pixels // 2):
        x = np.random.randint(0, hauteur)
        y = np.random.randint(0, largeur)
        image_bruitee[x, y] = 255

    # Pixels "poivre" -> noir
    for _ in range(nombre_pixels // 2):
        x = np.random.randint(0, hauteur)
        y = np.random.randint(0, largeur)
        image_bruitee[x, y] = 0

    return image_bruitee


# --------------------------------------------------
# 4. Bruit uniforme
# --------------------------------------------------

def bruit_uniforme(image, minimum=-30, maximum=30):
    """
    Ajoute un bruit uniforme à l'image.
    """

    bruit = np.random.uniform(
        minimum,
        maximum,
        image.shape
    )

    image_bruitee = image.astype(np.float32) + bruit

    # Limiter les valeurs entre 0 et 255
    image_bruitee = np.clip(image_bruitee, 0, 255)

    return image_bruitee.astype(np.uint8)


# --------------------------------------------------
# 5. Génération des trois images bruitées
# --------------------------------------------------

image_gaussien = bruit_gaussien(image)
image_sel_poivre = bruit_sel_poivre(image)
image_uniforme = bruit_uniforme(image)


# --------------------------------------------------
# 6. Affichage
# --------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Image originale")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image_gaussien, cv2.COLOR_BGR2RGB))
plt.title("Bruit gaussien")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(image_sel_poivre, cv2.COLOR_BGR2RGB))
plt.title("Bruit sel-poivre")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image_uniforme, cv2.COLOR_BGR2RGB))
plt.title("Bruit uniforme")
plt.axis("off")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. Sauvegarde des images
# --------------------------------------------------

cv2.imwrite(os.path.join(DOSSIER, "image_bruit_gaussien.jpg"), image_gaussien)
cv2.imwrite(os.path.join(DOSSIER, "image_bruit_sel_poivre.jpg"), image_sel_poivre)
cv2.imwrite(os.path.join(DOSSIER, "image_bruit_uniforme.jpg"), image_uniforme)