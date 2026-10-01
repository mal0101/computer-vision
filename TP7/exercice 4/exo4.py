import cv2
import numpy as np
import os

DOSSIER = os.path.dirname(os.path.abspath(__file__))

original = cv2.imread(os.path.join(DOSSIER, "image_bruitee_2.jpg"))
if original is None:
    raise FileNotFoundError("image_bruitee_2.jpg introuvable dans " + DOSSIER)

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

mean = cv2.blur(gray, (5,5))
gaussian = cv2.GaussianBlur(gray, (5,5), 0)
median = cv2.medianBlur(gray, 5)
bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

def mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

print("MSE moyenneur :", mse(gray, mean))
print("MSE gaussien :", mse(gray, gaussian))
print("MSE médian :", mse(gray, median))
print("MSE bilatéral :", mse(gray, bilateral))