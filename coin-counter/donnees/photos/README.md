# Banc de détection — photographies réelles

Six photographies servant de **banc de détection** (compter les disques), et
non de classification : aucune n'est marocaine, donc aucune dénomination n'y
est attendue. Elles couvrent les régimes que `coins.jpg` n'exerce pas —
surtout le **fond clair** (polarité inverse), que le reste du projet suppose
sombre.

| fichier | pièces | fond | pourquoi il est ici |
|---|---|---|---|
| `us_vrac_8.jpg` | 8 | blanc | huit pièces US bien séparées, fond uniforme clair |
| `us_grille_12.png` | 12 | blanc | grille 4×3, tailles très contrastées (1¢ → 1 \$) |
| `jouet_6.jpg` | 6 | blanc | pièces-jouets, reliefs émoussés, contraste faible |
| `or_fond_blanc_5.jpg` | 5 | blanc pur | pièces dorées sur blanc : le cas le plus dur (faible contraste, ombres portées) |
| `uae_grille_12.png` | 12 | blanc | 12 dirhams des EAU, quasi identiques en taille |
| `table_reelle_3.png` | 3 | gris clair | **vraie photo de table**, flou léger — le plus proche d'une scène webcam |

Total : **46 pièces**.

Les décomptes ont été établis visuellement, pièce par pièce, puis recoupés
avec une segmentation indépendante du détecteur du projet. Ils sont figés
dans `donnees/photos.json`.

## Provenance

Jeu de données `coinImages` de Mariam Al Kuwaiti, publié sur Kaggle
(<https://www.kaggle.com/datasets/mariamalkuwaiti/coinimages>), accompagnant
le carnet « OpenCV Coin Detection and Counting ». Récupéré le 2026-10-20.

Deux fichiers portaient l'extension `.png` alors qu'ils étaient en réalité
encodés en AVIF et en WebP ; ils ont été ré-encodés en PNG véritable, sans
autre modification. Les quatre autres sont les octets d'origine, renommés.
