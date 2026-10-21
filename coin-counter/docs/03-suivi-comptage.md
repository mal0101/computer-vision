# 03 — Suivi et comptage (M3)

Ce document décrit la machine d'état des pistes et la somme « monotone ».
Implémenté et testé dans `pieces/suivi.py`, `pieces/etat.py`,
`pieces/traitement.py` et `tests/test_suivi.py` (+ `tests/test_m3.py` for
le pipeline complet).

## 1. Objectif

En direct, une pièce passe par phases `tentative → confirmee → comptee →
perdue`. Deux règles, à ne pas confondre :

1. **le total ne baisse jamais** — on ne décompte pas, on gèle ;
2. **une même pièce physique n'est comptée qu'une fois**.

La seconde ne découlait pas de la première. Le total était bien monotone,
mais une pièce qui sortait du champ puis revenait, ou dont le rayon mesuré
dérivait de plus de `portee_rayon_rel`, ouvrait une piste neuve et était
recomptée. Une empreinte (centre, rayon, valeur) des pistes déjà comptées
est donc conservée `memoire_comptees` frames ; au-delà, une pièce
réellement reposée sur la table redevient comptable — limite assumée et
bornée, pas une garantie absolue.

**Deux axes portent des noms voisins et n'ont aucun rapport** : la *phase*
d'une piste (`tentative`, `confirmee`…) décrit sa présence temporelle, la
*certitude* d'une classification (`tentative`, `confirme`…) décrit la
confiance d'identification. Le comptage exige désormais les deux ; il
n'exigeait que la première, et additionnait donc des identifications
simplement « tentatives ».

## 2. Machine d'états et seuils (`Config.suivi`)

| Seuil | Valeur | Rôle |
|---|---|---|
| `hits_confirmer` | 3 | présences requises dans la fenêtre |
| `fenetre_confirmer` | 5 | fenêtre glissante de confirmation |
| `stabilite_requise` | 2 | frames de libellé identique avant gel |
| `mediane_libelle` | 7 | fenêtre du **vote majoritaire** du libellé |
| `max_manques` | 6 | absences avant déclaration « perdue » |
| `fenetre_perdu` | 10 | frames (délai) avant déclaration « perdue » |
| `ema_alpha` | 0,7 | lissage de la vitesse de chaque piste |
| `facteur_association` | 0,35 | seuil d'association : `f·rayon + g·‖vitesse‖` |
| `facteur_vitesse` | 2,0 | part de la vitesse dans le seuil d'association |
| `portee_rayon_rel` | 0,15 | `Δr / r` maximal pour rester la même pièce |

### Phases

1. **tentative** — piste créée à la première apparition. Une valeur est
   *candidate* dès que la classification en produit une **confirmée** ; elle
   ne devient officielle qu'après `stabilite_requise` frames identiques.
   Le départage se fait par **vote majoritaire** sur `mediane_libelle`, et
   non par médiane : une dénomination est une étiquette catégorielle, où la
   médiane n'a pas de sens — `median([100, 1000])` vaut 550, qui n'est
   aucune pièce et bloquait la piste sans valeur pour toujours. Une égalité
   de voix ne tranche pas : on attend.
2. **confirmee** — `hits_confirmer` présences dans `fenetre_confirmer` :
   le libellé et la valeur sont gelés ; **ajoutée une seule fois** au
   total (comptée), jamais retirée.
3. **comptee** — sous-état marquant le passage dans `total_centimes`
   (monotone). Le `piste_id` est posé sur les objets `PieceDetectee` de la
   frame pour le HUD.
4. **perdue** — `manques ≥ max_manques` ou délai > `fenetre_perdu`. Le
   total reste figé ; la piste est purgée des pistes actives.

### Association frame → piste

Le centre prédit est `centre_prev + vitesse·dt` (EMA `ema_alpha`). Un
candidat est associé si `‖centre − centre_prédit‖ ≤ facteur_association·r +
facteur_vitesse·‖vitesse‖` **et** `Δr/r ≤ portee_rayon_rel`. C'est ce
second test qui a révélé (et écarte) le faux renouvellement d'identité
quand une pièce en cache une autre : un écart de rayon de 87 % crée une
**nouvelle piste** (testé), donc deux pièces proches ne fusionnent pas.

## 3. Le total : somme exacte, monotone, en centimes

- `total_centimes` plateforme unique : **textuel centimes entiers**
  (jamais de flottants), `formater_montant` produit `12,50 DH`.
- Le filtre `centimes` (`compter_centimes=False` par défaut) s'applique au
  moment du comptage : une pièce de **20c est toujours identifiée** (M2)
  mais n'entre pas dans le total sans `--centimes`.
- Motifs d'incertitude — *jamais comptés* : `occlusion`, `eclat`,
  `inclinaison`, `trop_petit`, `chevauchement` non résolu, `hors_portee`
  (16,5–30,5 mm), `inconnue` (famille ou taille ambiguë).
- `Suiveur.reinitialiser()` remet pistes, total et compteur à zéro.
  `par_valeur_total` expose le décompte consolidé au HUD.

## 4. Pipeline (`traiter_image`) et étalonnage de session

`traiter_image(frame, etat)` : `reduire_a_analyse` (INTER_AREA, échelle
d'analyse `config.largeur_analyse`) → `detecter` → `classifier_pieces`
(avec l'échelle courante, `None` tant que non étalonnée) →
`_actualiser_etalonnage` (si mode étalonnage : rayon de la plus grande
pièce valide → observateur → `ppar_piece` > validation → échelle de
session) → `suiveur.maj` → `ResultatImage` (coordonnées ramenées au plein
cadre par `echelle`). L'étalonnage n'est **jamais persisté** : il vit dans
`EtatApplication.px_par_mm` pour la session.

## 5. Comportements vérifiés par tests (extrait)

- total atteint **100 centimes** à la 3ᵉ frame présentée (confirmation).
- deux pièces distinctes → 300 ; `inconnue` jamais comptée ; motif
  d'incertitude jamais suivi.
- scintillement lent (2/6 frames) → jamais confirmée, total figé.
- perte → total figé ; déplacement → une seule piste ; 3 absences puis
  réassociation OK ; saut de rayon → deux pistes.
- déterminisme : deux sessions identiques donnent les mêmes centres.
- suite complète (pipeline + HUD + bureau CLI) : `tests/test_m3.py`.