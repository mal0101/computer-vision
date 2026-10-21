# Panorama stitcher

Stitches `part_1.jpg … part_5.jpg` into a validated panorama (correct result: **612×154**).

```bash
python main.py                          # writes panorama.jpg next to the tiles
python -m pytest test_stitch.py         # 15 regression tests
```

Requires OpenCV + NumPy only. Run from this directory (or pass `--dir`).

## Why the original `cv2.Stitcher` version produced garbage

The previous 20-line script used `cv2.Stitcher_create(Stitcher_PANORAMA)` with default
settings. On these tiles it never produced a correct panorama:

| What was measured (10 runs, confidence 0.5) | Result |
|---|---|
| Failures (`ERR_NEED_MORE_IMGS`) | 6 / 10 |
| "Successes" with degenerate canvases | 31×63, 86×149, 110×218, 989×2084 |
| Correct 612×154 | **0 / 10** |

The committed `panorama.jpg` (63×31 — smaller than a single input tile) is one of those
degenerate RANSAC draws that was saved to disk. Lowering the confidence threshold did not
help (0.3 → 85×187, 0.0 → 744×1521); raising it made every run fail.

### Root causes (all verified numerically)

1. **Resolution starvation.** Tiles are 161×154 px. SIFT finds only 25–72 keypoints per
   tile; ORB (the stitcher's default-class detector) yields 1–3 valid matches per pair —
   below the 4 needed to estimate even one homography. Pair `part_1→part_2` has just
   3 SIFT matches, breaking the connection graph entirely.
2. **Wrong motion model.** The tiles are exact rectangular crops of one flat 612×154
   image cut at a constant **112 px stride** (overlap: 49 px, ~30%). Overlap pixels are
   ~96 % bit-identical (mean |diff| 0.01/255, correlation 1.000, dy = 0). The correct
   model is pure translation; fitting full perspective homographies to starved, noisy
   matches amplifies error into collapsed or exploded canvases.
3. **No output validation.** `Stitcher.stitch()` returning `OK` says the internal
   pipeline completed — not that the geometry is sane. A 63×31 result passed silently.

## How this version works

Ordered-tile pipeline (`main.py`):

1. **Integer offset search** between consecutive tiles (exhaustive dx/dy, scored by
   mean-absolute-difference over the overlap).
2. **Parabolic sub-pixel step**, then **ECC refinement** (`cv2.findTransformECC`,
   translation model). Every candidate is re-scored against the actual pixels in the
   forward warp convention — a refinement is kept only if it lowers the residual.
3. **Acceptance gates** per pair: MAD ≤ 8/255, correlation ≥ 0.95, |dy| ≤ 5 px,
   overlap ≥ 40 px. Offsets within 0.05 px of an integer are snapped (the tiles are
   pixel crops, and this keeps runs bit-deterministic).
4. **SIFT + RANSAC affine fallback** (Lowe ratio 0.75, ≥ 10 inliers, median reprojection
   ≤ 3 px, span and residual gates) for tile sets that are not pure translations.
5. **Composition** on a bounding-box canvas with distance-transform feather blending.
6. **Output gates**: the panorama may not be smaller than the largest tile, may not
   exceed the sum of tile extents (the degenerate-warp signature), and must cover
   ≥ 80 % of the canvas — otherwise nothing is written.

Measured on the bundled tiles: all four pairs align at dx = +112.00, MAD ≈ 0.01,
corr = 1.000 → canvas 612×154, 100 % coverage, non-overlap pixels **bit-identical** to
exact integer placement, identical across repeated runs (~3 s).

### CLI

```
--dir PATH       tile directory        (default: this folder)
--inputs NAMES   tile names in order   (default: part_1.jpg … part_5.jpg)
--output PATH    output file           (default: <dir>/panorama.jpg)
--no-ecc         skip ECC refinement   (integer search + parabolic only)
--no-sift        disable the SIFT affine fallback
```

## Shooting guidelines (for future panoramas)

- Keep **original resolution** — thumbnails under ~500 px per side starve every
  feature detector; this dataset sits at the absolute floor of what is stitchable.
- Aim for **30–50 % overlap** between consecutive shots, with textured content
  (not pure sky) inside the overlap.
- Keep a steady exposure and hold the camera height/rotation constant; pan like a
  tripod head (yaw only).
- If the default stitcher fails on your own photos, this script's SIFT fallback +
  gates is the safe path — it will either return a sane panorama or a precise error
  naming the failing pair and threshold.
