from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

MAD_MAX = 8.0
CORR_MIN = 0.95
MAX_ABS_DY = 5.0
MIN_OVERLAP_PX = 40
SNAP_TOL = 0.05

ECC_MAX_ITERS = 200
ECC_EPS = 1e-6
ECC_GAUSS = 5

SIFT_RATIO = 0.75
SIFT_MIN_INLIERS = 10
RANSAC_REPROJ = 3.0
SPAN_MIN_FACTOR = 0.3
SPAN_MAX_FACTOR = 3.0

COVERAGE_MIN = 0.80

DEFAULT_NAMES = ["part_1.jpg", "part_2.jpg", "part_3.jpg", "part_4.jpg", "part_5.jpg"]


class StitchError(RuntimeError):
    """Tiles cannot be registered, or the result fails the sanity gates."""


@dataclass
class PairAlignment:
    method: str
    matrix: np.ndarray
    mad: float
    corr: float
    dx: float
    dy: float
    inliers: int | None = None


@dataclass
class StitchReport:
    pairs: list[PairAlignment] = field(default_factory=list)
    canvas: tuple[int, int] = (0, 0)
    coverage: float = 0.0


def load_tiles(directory: str, names: list[str]) -> list[np.ndarray]:
    tiles = []
    for name in names:
        path = os.path.join(directory, name)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise StitchError(f"cannot read image: {path}")
        tiles.append(img)
    return tiles


def _gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _gray_f32(img: np.ndarray) -> np.ndarray:
    return _gray_u8(img).astype(np.float32)


def _overlap_slices(shape_a: tuple[int, int], shape_b: tuple[int, int], dx: float, dy: float):
    h_a, w_a = shape_a
    h_b, w_b = shape_b
    x0 = max(0, int(np.ceil(dx)))
    y0 = max(0, int(np.ceil(dy)))
    x1 = min(w_a, int(np.floor(w_b + dx)))
    y1 = min(h_a, int(np.floor(h_b + dy)))
    ov_w = x1 - x0
    ov_h = y1 - y0
    if ov_w < MIN_OVERLAP_PX or ov_h < MIN_OVERLAP_PX:
        return None
    a_sl = (y0, y1, x0, x1)
    b_sl = (y0 - dy, y1 - dy, x0 - dx, x1 - dx)
    return a_sl, b_sl


def _overlap_stats(a_f: np.ndarray, b_f: np.ndarray, dx: int, dy: int):
    sl = _overlap_slices(a_f.shape, b_f.shape, dx, dy)
    if sl is None:
        return None
    (ay0, ay1, ax0, ax1), (by0, by1, bx0, bx1) = sl
    ra = a_f[ay0:ay1, ax0:ax1]
    rb = b_f[round(by0):round(by1), round(bx0):round(bx1)]
    if ra.shape != rb.shape:
        return None
    mad = float(np.abs(ra - rb).mean())
    va = ra.ravel()
    vb = rb.ravel()
    if float(va.std()) > 1e-6 and float(vb.std()) > 1e-6:
        corr = float(np.corrcoef(va, vb)[0, 1])
    else:
        corr = 0.0
    return mad, corr


def exhaustive_search(a_f: np.ndarray, b_f: np.ndarray):
    h_a, w_a = a_f.shape
    h_b, w_b = b_f.shape
    dx_lo = -(w_b - MIN_OVERLAP_PX)
    dx_hi = w_a - MIN_OVERLAP_PX
    dy_lim = int(min(MAX_ABS_DY + 7, max(4, min(h_a, h_b) // 8)))
    best = None
    for dy in range(-dy_lim, dy_lim + 1):
        for dx in range(dx_lo, dx_hi + 1):
            stats = _overlap_stats(a_f, b_f, dx, dy)
            if stats is None:
                continue
            mad = stats[0]
            if best is None or mad < best[0]:
                best = (mad, dx, dy)
    if best is None:
        return None
    _, dx, dy = best
    full = _overlap_stats(a_f, b_f, dx, dy)
    if full is None:
        return None
    return dx, dy, full[0], full[1]


def parabolic_min(y_minus: float, y0: float, y_plus: float) -> float:
    denom = y_minus - 2.0 * y0 + y_plus
    if denom <= 1e-12:
        return 0.0
    delta = 0.5 * (y_minus - y_plus) / denom
    return float(np.clip(delta, -1.0, 1.0))


def _snap(value: float, tol: float = SNAP_TOL) -> float:
    rounded = round(value)
    if abs(value - rounded) <= tol:
        return float(rounded)
    return float(value)


def ecc_refine(a_f: np.ndarray, b_f: np.ndarray, dx: float, dy: float):
    """Refine with ECC; return (dx, dy) in the forward b->a convention or None."""
    w_init = np.array([[1.0, 0.0, -float(dx)], [0.0, 1.0, -float(dy)]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITERS, ECC_EPS)
    try:
        _cc, w = cv2.findTransformECC(a_f, b_f, w_init, cv2.MOTION_TRANSLATION, crit, None, ECC_GAUSS)
    except cv2.error:
        return None
    return float(-w[0, 2]), float(-w[1, 2])


def aligned_residual(a_f: np.ndarray, b_f: np.ndarray, matrix: np.ndarray):
    """MAD/corr/area of b warped onto a by a forward 2x3 map, over valid pixels."""
    h_a, w_a = a_f.shape
    h_b, w_b = b_f.shape
    m = np.asarray(matrix, np.float32).reshape(2, 3)
    warped = cv2.warpAffine(b_f, m, (w_a, h_a))
    mask = cv2.warpAffine(
        np.full((h_b, w_b), 255, np.uint8), m, (w_a, h_a), flags=cv2.INTER_NEAREST
    )
    valid = mask > 127
    area = int(valid.sum())
    if area < MIN_OVERLAP_PX * MIN_OVERLAP_PX:
        return None
    va = a_f[valid]
    vb = warped[valid]
    mad = float(np.abs(va - vb).mean())
    if float(va.std()) > 1e-6 and float(vb.std()) > 1e-6:
        corr = float(np.corrcoef(va, vb)[0, 1])
    else:
        corr = 0.0
    return mad, corr, area


def align_translation(
    a_f: np.ndarray, b_f: np.ndarray, use_ecc: bool = True
) -> tuple[PairAlignment | None, str]:
    if min(a_f.shape + b_f.shape) < MIN_OVERLAP_PX:
        return None, "tiles smaller than the minimum overlap"
    search = exhaustive_search(a_f, b_f)
    if search is None:
        return None, "no integer offset with a usable overlap"
    dx0, dy0, mad0, _corr0 = search
    if mad0 > MAD_MAX:
        return None, (
            f"best integer offset dx={dx0} dy={dy0} has MAD {mad0:.1f} > {MAD_MAX}"
        )
    seeds = [(float(dx0), float(dy0))]
    dm = _overlap_stats(a_f, b_f, dx0 - 1, dy0)
    dp = _overlap_stats(a_f, b_f, dx0 + 1, dy0)
    if dm is not None and dp is not None:
        step = parabolic_min(dm[0], mad0, dp[0])
        if step != 0.0:
            seeds.append((dx0 + step, float(dy0)))
    best = None
    for sx, sy in seeds:
        residual = aligned_residual(a_f, b_f, np.array([[1, 0, sx], [0, 1, sy]], np.float32))
        if residual is None:
            continue
        mad, corr, _area = residual
        if best is None or mad < best[0]:
            best = (mad, corr, sx, sy)
    if use_ecc and best is not None:
        refined = ecc_refine(a_f, b_f, best[2], best[3])
        if refined is not None:
            residual = aligned_residual(
                a_f, b_f, np.array([[1, 0, refined[0]], [0, 1, refined[1]]], np.float32)
            )
            if residual is not None and residual[0] < best[0]:
                best = (residual[0], residual[1], refined[0], refined[1])
    if best is None:
        return None, "residual evaluation failed for every candidate offset"
    mad, corr, dx, dy = best
    if mad > MAD_MAX or corr < CORR_MIN or abs(dy) > MAX_ABS_DY:
        return None, (
            f"aligned residual out of gate: MAD={mad:.2f} (max {MAD_MAX}), "
            f"corr={corr:.3f} (min {CORR_MIN}), dy={dy:.2f} (max {MAX_ABS_DY})"
        )
    dx = _snap(dx)
    dy = _snap(dy)
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
    return PairAlignment("translation", matrix, mad, corr, dx, dy), ""


def align_sift(
    a_f: np.ndarray, b_f: np.ndarray
) -> tuple[PairAlignment | None, str]:
    gray_a = np.clip(a_f, 0, 255).astype(np.uint8)
    gray_b = np.clip(b_f, 0, 255).astype(np.uint8)
    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return None, "not enough SIFT keypoints"
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in matches if m.distance < SIFT_RATIO * n.distance]
    if len(good) < 4:
        return None, f"only {len(good)} Lowe-ratio matches"
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.estimateAffine2D(
        pts_b, pts_a, method=cv2.RANSAC, rprojThresh=RANSAC_REPROJ
    )
    if matrix is None or inlier_mask is None:
        return None, "estimateAffine2D failed"
    inliers = int(inlier_mask.sum())
    if inliers < SIFT_MIN_INLIERS:
        return None, f"only {inliers} RANSAC inliers (need {SIFT_MIN_INLIERS})"
    projected = cv2.transform(pts_b, matrix)
    errors = np.linalg.norm(projected - pts_a, axis=2).reshape(-1)
    median_err = float(np.median(errors[inlier_mask.ravel() > 0]))
    if median_err > RANSAC_REPROJ:
        return None, f"median reprojection error {median_err:.2f}px"
    h_b, w_b = gray_b.shape
    corners = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
    warped_corners = cv2.transform(corners, matrix).reshape(-1, 2)
    span = warped_corners.max(0) - warped_corners.min(0)
    if not (
        SPAN_MIN_FACTOR * w_b <= span[0] <= SPAN_MAX_FACTOR * w_b
        and SPAN_MIN_FACTOR * h_b <= span[1] <= SPAN_MAX_FACTOR * h_b
    ):
        return None, f"projected span {tuple(np.round(span, 1))} out of bounds"
    residual = aligned_residual(a_f, b_f, matrix)
    if residual is None:
        return None, "affine warp leaves too little overlap"
    mad, corr, _area = residual
    if mad > MAD_MAX or corr < CORR_MIN:
        return None, (
            f"affine residual out of gate: MAD={mad:.2f}, corr={corr:.3f}"
        )
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    return PairAlignment(
        "sift-affine", matrix.astype(np.float32), mad, corr, dx, dy, inliers
    ), ""


def align_pair(
    a_f: np.ndarray,
    b_f: np.ndarray,
    use_ecc: bool = True,
    use_sift: bool = True,
) -> PairAlignment:
    translation, tr_err = align_translation(a_f, b_f, use_ecc=use_ecc)
    if translation is not None:
        return translation
    if use_sift:
        affine, sf_err = align_sift(a_f, b_f)
        if affine is not None:
            return affine
    else:
        sf_err = "disabled"
    raise StitchError(
        f"cannot align pair: translation failed ({tr_err}); SIFT fallback failed ({sf_err})"
    )


def _as_3x3(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(matrix, np.float32), np.float32([0.0, 0.0, 1.0])])


def chain_matrices(alignments: list[PairAlignment]) -> list[np.ndarray]:
    mats = [np.eye(2, 3, dtype=np.float32)]
    for alignment in alignments:
        mats.append((mats[-1] @ _as_3x3(alignment.matrix))[:2].astype(np.float32))
    return mats


def compute_canvas(tiles: list[np.ndarray], mats: list[np.ndarray]):
    points = []
    for img, forward in zip(tiles, mats, strict=True):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        points.append(cv2.transform(corners, forward).reshape(-1, 2))
    all_points = np.vstack(points)
    mn = np.floor(all_points.min(0))
    mx = np.ceil(all_points.max(0))
    canvas_w = int(mx[0] - mn[0])
    canvas_h = int(mx[1] - mn[1])
    if canvas_w <= 0 or canvas_h <= 0:
        raise StitchError("degenerate canvas size")
    return canvas_w, canvas_h, mn.astype(np.float32)


def compose(
    tiles: list[np.ndarray],
    mats: list[np.ndarray],
    canvas_w: int,
    canvas_h: int,
    origin: np.ndarray,
) -> tuple[np.ndarray, float]:
    canvas = np.zeros((canvas_h, canvas_w, 3), np.float32)
    weight = np.zeros((canvas_h, canvas_w), np.float32)
    for img, forward in zip(tiles, mats, strict=True):
        m = np.asarray(forward, np.float32).copy()
        m[0, 2] -= float(origin[0])
        m[1, 2] -= float(origin[1])
        warped = cv2.warpAffine(img.astype(np.float32), m, (canvas_w, canvas_h))
        mask = cv2.warpAffine(
            np.full(img.shape[:2], 255, np.uint8),
            m,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
        )
        valid = mask > 127
        if not valid.any():
            continue
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3).astype(np.float32)
        wgt = dist + 1.0
        wgt[~valid] = 0.0
        canvas += warped * wgt[..., None]
        weight += wgt
    coverage = float((weight > 0).mean())
    if coverage == 0.0:
        raise StitchError("composition produced an empty canvas")
    blended = canvas / np.maximum(weight, 1e-9)[..., None]
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8), coverage


def validate_output(pano: np.ndarray, tiles: list[np.ndarray], coverage: float) -> None:
    ph, pw = pano.shape[:2]
    max_w = max(t.shape[1] for t in tiles)
    max_h = max(t.shape[0] for t in tiles)
    sum_w = sum(t.shape[1] for t in tiles)
    sum_h = sum(t.shape[0] for t in tiles)
    problems = []
    if pw < max_w or ph < max_h:
        problems.append(
            f"panorama {pw}x{ph} is smaller than the largest tile {max_w}x{max_h}"
        )
    if pw > sum_w or ph > sum_h:
        problems.append(
            f"panorama {pw}x{ph} exceeds the sum of tile extents "
            f"{sum_w}x{sum_h} (degenerate warp)"
        )
    if coverage < COVERAGE_MIN:
        problems.append(f"coverage {coverage:.0%} below {COVERAGE_MIN:.0%}")
    if problems:
        raise StitchError("output validation failed: " + "; ".join(problems))


def stitch_images(
    tiles: list[np.ndarray],
    use_ecc: bool = True,
    use_sift: bool = True,
) -> tuple[np.ndarray, StitchReport]:
    if len(tiles) < 2:
        raise StitchError("need at least 2 tiles")
    grays_f = [_gray_f32(t) for t in tiles]
    report = StitchReport()
    for k in range(len(tiles) - 1):
        try:
            alignment = align_pair(grays_f[k], grays_f[k + 1], use_ecc=use_ecc, use_sift=use_sift)
        except StitchError as exc:
            raise StitchError(f"pair {k + 1}->{k + 2}: {exc}") from exc
        report.pairs.append(alignment)
    mats = chain_matrices(report.pairs)
    canvas_w, canvas_h, origin = compute_canvas(tiles, mats)
    pano, coverage = compose(tiles, mats, canvas_w, canvas_h, origin)
    validate_output(pano, tiles, coverage)
    report.canvas = (canvas_w, canvas_h)
    report.coverage = coverage
    return pano, report


def format_report(report: StitchReport, names: list[str] | None = None) -> str:
    lines = ["Registration report"]
    for idx, pair in enumerate(report.pairs):
        label = (
            f"{names[idx]} -> {names[idx + 1]}"
            if names is not None and idx + 1 < len(names)
            else f"pair {idx + 1}->{idx + 2}"
        )
        extra = f"  inliers={pair.inliers}" if pair.inliers is not None else ""
        lines.append(
            f"  {label}: {pair.method}  dx={pair.dx:+.2f}  dy={pair.dy:+.2f}  "
            f"MAD={pair.mad:.3f}  corr={pair.corr:.3f}{extra}"
        )
    lines.append(f"  canvas   : {report.canvas[0]} x {report.canvas[1]}")
    lines.append(f"  coverage : {report.coverage:.1%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validated ordered-tile panorama stitcher (see module docstring)"
    )
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--dir", default=here, help="directory holding the tile images")
    parser.add_argument(
        "--output", default=None, help="output image path (default: <dir>/panorama.jpg)"
    )
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_NAMES, help="tile filenames in order")
    parser.add_argument("--no-ecc", action="store_true", help="skip ECC subpixel refinement")
    parser.add_argument("--no-sift", action="store_true", help="disable the SIFT affine fallback")
    args = parser.parse_args(argv)

    try:
        tiles = load_tiles(args.dir, args.inputs)
        pano, report = stitch_images(tiles, use_ecc=not args.no_ecc, use_sift=not args.no_sift)
    except StitchError as exc:
        print(f"Stitching failed: {exc}", file=sys.stderr)
        return 1

    print(format_report(report, args.inputs))
    output = args.output or os.path.join(args.dir, "panorama.jpg")
    if not cv2.imwrite(output, pano):
        print(f"Cannot write {output}", file=sys.stderr)
        return 1
    print(f"  written  : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
