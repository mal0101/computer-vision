"""Regression tests for the panorama stitcher against the bundled tiles."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (  # noqa: E402
    MAD_MAX,
    StitchError,
    load_tiles,
    stitch_images,
    validate_output,
)

PANORAMA_DIR = os.path.dirname(os.path.abspath(__file__))
NAMES = [f"part_{i}.jpg" for i in range(1, 6)]
STRIDE = 112
EXPECTED_W = STRIDE * 4 + 164
EXPECTED_H = 154


@pytest.fixture(scope="module")
def tiles():
    return load_tiles(PANORAMA_DIR, NAMES)


@pytest.fixture(scope="module")
def stitched(tiles):
    return stitch_images(tiles)


def test_tiles_load_with_expected_shapes(tiles):
    assert len(tiles) == 5
    for tile in tiles[:4]:
        assert tile.shape == (EXPECTED_H, 161, 3)
    assert tiles[4].shape == (EXPECTED_H, 164, 3)


def test_pair_offsets_are_exact_stride(stitched):
    _pano, report = stitched
    assert len(report.pairs) == 4
    for pair in report.pairs:
        assert pair.method == "translation"
        assert abs(pair.dx - STRIDE) <= 2
        assert abs(pair.dy) <= 2
        assert pair.mad < 1.0
        assert pair.corr > 0.99


def test_canvas_dimensions(stitched):
    pano, report = stitched
    assert report.canvas == (EXPECTED_W, EXPECTED_H)
    assert pano.shape == (EXPECTED_H, EXPECTED_W, 3)


def test_coverage_full(stitched):
    _pano, report = stitched
    assert report.coverage >= 0.95


def test_output_matches_integer_placement(tiles, stitched):
    pano, _report = stitched
    ref = np.zeros((EXPECTED_H, EXPECTED_W, 3), np.uint8)
    overlap = np.zeros((EXPECTED_H, EXPECTED_W), bool)
    for k, tile in enumerate(tiles):
        x0 = STRIDE * k
        ref[:, x0 : x0 + tile.shape[1]] = tile
    for k in range(4):
        overlap[:, STRIDE * (k + 1) : STRIDE * (k + 1) + 49] = True
    diff = np.abs(pano.astype(int) - ref.astype(int))
    assert diff[~overlap].max() == 0
    assert float(diff[overlap].mean()) < MAD_MAX


def test_deterministic(tiles):
    pano_a, _ = stitch_images(tiles)
    pano_b, _ = stitch_images(tiles)
    assert (pano_a == pano_b).all()


def test_no_ecc_path_still_correct(tiles):
    pano, report = stitch_images(tiles, use_ecc=False)
    assert pano.shape == (EXPECTED_H, EXPECTED_W, 3)
    assert all(abs(p.dx - STRIDE) <= 2 for p in report.pairs)


def test_output_not_smaller_than_inputs(tiles, stitched):
    pano, _report = stitched
    max_w = max(t.shape[1] for t in tiles)
    max_h = max(t.shape[0] for t in tiles)
    assert pano.shape[1] >= max_w
    assert pano.shape[0] >= max_h


def test_validate_output_rejects_collapsed_pano(tiles):
    tiny = np.zeros((31, 63, 3), np.uint8)
    with pytest.raises(StitchError, match="smaller than the largest tile"):
        validate_output(tiny, tiles, coverage=1.0)


def test_validate_output_rejects_exploded_pano(tiles):
    huge = np.zeros((2084, 989, 3), np.uint8)
    with pytest.raises(StitchError, match="exceeds the sum"):
        validate_output(huge, tiles, coverage=1.0)


def test_validate_output_rejects_low_coverage(tiles):
    pano = np.zeros((EXPECTED_H, EXPECTED_W, 3), np.uint8)
    with pytest.raises(StitchError, match="coverage"):
        validate_output(pano, tiles, coverage=0.3)


def test_unalignable_tiles_rejected():
    a = np.full((120, 150, 3), 128, np.uint8)
    b = np.zeros((120, 150, 3), np.uint8)
    with pytest.raises(StitchError, match="pair 1->2"):
        stitch_images([a, b])


def test_single_tile_rejected(tiles):
    with pytest.raises(StitchError, match="at least 2"):
        stitch_images(tiles[:1])


def test_missing_file_rejected(tmp_path):
    with pytest.raises(StitchError, match="cannot read image"):
        load_tiles(str(tmp_path), ["nope.jpg"])


def test_cli_end_to_end(tmp_path, tiles):
    import main as panorama_main

    out = tmp_path / "out.jpg"
    code = panorama_main.main(
        ["--dir", PANORAMA_DIR, "--inputs", *NAMES, "--output", str(out)]
    )
    assert code == 0
    result = cv2.imread(str(out))
    assert result is not None
    assert result.shape == (EXPECTED_H, EXPECTED_W, 3)
