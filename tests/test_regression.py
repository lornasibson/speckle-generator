"""
TEST: Regression test for Speckle class
"""

import os
import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import numpy.testing as npt
from specklegenerator.specklegenerator import Speckle, SpeckleData, FileFormat, save_image

@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup
    yield
    # Teardown - remove output files
    test_dir = Path.cwd() / "images"
    all_files = os.listdir(test_dir)
    for ff in all_files:
        if "test" in ff:
            os.remove(test_dir / ff)

def test_happy_speckle_in_mem():
    seed = 8
    data = SpeckleData(
        size_x=100,
        size_y=100,
        radius=5,
        b_w_ratio=0.5,
        white_bg=True,
        image_res=200,
        file_format=FileFormat.TIFF,
        bits=8,
    )
    speckler = Speckle(data, seed)
    image = speckler.make()

    # Load ref image
    ref_image = Image.open(
        "/home/lorna/speckle_generator/images/regression_image_in_mem.tiff"
    )
    ref_image_array = np.asarray(ref_image)

    npt.assert_array_equal(image, ref_image_array)

def test_happy_speckle_in_loop():
    seed = 8
    data = SpeckleData(
        size_x=700,
        size_y=700,
        radius=5,
        bits = 8
    )
    speckler = Speckle(data, seed)
    image = speckler.make()

    # Load ref image
    ref_image = Image.open(
        "/home/lorna/speckle_generator/images/regression_image_loop.tiff"
    )
    ref_image_array = np.asarray(ref_image)

    npt.assert_array_equal(image, ref_image_array)

def test_happy_speckle_larger_radius():
    seed = 8
    data = SpeckleData(
        size_x=700,
        size_y=700,
        radius=10,
        bits = 8
    )
    speckler = Speckle(data, seed)
    image = speckler.make()

    # Load ref image
    ref_image = Image.open(
        "/home/lorna/speckle_generator/images/regression_image_larger_radius.tiff"
    )
    ref_image_array = np.asarray(ref_image)

    npt.assert_array_equal(image, ref_image_array)

def test_happy_speckle_black_bg_small():
    seed = 8
    data = SpeckleData(
        size_x=100,
        size_y=100,
        radius=5,
        white_bg=False,
        bits=8,
    )
    speckler = Speckle(data, seed)
    image = speckler.make()

    # Load ref image
    ref_image = Image.open(
        "/home/lorna/speckle_generator/images/regression_image_black_bg_small.tiff"
    )
    ref_image_array = np.asarray(ref_image)

    npt.assert_array_equal(image, ref_image_array)

def test_happy_speckle_black_bg_large():
    seed = 8
    data = SpeckleData(
        size_x=700,
        size_y=700,
        radius=5,
        white_bg=False,
        bits = 8
    )
    speckler = Speckle(data, seed)
    image = speckler.make()

    # Load ref image
    ref_image = Image.open(
        "/home/lorna/speckle_generator/images/regression_image_black_bg_large.tiff"
    )
    ref_image_array = np.asarray(ref_image)

    npt.assert_array_equal(image, ref_image_array)


def test_happy_speckle_bmp():
    seed = 8
    data = SpeckleData(
        size_x=100,
        size_y=100,
        radius=5,
        white_bg=True,
        file_format=FileFormat.BITMAP,
        bits = 8
    )
    speckler = Speckle(data, seed)
    image = speckler.make()

    # Load ref image
    ref_image = Image.open(
        "/home/lorna/speckle_generator/images/regression_image_bmp.bmp"
    )
    ref_image_array = np.asarray(ref_image)

    npt.assert_array_equal(image, ref_image_array)


def test_happy_speckle_image_res():
    seed = 8
    image_res = 300
    data = SpeckleData(
        size_x=100,
        size_y=100,
        radius=5,
        white_bg=True,
        image_res=image_res,
        bits = 8
    )
    speckler = Speckle(data, seed)
    image = speckler.make()
    directory = Path.cwd() / "images"
    filename = "regression_test_image_res"
    save_image(image, directory, filename, data.bits, data.file_format, data.image_res)

    saved_image = Image.open(
        "/home/lorna/speckle_generator/images/regression_test_image_res.tiff"
    )
    xres, yres = saved_image.info['dpi']

    assert xres == image_res







