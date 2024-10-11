"""
TEST: Function outputs in specklegenerator.py
"""

import os
from pathlib import Path
import pytest
from PIL import Image
import numpy as np
import numpy.testing as npt
from specklegenerator.specklegenerator import (
    Speckle,
    SpeckleData,
    )


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


def test_optimal_dot_number():
    data = SpeckleData(size_x=100, size_y=100, radius=5, b_w_ratio=0.5)
    speckler = Speckle(data)
    num_dots_x, num_dots_y, n_tot = speckler._optimal_dot_number()

    assert num_dots_x == 8
    assert num_dots_y == 8
    assert n_tot == 64


def test_dot_locations():
    from specklegenerator.specklegenerator import _random_location
    seed = 5
    data = SpeckleData(size_x=10, size_y=10, radius=1)
    speckler = Speckle(data, seed)
    x_dot_2d, y_dot_2d = speckler._dot_locations(4, 4, 16)

    dot_x_vec = np.array(
        [
            1.25,
            3.75,
            6.25,
            8.75,
            1.25,
            3.75,
            6.25,
            8.75,
            1.25,
            3.75,
            6.25,

            8.75,
            1.25,
            3.75,
            6.25,
            8.75,
        ]
    )
    dot_y_vec = np.array(
        [
            1.25,
            1.25,
            1.25,
            1.25,
            3.75,
            3.75,
            3.75,
            3.75,
            6.25,
            6.25,
            6.25,
            6.25,
            8.75,
            8.75,
            8.75,
            8.75,
        ]
    )
    random_array = _random_location(seed, 1, 16)
    random_dots_x = np.add(dot_x_vec, random_array)
    random_dots_y = np.add(dot_y_vec, random_array)
    random_dots_2d_x = np.atleast_2d(random_dots_x)
    random_dots_2d_y = np.atleast_2d(random_dots_y)

    npt.assert_array_equal(x_dot_2d, random_dots_2d_x,
                           err_msg="The dot locations in the x-dir are not equal")
    npt.assert_array_equal(y_dot_2d, random_dots_2d_y,
                           err_msg="The dot locations in the y-dir are not equal")

def test_random_location():
    from specklegenerator.specklegenerator import _random_location
    seed = 5
    data = SpeckleData(radius=1)
    speckler = Speckle(data, seed)
    n_dots = 16
    random_array = _random_location(speckler.seed, speckler.speckle_data.radius, n_dots)

    random_locations = np.array(
        [
            -0.36451428,
            -0.60198136,
            -0.11289165,
            0.19111147,
            0.51638479,
            0.04986655,
            -0.25120333,
            -0.35671834,
            0.34033899,
            0.7430832,
            0.12398581,
            -0.56060394,
            -0.43557509,
            0.7272814,
            0.09221929,
            -0.78733402,
        ]
    )

    npt.assert_array_almost_equal(random_array, random_locations)


def test_px_locations():
    from specklegenerator.specklegenerator import _px_locations
    size_x = 4
    size_y = 4

    grid_shape, x_px_trans, y_px_trans = _px_locations(size_x, size_y)

    x_px = np.array(
        [
            [0.5],
            [1.5],
            [2.5],
            [3.5],
            [0.5],
            [1.5],
            [2.5],
            [3.5],
            [0.5],
            [1.5],
            [2.5],
            [3.5],
            [0.5],
            [1.5],
            [2.5],
            [3.5],
        ]
    )
    y_px = np.array(
        [
            [0.5],
            [0.5],
            [0.5],
            [0.5],
            [1.5],
            [1.5],
            [1.5],
            [1.5],
            [2.5],
            [2.5],
            [2.5],
            [2.5],
            [3.5],
            [3.5],
            [3.5],
            [3.5],
        ]
    )

    assert grid_shape == (4, 4)
    npt.assert_array_equal(x_px_trans, x_px,
                           err_msg="The pixel locations in the x-dir are not equal")
    npt.assert_array_equal(y_px_trans, y_px,
                           err_msg="The pixel locations in the y-dir are not equal")


def test_save_image():
    from specklegenerator.specklegenerator import save_image
    filename = "save_image_test"
    directory = Path.cwd() / "images"
    data = SpeckleData()
    speckle = Speckle(data)
    image = speckle.make()

    save_image(image, directory, filename)

    filename_full = filename + "." + data.file_format.value
    all_files = os.listdir(directory)
    test_file = ""
    for ff in all_files:
        if ff == filename_full:
            test_file = ff

    assert test_file == filename_full


def test_colour_count():
    from specklegenerator.specklegenerator import _colour_count
    image = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
    size_x = 4
    size_y = 4

    proportion = _colour_count(size_x, size_y, image)

    assert proportion == 0.5


def test_threshold_image():
    from specklegenerator.specklegenerator import _threshold_image
    radius = 5
    dist = np.array([10, 7.5, 7, 6, 5.5, 5.4, 5.3, 5.2, 5, 4])

    image = np.zeros((10, 10))

    correct_image = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    image_output = _threshold_image(radius, image, dist)
    npt.assert_array_equal(image_output, correct_image)


def test_colour_switch():
    image = np.array(
        [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1], [0.8, 0.8, 0.8, 0.8]]
    )

    correct_output = np.array(
        [[1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2]]
    )

    image_switch = image * -1 + 1
    npt.assert_array_almost_equal(image_switch, correct_output)


@pytest.mark.parametrize(
    "bits, output",
    [
        pytest.param(2, (2**2 - 1), id="bits = 2"),
        pytest.param(4, (2**4 - 1), id="bits = 4"),
        pytest.param(8, (2**8 - 1), id="bits = 8"),
        pytest.param(12, (2**12 - 1), id="bits = 12"),
        pytest.param(16, (2**16 - 1), id="bits = 16"),
    ],
)
def test_bit_size_generated_image(bits, output):
    data = SpeckleData(bits=bits)
    speckle = Speckle(data)
    image = speckle.make()
    bit_size = np.max(image)
    assert bit_size == output


@pytest.mark.parametrize(
    "bits, output",
    [
        pytest.param(2, np.uint8, id="bits = 2"),
        pytest.param(4, np.uint8, id="bits = 4"),
        pytest.param(8, np.uint8, id="bits = 8"),
        pytest.param(12, np.uint16, id="bits = 12"),
        pytest.param(16, np.uint16, id="bits = 16"),
    ],
)
def test_bit_size_saved_image(bits, output):
    from specklegenerator.specklegenerator import save_image
    data = SpeckleData(bits=bits)
    speckle = Speckle(data)
    image = speckle.make()
    directory = Path.cwd() / "images"
    filename = "test_bit_size"
    save_image(image, directory, filename, bits)

    saved_image = Image.open("/home/lorna/speckle_generator/images/test_bit_size.tiff")
    saved_image_array = np.asarray(saved_image)

    bit_size = saved_image_array.dtype

    assert bit_size == output
