import os
from pathlib import Path
import pytest
import numpy as np
from specklegenerator.specklegenerator import (Speckle,
                                               SpeckleData,
                                               _random_location,
                                               _px_locations,
                                               save_image,
                                               _colour_count,
                                               _threshold_image)

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
    data = SpeckleData(size_x=100, size_y=100, radius=5, proportion_goal=0.5)
    speckler = Speckle(data)
    num_dots_x, num_dots_y, n_tot = speckler._optimal_dot_number()

    assert num_dots_x == 8
    assert num_dots_y == 8
    assert n_tot == 64

def test_dot_locations(): # NEED TO FIX
    seed = 5
    data = SpeckleData(size_x=10, size_y=10)
    speckler = Speckle(data, seed)
    x_dot_2d, y_dot_2d = speckler._dot_locations(4, 4, 16)

    dot_location_x = np.array([1.25, 3.75, 6.25, 8.75])
    dot_location_y = np.array([1.25, 3.75, 6.25, 8.75])
    dot_x_grid, dot_y_grid = np.meshgrid(dot_location_x, dot_location_y)
    x_dot_vec = dot_x_grid.flatten()
    y_dot_vec = dot_y_grid.flatten()
    random_array = _random_location(seed, 1, 16)
    random_dots_x = np.add(x_dot_vec, random_array)
    random_dots_y = np.add(y_dot_vec, random_array)
    random_dots_2d_x = np.atleast_2d(random_dots_x)
    random_dots_2d_y = np.atleast_2d(random_dots_y)

    # assert x_dot_2d == pytest.approx(random_dots_2d_x)
    # assert y_dot_2d == pytest.approx(random_dots_2d_y)

def test_random_location():
    seed = 5
    data = SpeckleData(radius=1)
    speckler = Speckle(data, seed)
    n_dots = 16
    random_array = _random_location(speckler.seed, speckler.speckle_data.radius,
                                    n_dots)

    random_locations = np.array([-0.36451428, -0.60198136, -0.11289165,
                                 0.19111147, 0.51638479, 0.04986655,
                                 -0.25120333, -0.35671834, 0.34033899,
                                 0.7430832, 0.12398581, -0.56060394,
                                 -0.43557509, 0.7272814, 0.09221929, -0.78733402])

    assert random_array == pytest.approx(random_locations)

def test_px_locations():
    size_x = 4
    size_y = 4

    grid_shape, x_px_trans, y_px_trans = _px_locations(size_x, size_y)

    x_px = np.array([[0.5],
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
       [3.5]])
    y_px = np.array([[0.5],
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
       [3.5]])

    assert grid_shape == (4, 4)
    assert x_px_trans == pytest.approx(x_px)
    assert y_px_trans == pytest.approx(y_px)

def test_save_image():
    filename = 'save_image_test'
    directory = Path.cwd() / "images"
    data = SpeckleData()
    speckle = Speckle(data)
    image = speckle.make()

    save_image(image, directory, filename)

    filename_full = filename + '.' + data.file_format
    all_files = os.listdir(directory)
    test_file = ''
    for ff in all_files:
        if ff == filename_full:
            test_file = ff

    assert test_file == filename_full

def test_colour_count():
    image = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
    size_x = 4
    size_y = 4

    proportion = _colour_count(size_x, size_y, image)

    assert proportion == 0.5

def test_threshold_image():
    radius = 5
    dist = np.array([10, 7.5, 7, 6, 5.5, 5.4, 5.3, 5.2, 5, 4])

    image = np.zeros((10, 10))

    correct_image = np.array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
       [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
       [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
       [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ],
       [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ]])

    image_output = _threshold_image(radius, image, dist)

    assert image_output == pytest.approx(correct_image)

def test_colour_switch():
    image = np.array([[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5],
                      [1, 1, 1, 1], [0.8, 0.8, 0.8, 0.8]])

    correct_output = np.array([[1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5],
                               [0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2]])

    image_switch = image * -1 + 1

    assert image_switch == pytest.approx(correct_output)










