"""
TEST: Input parameters to Speckle class
"""

import pytest
from specklegenerator.specklegenerator import Speckle, SpeckleData, FileFormat


@pytest.mark.parametrize(
    "size_x, output",
    [
        pytest.param(0, True, id="size_x equal to 0"),
        pytest.param(2, True, id="size_x too small"),
        pytest.param("100", True, id="size_x not an integer"),
        pytest.param(-100, True, id="size_x negative"),
        pytest.param(100, False, id="size_x acceptable value"),
    ],
)
def test_size_inputs(size_x, output):
    data = SpeckleData(size_x=size_x)
    speckle = Speckle(data)
    bad_parameter = speckle._check_parameters()
    assert bad_parameter == output


@pytest.mark.parametrize(
    "radius, output",
    [
        pytest.param(0, True, id="radius equal to 0"),
        pytest.param(600, True, id="radius too big"),
        pytest.param("10", True, id="radius not an integer"),
        pytest.param(-10, True, id="radius negative"),
        pytest.param(10, False, id="radius acceptable value"),
    ],
)
def test_radius_inputs(radius, output):
    data = SpeckleData(radius=radius)
    speckle = Speckle(data)
    bad_parameter = speckle._check_parameters()
    assert bad_parameter == output


@pytest.mark.parametrize(
    "prop_goal, output",
    [
        pytest.param(0, True, id="prop goal equal to 0"),
        pytest.param(1, True, id="prop goal equal to 1"),
        pytest.param("0.5", True, id="prop goal not a float"),
        pytest.param(-0.5, True, id="prop goal negative"),
        pytest.param(1.5, True, id="prop goal larger than 1"),
        pytest.param(0.5, False, id="prop goal acceptable value"),
    ],
)
def test_b_w_ratio_inputs(prop_goal, output):
    data = SpeckleData(b_w_ratio=prop_goal)
    speckle = Speckle(data)
    bad_parameter = speckle._check_parameters()
    assert bad_parameter == output


@pytest.mark.parametrize(
    "white_bg, output",
    [
        pytest.param("True", True, id="white bg not Boolean"),
        pytest.param(True, False, id="white bg True"),
        pytest.param(False, False, id="white bg false"),
    ],
)
def test_white_bg_inputs(white_bg, output):
    data = SpeckleData(white_bg=white_bg)
    speckle = Speckle(data)
    bad_parameter = speckle._check_parameters()
    assert bad_parameter == output


@pytest.mark.parametrize(
    "image_res, output",
    [
        pytest.param("200", True, id="image res not an integer"),
        pytest.param(0, True, id="image res equal to 0"),
        pytest.param(-200, True, id="image res negative"),
        pytest.param(200, False, id="image res acceptable value"),
    ],
)
def test_image_res_inputs(image_res, output):
    data = SpeckleData(image_res=image_res)
    speckle = Speckle(data)
    bad_parameter = speckle._check_parameters()

    assert bad_parameter == output


@pytest.mark.parametrize(
    "file_format, output",
    [
        pytest.param("tiff", True, id="non-enum value"),
        pytest.param(FileFormat.TIFF, False, id="file format correct"),
    ],
)
def test_file_format_inputs(file_format, output):
    data = SpeckleData(file_format=file_format)
    speckle = Speckle(data)
    bad_parameter = speckle._check_parameters()
    assert bad_parameter == output


@pytest.mark.parametrize(
    "bits, output",
    [
        pytest.param(1, True, id="bit size less than 2"),
        pytest.param(18, True, id="bit size great than 16"),
        pytest.param("8", True, id="bit size not an integer"),
        pytest.param(16, False, id="bit size acceptable value"),
    ],
)
def test_bits_inputs(bits, output):
    data = SpeckleData(bits=bits)
    speckle = Speckle(data)
    bad_parameter = speckle._check_parameters()

    assert bad_parameter == output
