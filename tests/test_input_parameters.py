"""
TEST: Input parameters to Speckle class
"""

import pytest
from contextlib import nullcontext as does_not_raise
from specklegenerator.specklegenerator import (Speckle,
                                               SpeckleData,
                                               FileFormat,
                                               SpeckleError,
                                               validate_speckle_data)


@pytest.mark.parametrize(
    "size_x, expectation",
    [
        pytest.param(0, pytest.raises(SpeckleError), id="size_x equal to 0"),
        pytest.param(2, pytest.raises(SpeckleError), id="size_x too small"),
        pytest.param(-100, pytest.raises(SpeckleError), id="size_x negative"),
        pytest.param(100, does_not_raise(), id="size_x acceptable value"),
    ],
)
def test_size_inputs(size_x, expectation):
    data = SpeckleData(size_x=size_x)
    with expectation:
        validate_speckle_data(data)




@pytest.mark.parametrize(
    "radius, output",
    [
        pytest.param(0, pytest.raises(SpeckleError), id="radius equal to 0"),
        pytest.param(600, pytest.raises(SpeckleError), id="radius too big"),
        pytest.param(-10, pytest.raises(SpeckleError), id="radius negative"),
        pytest.param(10, does_not_raise(), id="radius acceptable value"),
    ],
)
def test_radius_inputs(radius, output):
    data = SpeckleData(radius=radius)
    with output:
        validate_speckle_data(data)


@pytest.mark.parametrize(
    "b_w_ratio, output",
    [
        pytest.param(0, pytest.raises(SpeckleError), id="prop goal equal to 0"),
        pytest.param(1, pytest.raises(SpeckleError), id="prop goal equal to 1"),
        pytest.param(-0.5, pytest.raises(SpeckleError), id="prop goal negative"),
        pytest.param(1.5, pytest.raises(SpeckleError), id="prop goal larger than 1"),
        pytest.param(0.5, does_not_raise(), id="prop goal acceptable value"),
    ],
)
def test_b_w_ratio_inputs(b_w_ratio, output):
    data = SpeckleData(b_w_ratio=b_w_ratio)
    with output:
        validate_speckle_data(data)


@pytest.mark.parametrize(
    "white_bg, output",
    [
        pytest.param(True, does_not_raise(), id="white bg True"),
        pytest.param(False, does_not_raise(), id="white bg false"),
    ],
)
def test_white_bg_inputs(white_bg, output):
    data = SpeckleData(white_bg=white_bg)
    with output:
        validate_speckle_data(data)


@pytest.mark.parametrize(
    "image_res, output",
    [
        pytest.param(0, pytest.raises(SpeckleError), id="image res equal to 0"),
        pytest.param(-200, pytest.raises(SpeckleError), id="image res negative"),
        pytest.param(200, does_not_raise(), id="image res acceptable value"),
    ],
)
def test_image_res_inputs(image_res, output):
    data = SpeckleData(image_res=image_res)
    with output:
        validate_speckle_data(data)


def test_file_format_inputs():
    fileformat = FileFormat.TIFF
    data = SpeckleData(file_format=fileformat)
    assert validate_speckle_data(data) is None


@pytest.mark.parametrize(
    "bits, output",
    [
        pytest.param(1, pytest.raises(SpeckleError), id="bit size less than 2"),
        pytest.param(18, pytest.raises(SpeckleError), id="bit size great than 16"),
        pytest.param(16, does_not_raise(), id="bit size acceptable value"),
    ],
)
def test_bits_inputs(bits, output):
    data = SpeckleData(bits=bits)
    with output:
        validate_speckle_data(data)
