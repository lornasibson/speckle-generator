import pytest
from PIL import Image
import numpy as np
from specklegenerator.specklegenerator import Speckle, SpeckleData, FileFormat


def test_happy_speckle():
    seed = 8
    SpeckleData(size_x=100, size_y=100, radius=5, proportion_goal=0.5, white_bg=True,
                image_res=200, file_format=FileFormat.TIFF.value, bits=8)
    speckler = Speckle(SpeckleData, seed)
    image = speckler.make()

    # Load ref image
    ref_image = Image.open('/home/lorna/speckle_generator/images/regression_image_100.tiff')
    ref_image_array = np.asarray(ref_image)

    assert image == pytest.approx(ref_image_array)
