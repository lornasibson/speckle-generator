import os
from pathlib import Path
import numpy as np
from specklegenerator.specklegenerator import Speckle, SpeckleData, save_image, _colour_count

filename = 'save_image_test'
directory = Path.cwd() / "images"
filename_full = filename + '.' + SpeckleData.file_format

all_files = os.listdir(directory)
for ff in all_files:
    if ff == filename_full:
        print(True)

