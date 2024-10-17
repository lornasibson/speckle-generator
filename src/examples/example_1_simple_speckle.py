"""
example: generating a simple speckle pattern
"""

from pathlib import Path
import numpy as np
from specklegenerator.specklegenerator import (
    Speckle,
    SpeckleData,
    show_image,
    save_image,
    mean_intensity_gradient,
    validate_speckle_data,
)


def main() -> None:
    """
    Speckle example: generate simple speckle pattern
    - Size of image can be specified
    - Radius of circle and the b/w ratio can be specified
    - Image displayed on screen
    - Image saved to specifed filename in specified location
    """
    filename = "speckle_new_threshold"
    directory = Path.cwd() / "images"
    speckle_data = SpeckleData(size_x=400,
                               size_y=400,
                               radius=7,
                               b_w_ratio=0.5,
                               white_bg=True,
                               bits=8,
                               )

    speckle = Speckle(speckle_data)
    image = speckle.make()
    print(f"{mean_intensity_gradient(image)=}")
    show_image(image)
    save_image(image, directory, filename, speckle_data.bits)


if __name__ == "__main__":
    main()
