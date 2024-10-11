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
    _colour_count,
)
# mean_intensity_gradient)


def main() -> None:
    """
    Speckle example: generate simple speckle pattern
    - Size of image can be specified
    - Radius of circle and the b/w ratio can be specified
    - Image displayed on screen
    - Image saved to specifed filename in specified location
    """
    filename = "speckle_bg"
    directory = Path.cwd() / "images"
    speckle_data = SpeckleData(size_x=500,
                               size_y=500,
                               radius=10,
                               b_w_ratio=0.75,
                               white_bg=False,
                               bits=8,
                               )

    speckle = Speckle(speckle_data)

    image = speckle.make()
    print(f"{mean_intensity_gradient(image)=}")
    save_image(image, directory, filename, bits=speckle_data.bits)



if __name__ == "__main__":
    main()
