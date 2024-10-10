"""
example: generating a simple speckle pattern
"""

from pathlib import Path
from specklegenerator.specklegenerator import (
    Speckle,
    SpeckleData,
    show_image,
    save_image,
    mean_intensity_gradient,
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
    filename = "speckle_large"
    directory = Path.cwd() / "images"
    speckle_data = SpeckleData()

    speckle = Speckle(speckle_data)
    bad_parameter = speckle._check_parameters()
    if bad_parameter is True:
        print('An unsuitable input parameter has be chosen, so the programme cannot run')
    else:
        image = speckle.make()
        show_image(image)
        # save_image(image, directory, filename)
        # mean_intensity_gradient(image)


if __name__ == "__main__":
    main()
