"""
example: generating a simple speckle pattern
"""

from pathlib import Path
from specklegenerator.specklegenerator import (
    Speckle,
    SpeckleData,
    FileFormat,
    show_image,
    save_image,
    mean_intensity_gradient,
)


def main() -> None:
    """
    Speckle example: generate simple speckle pattern
    - Size of image can be specified
    - Radius of circle and the b/w ratio can be specified
    - Image displayed on screen
    - Image saved to specifed filename in specified location
    """
    filename = "test_image_res_300"
    directory = Path.cwd() / "images"
    speckle_data = SpeckleData(size_x=600,
                               size_y=600,
                               radius=5,
                               b_w_ratio=0.5,
                               white_bg=True,
                               bits=8,
                               )

    speckle = Speckle(speckle_data, seed=8)
    image = speckle.make()
    print(f"{mean_intensity_gradient(image)=}")
    show_image(image)
    # save_image(image, directory, filename, speckle_data.bits, speckle_data.file_format,
            #    speckle_data.image_res)



if __name__ == "__main__":
    main()
