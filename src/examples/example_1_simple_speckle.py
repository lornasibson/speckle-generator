'''
example: generating a simple speckle pattern
'''
from pathlib import Path
from specklegenerator.specklegenerator import Speckle, SpeckleData, show_image, save_image

def main() -> None:
    '''
    Speckle example: generate simple speckle pattern
    - Size of image can be specified
    - Radius of circle and the b/w ratio can be specified
    - Image displayed on screen
    - Image saved to specifed filename in specified location
    '''

    filename = 'speckle_pattern_set0'
    file_format = 'tiff'
    directory = Path.cwd() / "images"
    speckle_data = SpeckleData()
    speckle = Speckle(filename, file_format, directory, speckle_data)
    image = speckle.generate_speckle()
    show_image(image)
    save_image(image)

if __name__ == '__main__':
    main()