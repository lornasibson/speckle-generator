'''
example: generating a simple speckle pattern
'''
import sys
sys.path.insert(0, '/home/lorna/speckle_generator/src/speckle')
from speckle import Speckle

def main() -> None:
    '''
    Speckle example: generate simple speckle pattern
    - Size of image can be specified
    - Radius of circle and the b/w ratio can be specified
    - Image displayed on screen
    - Image saved to specifed filename in specified location
    '''
    size_x = 1000
    size_y = 1000
    radius = 10
    proportion_goal = 50
    filename = 'speckle_pattern_opt_radius'
    file_format = 'tiff'
    directory = '/home/lorna/speckle_generator'
    white_bg = True #Set to True for white background with black speckles, set to False for black background with white speckles
    image_res = 100
    speckle = Speckle(size_x, size_y, radius, proportion_goal, filename, file_format, directory, white_bg, image_res)
    speckle.generate_speckle() 

if __name__ == '__main__':
    main()