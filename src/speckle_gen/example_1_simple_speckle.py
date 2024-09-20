from speckle import Speckle

if __name__ == '__main__':
    size_x = 1000
    size_y = 1000
    radius = 8
    proportion_goal = 50
    filename = 'speckle_pattern_opt_radius'
    file_format = 'tiff'
    directory = '/home/lorna/speckle_generator'
    white_bg = True #Set to True for white background with black speckles, set to False for black background with white speckles
    image_res = 100
    speckle = Speckle(size_x, size_y, radius, proportion_goal, filename, file_format, directory, white_bg, image_res)
    speckle.generate_speckle()