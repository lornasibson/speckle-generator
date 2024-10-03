from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class FileFormat(Enum):
    '''
    Enum class to restrict the allowable output file formats
    '''
    TIFF = 'tiff'
    BITMAP = 'bmp'

@dataclass
class SpeckleData:
    '''
    Data class to store default parameters
    Parameters:
            size_x (int): An integer value of the horizontal size of speckle image in pixels
            size_y (int): An integer value of the vertical size of speckle image in pixels
            radius (int): An integer value of the radius of circles used in the speckle pattern
            proportion_goal (int): An integer value of the desired black/white balance as a percentage
            white_bg (bool): A Boolean value for whether a white or black background is wanted
            image_res (int): An integer value of the desired image resolution in dpi
            file_format (enum): A string of the desired file format of the saved file
            bits (int): An integer value of the bit size of the image
    '''
    size_x: int = 500
    size_y:int = 500
    radius:int = 7
    proportion_goal:float = 0.5
    white_bg:bool = True
    image_res:int = 200
    file_format:FileFormat = FileFormat.TIFF
    gauss_blur: float = 1
    bits: int = 16


class Speckle:
    '''
    A class to generate, show and save a speckle pattern subject to input parameters
    '''
    def __init__(self,
                 speckle_data: SpeckleData,
                 seed: int | None = None) -> None:
        self.speckle_data = speckle_data
        self.seed = seed

    def _check_parameters(self):
        '''
        Method to check the input parameters of the class to ensure that they are reasonable
            Returns:
                bad_parameter (bool): A Boolean value indicating whether class
                    instance contains a bad parameter or not
        '''
        # Sizes
        bad_parameter = False
        if not (isinstance(self.speckle_data.size_x, int) and
              isinstance(self.speckle_data.size_y, int)):
            print('The image size must be an integer')
            bad_parameter = True
            return bad_parameter
        elif self.speckle_data.size_x == 0 or self.speckle_data.size_y == 0:
            print('The image size cannot be 0, please enter a suitable integer')
            bad_parameter = True
        elif self.speckle_data.size_x <0 or self.speckle_data.size_y <0:
            print('The image size cannot be negative')
            bad_parameter = True
        elif (self.speckle_data.size_x <= 20 or
              self.speckle_data.size_y <= 20):
            print('The image size is too small compared to the speckle radius')
            bad_parameter = True


        # Radius
        if not isinstance(self.speckle_data.radius, int):
            print('The radius must be an integer')
            bad_parameter = True
            return bad_parameter
        elif self.speckle_data.radius == 0:
            print('The radius cannot be 0, please enter a suitable integer')
            bad_parameter = True
        elif self.speckle_data.radius <0:
            print('The radius cannot be negative')
            bad_parameter = True
        elif (self.speckle_data.radius > (self.speckle_data.size_x / 2) or
              self.speckle_data.radius > self.speckle_data.size_y / 2):
            print('The radius is too large compared to the image size')
            bad_parameter = True

        # Proportion
        if not isinstance(self.speckle_data.proportion_goal, float):
            print('The proportion goal must be a float')
            bad_parameter = True
            return bad_parameter
        elif  (self.speckle_data.proportion_goal < 0.0 or
               self.speckle_data.proportion_goal > 1.0):
            print('The proportion goal must be between 0 and 1')
            bad_parameter = True
        elif self.speckle_data.proportion_goal == 0.0:
            print('The proportion goal cannot be 0')
            bad_parameter = True
        elif self.speckle_data.proportion_goal == 1.0:
            print('The proportion goal cannot be 1')
            bad_parameter = True

        # White bg
        if not isinstance(self.speckle_data.white_bg, bool):
            print('The white background parameter must be a Boolean')
            bad_parameter = True

        # Image resolution
        if not isinstance(self.speckle_data.image_res, int):
            print('The image resolution must be an integer')
            bad_parameter = True
            return bad_parameter
        elif  self.speckle_data.image_res < 0.0:
            print('The image resolution cannot be negative')
            bad_parameter = True
        elif self.speckle_data.image_res == 0.0:
            print('The image resolution cannot be 0')
            bad_parameter = True

        # File format
        if not isinstance(self.speckle_data.file_format, FileFormat):
            print('The file format has to be an enum, defined in the FileFormat class')
            bad_parameter = True

        # Bits
        if not isinstance(self.speckle_data.bits, int):
            print('The bit size must be an integer')
            bad_parameter = True
            return bad_parameter
        elif  self.speckle_data.bits < 2:
            print('The bit size cannot be smaller than 2')
            bad_parameter = True
        elif self.speckle_data.bits > 16:
            print('The bit size cannot be larger than 16')
            bad_parameter = True

        return bad_parameter

    def make(self) -> np.ndarray:
        '''
        Produces a random speckle pattern with given parameters
            Returns:
                image (np.ndarray): An image array containing a speckle pattern
        '''
        bad_parameter = self._check_parameters()
        if bad_parameter is True:
            return

        num_dots_x, num_dots_y, n_tot = self._optimal_dot_number()
        x_dot_2d, y_dot_2d = self._dot_locations(num_dots_x, num_dots_y, n_tot)
        grid_shape, x_px_trans, y_px_trans = _px_locations(self.speckle_data.size_x,
                                                          self.speckle_data.size_y)

        x_px_same_dim = np.repeat(x_px_trans, n_tot, axis=1)
        x_dot_same_dim = np.repeat(x_dot_2d, (self.speckle_data.size_x *
                                              self.speckle_data.size_y), axis=0)
        y_px_same_dim = np.repeat(y_px_trans, n_tot, axis=1)
        y_dot_same_dim = np.repeat(y_dot_2d, (self.speckle_data.size_x *
                                              self.speckle_data.size_y), axis=0)
        del(x_px_trans, x_dot_2d, y_px_trans, y_dot_2d)

        dist = np.sqrt((x_dot_same_dim - x_px_same_dim)**2 + (y_dot_same_dim - y_px_same_dim)**2)

        del(x_dot_same_dim, x_px_same_dim, y_dot_same_dim, y_px_same_dim)

        image = np.zeros_like(dist)
        image = _threshold_image(self.speckle_data.radius, image, dist)
        del(dist)

        # image = gaussian_filter(image, self.speckle_data.gauss_blur)

        image = np.max(image,axis=1)
        image = image.reshape(grid_shape)

        if self.speckle_data.white_bg:
            image = image * -1 + 1

        bits_pp = (2**self.speckle_data.bits) - 1
        image = bits_pp * image
        image = np.floor(image)

        ratio = _colour_count(self.speckle_data.size_x, self.speckle_data.size_y,
                               image)
        print('Final b/w ratio:', ratio)

        return image

    def _optimal_dot_number(self):
        '''
        Function to calculate the number of dots needed in the x and y directions
        to give the required b/w ratio
            Returns:
                num_dots_x (int): The number of dots in the x-dir
                num_dots_y (int): The number of dots in the y-dir
                n_tot (int): The total number of dots
        '''
        number_dots = ((self.speckle_data.proportion_goal * self.speckle_data.size_x
                        * self.speckle_data.size_y)
                       / (np.pi * self.speckle_data.radius**2))

        x_y_ratio = self.speckle_data.size_x / self.speckle_data.size_y
        area = self.speckle_data.size_x * self.speckle_data.size_y
        num_dots_y = np.sqrt(number_dots / x_y_ratio)
        num_dots_x = number_dots / num_dots_y
        num_dots_y = int(np.round(num_dots_y))
        num_dots_x = int(np.round(num_dots_x))
        n_tot = num_dots_x * num_dots_y
        b_w_ratio = round(((np.pi * self.speckle_data.radius**2 * n_tot) /
                     (area)), 3)
        print('Initial b/w ratio', b_w_ratio)

        return (num_dots_x, num_dots_y, n_tot)



    def _dot_locations(self, num_dots_x: int, num_dots_y: int, n_tot: int) -> tuple[np.ndarray, np.ndarray]:
        '''
        Finds the dot coordinates in an even grid and applies a function to move them randomly
            Parameters:
                num_dots_x (int): Number of dots in the x-dir for the required b/w ratio
                num_dots_y (int): Number of dots in the y-dir for the required b/w ratio
                n_tot (int): The total number of dots for the required b/w ratio
            Returns:
                x_dot_2d (np.ndarray): The dot coordinates in the x-dir, as a 2D array
                y_dot_2d (np.ndarray): The dot coordinates in the y-dir, as a 2D array
        '''
        x_first_dot_pos = self.speckle_data.size_x / (num_dots_x * 2)
        y_first_dot_pos = self.speckle_data.size_y / (num_dots_y * 2)
        dot_centre_x = np.linspace(x_first_dot_pos, (self.speckle_data.size_x - x_first_dot_pos),
                                   num=num_dots_x)
        dot_centre_y = np.linspace(y_first_dot_pos, (self.speckle_data.size_y - y_first_dot_pos),
                                   num=num_dots_y)
        dot_x_grid, dot_y_grid = np.meshgrid(dot_centre_x, dot_centre_y)
        del(dot_centre_x, dot_centre_y)
        x_dot_vec = dot_x_grid.flatten()
        y_dot_vec = dot_y_grid.flatten()
        del(dot_x_grid, dot_y_grid)
        x_dot_random = np.add(x_dot_vec, _random_location(self.seed,
                                                          self.speckle_data.radius,
                                                          n_tot))
        y_dot_random = np.add(y_dot_vec, _random_location(self.seed,
                                                         self.speckle_data.radius,
                                                         n_tot))
        del(x_dot_vec, y_dot_vec)
        x_dot_2d = np.atleast_2d(x_dot_random)
        y_dot_2d = np.atleast_2d((y_dot_random))
        del(x_dot_random, y_dot_random)

        return (x_dot_2d, y_dot_2d)

def _threshold_image(radius: int, image: np.ndarray, dist: np.ndarray) -> np.ndarray:
    '''
    Takes an image and adds dots in a contrasting colour
    Different grey levels are added between the background and dots to reduce the
    spatial frequency
        Parameters:
            radius (int): The radius of the dots added
            image (np.ndarray): The image array to which the dots are added
            dist (np.ndarray): An array of distances from each pixel centre to
                each dot centre
        Returns:
            image (np.ndarray): The image array with dots added
    '''
    grey_threshold = radius + 0.5
    image[dist <= grey_threshold] = 0.2
    grey_threshold -= 0.1
    image[dist <= grey_threshold] = 0.5
    grey_threshold -= 0.1
    image[dist <= grey_threshold] = 0.8
    image[dist <= radius] = 1

    return image

def _random_location(seed: int | None, radius: int, n_tot: int) -> np.ndarray:
        '''
        Produces a vector the same size as the x and y coordinate vectors containing random values
            Parameters:
                n_tot (int): The numer of dots, so the size the vector needs to be
                seed (int): A value to initialise the random number generator
            Returns:
                random_array (np.ndarray): An array of random numbers the same
                    size as the dot location vector
        '''
        rng = np.random.default_rng(seed)
        sigma = radius / 2.2
        random_array = rng.normal(loc=0.0,scale=sigma,size=n_tot)
        return random_array

def _colour_count(size_x: int, size_y: int, image: np.ndarray) -> float:
    '''
    Return proportion of black pixels (with value 0) in image array as a percentage
        Parameters:
            image (array): A 2D image array of specified size, containing
                only pixels of value 0 and 225
            proportion (float): A float containing the percentage proportion
                of black (0) in the `image` array
        Returns:
            proportion (float): An updated percentage proportion of black
                in the image array
    '''
    count = 0
    proportion = 0
    colours, counts = np.unique(image, return_counts=1)
    for index, colour in enumerate(colours):
        count = counts[index]
        all_proportion = (100 * count) / (size_x * size_y)
        if colour == 0.0:
            proportion = round((all_proportion / 100), 3)
    del(colours, counts)

    return proportion

def _px_locations(size_x: int, size_y: int) -> tuple[tuple, np.ndarray, np.ndarray]:
    '''
    Returns arrays of the pixel locations in both the x and y directions
        Returns:
            x_px_grid (np.ndarray): The pixel locations in the x-dir as a 2D array the same shape as the image
            x_px_trans (np.ndarray): The pixel locations in the x-dir as a 2D array with values only in 1D
            y_px_trans (np.ndarray): The pixel locations in the y-dir as a 2D array with values only in 1D
    '''
    px_centre_x = np.linspace(0.5, (size_x - 0.5),
                                num=size_x)
    px_centre_y = np.linspace(0.5, (size_y - 0.5),
                                num=size_y)
    x_px_grid,y_px_grid  = np.meshgrid(px_centre_x, px_centre_y)
    del(px_centre_x, px_centre_y)
    grid_shape = x_px_grid.shape
    x_px_2d = np.atleast_2d(x_px_grid.flatten())
    y_px_2d = np.atleast_2d(y_px_grid.flatten())
    del(x_px_grid, y_px_grid)
    x_px_trans = x_px_2d.T
    y_px_trans = y_px_2d.T
    del(x_px_2d, y_px_2d)

    return (grid_shape, x_px_trans, y_px_trans)

def show_image(image: np.ndarray) -> None:
    '''
    Defines figure size and formatting (no axes) and plots the image array in greyscale
        Parameters:
            image (arrray): A 2D array to be plotted
    '''
    px = 1/plt.rcParams['figure.dpi']
    plt.figure(figsize=((SpeckleData.size_x * px), (SpeckleData.size_y  * px)))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='grey', vmin=0, vmax=1)
    plt.show()
    plt.close()

def save_image(image: np.ndarray, directory: Path, filename: str, bits: int = SpeckleData.bits) -> None:
    '''
    Saves image to specified filename and location
        Parameters:
            image (arrray): A 2D array to be plotted
    '''
    filename_full = filename + '.' + SpeckleData.file_format.value
    filepath = Path.joinpath(directory, filename_full)
    px = 1/plt.rcParams['figure.dpi']
    bits_pp = 2**bits - 1

    if 2 <= bits < 8:
        image_scaled_down = image / bits_pp
        relative_scale = 8 - bits
        image = image_scaled_down.astype(np.uint8)
        scale_up_factor = 2**(relative_scale + bits) - 1
        image = np.multiply(image, scale_up_factor)
    elif bits == 8:
        image = image.astype(np.uint8)
    elif 8 < bits < 16:
        image_scaled_down = image / bits_pp
        relative_scale = 16 - bits
        image = image_scaled_down.astype(np.uint16)
        scale_up_factor = 2**(relative_scale + bits) - 1
        image = np.multiply(image, scale_up_factor)
    elif bits == 16:
        image = image.astype(np.uint16)
    else:
        print('Error: Bit depth added not acceptable')

    tiff_image = Image.fromarray(image)
    res = SpeckleData.image_res
    tiff_image.save(filepath, SpeckleData.file_format.value, dpi=(res, res))

def mean_intensity_gradient(image: np.ndarray) -> tuple[float,float,float]:
    '''
    Calculates the mean intensity gradient and returns the overall MIG
    as well as in the x and y directions
        Parameters:
            image (np.ndarray): The final image array
        Returns:
            mig (float): The overall mean intensity gradient for the image
            mig_x (float): The mean intensity gradient in the x-direction
            mig_y (float): The mean intensity gradient in the y-direction
    '''
    intensity_gradient = np.gradient(image)

    mig_x = np.mean(intensity_gradient[0].flatten())
    mig_y = np.mean(intensity_gradient[1].flatten())
    mig = np.mean(np.array(mig_x,mig_y))

    return (mig,mig_x,mig_y)