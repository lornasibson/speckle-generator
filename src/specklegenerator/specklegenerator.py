from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class FileFormat(Enum):
    TIFF = 'tiff'
    BITMAP = 'raw'

@dataclass
class SpeckleData:
    '''
    Data class to store default parameters
    '''
    size_x: int = 500
    size_y:int = 500
    radius:int = 10
    proportion_goal:float = 0.5
    white_bg:bool = True
    image_res:int = 200
    file_format:FileFormat = FileFormat.TIFF.value
    gauss_blur: float = 1


class Speckle:
    '''
    A class to generate, show and save a speckle pattern subject to input parameters
        Parameters:
            size_x (int): An integer value of the horizontal size of speckle image in pixels
            size_y (int): An integer value of the vertical size of speckle image in pixels
            radius (int): An integer value of the radius of circles used in the speckle pattern
            proportion_goal (int): An integer value of the desired black/white balance as a percentage
            file_format (str): A string of the desired file format of the saved file
            white_bg (bool): A Boolean value for whether a white or black background is wanted
            image_res (int): An integer value of the desired image resolution in dpi
    '''
    def __init__(self,
                 speckle_data: SpeckleData) -> None:
        self.speckle_data = speckle_data

    def make_speckle(self) -> np.ndarray:
        '''
        Produces a random speckle pattern with given parameters
        '''
        area, num_dots_x, num_dots_y, n_tot, b_w_ratio = Speckle._optimal_dot_number(self)
        print('Initial b/w ratio', b_w_ratio)

        x_dot_2d, y_dot_2d = Speckle._dot_locations(self, num_dots_x, num_dots_y, n_tot)
        x_px_grid, x_px_trans, y_px_trans = Speckle._px_locations(self)

        x_px_same_dim = np.repeat(x_px_trans, n_tot, axis=1)
        x_dot_same_dim = np.repeat(x_dot_2d, area, axis=0)
        y_px_same_dim = np.repeat(y_px_trans, n_tot, axis=1)
        y_dot_same_dim = np.repeat(y_dot_2d, area, axis=0)

        dist = np.sqrt((x_dot_same_dim - x_px_same_dim)**2 + (y_dot_same_dim - y_px_same_dim)**2)

        image = np.zeros_like(dist)
        image = Speckle._threshold_image(self, image, dist)

        # image = gaussian_filter(image, self.speckle_data.gauss_blur)

        image = np.max(image,axis=1)
        image = image.reshape(x_px_grid.shape)

        if self.speckle_data.white_bg:
            image = image * -1 + 1

        ratio = Speckle._colour_count(self, image)
        print('Final b/w ratio:', ratio)

        return image

    def _optimal_dot_number(self):
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

        return area, num_dots_x, num_dots_y, n_tot, b_w_ratio

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
        x_dot_vec = dot_x_grid.flatten()
        y_dot_vec = dot_y_grid.flatten()
        x_dot_random = np.add(x_dot_vec,
                              Speckle._random_location(self, n_tot))
        y_dot_random = np.add(y_dot_vec,
                              Speckle._random_location(self, n_tot))
        x_dot_2d = np.atleast_2d(x_dot_random)
        y_dot_2d = np.atleast_2d((y_dot_random))

        return (x_dot_2d, y_dot_2d)

    def _px_locations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Returns arrays of the pixel locations in both the x and y directions
            Returns:
                x_px_grid (np.ndarray): The pixel locations in the x-dir as a 2D array the same shape as the image
                x_px_trans (np.ndarray): The pixel locations in the x-dir as a 2D array with values only in 1D
                y_px_trans (np.ndarray): The pixel locations in the y-dir as a 2D array with values only in 1D
        '''
        px_centre_x = np.linspace(0.5, (self.speckle_data.size_x - 0.5),
                                   num=self.speckle_data.size_x)
        px_centre_y = np.linspace(0.5, (self.speckle_data.size_y - 0.5),
                                   num=self.speckle_data.size_y)
        x_px_grid,y_px_grid  = np.meshgrid(px_centre_x, px_centre_y)
        x_px_2d = np.atleast_2d(x_px_grid.flatten())
        y_px_2d = np.atleast_2d(y_px_grid.flatten())
        x_px_trans = x_px_2d.T
        y_px_trans = y_px_2d.T

        return (x_px_grid, x_px_trans, y_px_trans)


    def _random_location(self, n_tot: int, seed: int | None = None) -> np.ndarray:
        '''
        Produces a vector the same size as the x and y coordinate vectors containing random values
            Parameters:
                n_tot (int): The numer of dots, so the size the vector needs to be
                seed (int): A value to initialise the random number generator
            Returns:
                random_array (np.ndarray): An array of random numbers the same size as the dot location vector
        '''
        rng = np.random.default_rng(seed)
        sigma = self.speckle_data.radius / 2.2
        random_array = rng.normal(loc=0.0,scale=sigma,size=n_tot)
        return random_array

    def _threshold_image(self, image: np.ndarray, dist: np.ndarray) -> np.ndarray:
        grey_threshold = self.speckle_data.radius + 0.5
        image[dist < grey_threshold] = 0.2
        grey_threshold -= 0.1
        image[dist < grey_threshold] = 0.5
        grey_threshold -= 0.1
        image[dist < grey_threshold] = 0.8
        image[dist < self.speckle_data.radius] = 1

        return image

    def _colour_count(self, image: np.ndarray) -> float:
        '''
        Return proportion of black pixels (with value 0) in image array as a percentage
            Parameters:
                image (array): A 2D image array of specified size, containing only pixels of value 0 and 225
                proportion (float): A float containing the percentage proportion of black (0) in the `image` array
            Returns:
                proportion (float): An updated percentage proportion of black in the image array
        '''
        count = 0
        proportion = 0
        colours, counts = np.unique(image, return_counts=1)
        for index, colour in enumerate(colours):
            count = counts[index]
            all_proportion = (100 * count) / (self.speckle_data.size_x * self.speckle_data.size_y)
            if colour == 0.0:
                proportion = round((all_proportion / 100), 3)
        return proportion

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

def save_image(image: np.ndarray, directory: Path, filename: str) -> None:
    '''
    Saves image to specified filename and location
        Parameters:
            image (arrray): A 2D array to be plotted
    '''
    filename_full = filename + '.' + SpeckleData.file_format
    px = 1/plt.rcParams['figure.dpi']
    plt.figure(figsize=((SpeckleData.size_x * px), (SpeckleData.size_y  * px)))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='grey', vmin=0, vmax=1)
    plt.savefig(Path.joinpath(directory, filename_full),
                format=SpeckleData.file_format, bbox_inches='tight',
                pad_inches=0, dpi=SpeckleData.image_res)
    plt.close()

def mean_intensity_gradient(image: np.ndarray) -> tuple[float,float,float]:
    '''
    Calculates the mean intensity gradient and returns the overall MIG as well as in the x and y directions
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

