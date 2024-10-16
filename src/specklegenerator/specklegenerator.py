from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class SpeckleError(Exception):
    pass

class FileFormat(Enum):
    """Enum class to restrict the allowable output file formats

    """

    TIFF = "tiff"
    BITMAP = "bmp"

@dataclass
class SpeckleData:
    """Data class to store default parameters

    Args:
            size_x (int): An integer value of the horizontal size of speckle image in pixels
            size_y (int): An integer value of the vertical size of speckle image in pixels
            radius (int): An integer value of the radius of circles used in the speckle pattern
            b_w_ratio (int): An integer value of the desired black/white balance as a percentage
            white_bg (bool): A Boolean value for whether a white or black background is wanted
            image_res (int): An integer value of the desired image resolution in dpi
            file_format (enum): A string of the desired file format of the saved file
            bits (int): An integer value of the bit size of the image
    """

    size_x: int = 500
    size_y: int = 500
    radius: int = 10
    b_w_ratio: float = 0.5
    white_bg: bool = True
    image_res: int = 200
    file_format: FileFormat = FileFormat.TIFF
    gauss_blur: float | None = None
    bits: int = 16


def validate_speckle_data(speckle_data: SpeckleData) -> None:
    """Method to check the input parameters of the class to ensure that they are reasonable

    Args:
        speckle_data (SpeckleData): Data class containing the input parameters

    Raises:
        SpeckleError: The image size cannot be 0, please enter a suitable integer
        SpeckleError: The image size cannot be negative
        SpeckleError: The image size is too small compared to the speckle radius
        SpeckleError: The radius cannot be 0, please enter a suitable integer
        SpeckleError: The radius cannot be negative
        SpeckleError: The radius is too large compared to the image size
        SpeckleError: The black/white ratio must be between 0 and 1
        SpeckleError: The black/white ratio cannot be 0
        SpeckleError: The black/white cannot be 1
        SpeckleError: The image resolution cannot be negative
        SpeckleError: The image resolution cannot be 0
        SpeckleError: The bit size cannot be smaller than 2
        SpeckleError: The bit size cannot be larger than 16
    """

    # Sizes
    if speckle_data.size_x == 0 or speckle_data.size_y == 0:
        raise SpeckleError("The image size cannot be 0, please enter a suitable integer")
    elif speckle_data.size_x < 0 or speckle_data.size_y < 0:
        raise SpeckleError("The image size cannot be negative")
    elif speckle_data.size_x <= 10 or speckle_data.size_y <= 10:
        raise SpeckleError("The image size is too small compared to the speckle radius")

    # Radius
    if speckle_data.radius == 0:
        raise SpeckleError("The radius cannot be 0, please enter a suitable integer")
    elif speckle_data.radius < 0:
        raise SpeckleError("The radius cannot be negative")
    elif (
        speckle_data.radius > (speckle_data.size_x / 2)
        or speckle_data.radius > speckle_data.size_y / 2
    ):
        raise SpeckleError("The radius is too large compared to the image size")

    # Proportion
    if (
        speckle_data.b_w_ratio < 0.0
        or speckle_data.b_w_ratio > 1.0
    ):
        raise SpeckleError("The black/white ratio must be between 0 and 1")
    elif speckle_data.b_w_ratio == 0.0:
        raise SpeckleError("The black/white ratio cannot be 0")
    elif speckle_data.b_w_ratio == 1.0:
        raise SpeckleError("The black/white cannot be 1")

    # Image resolution
    if speckle_data.image_res < 0.0:
        raise SpeckleError("The image resolution cannot be negative")
    elif speckle_data.image_res == 0.0:
        raise SpeckleError("The image resolution cannot be 0")

     # Bits
    if speckle_data.bits < 2:
        raise SpeckleError("The bit size cannot be smaller than 2")
    elif speckle_data.bits > 16:
        raise SpeckleError("The bit size cannot be larger than 16")


class Speckle:
    """A class to generate, show and save a speckle pattern subject to input parameters
    """

    def __init__(self, speckle_data: SpeckleData, seed: int | None = None) -> None:
        self.speckle_data = speckle_data
        self.seed = seed
        validate_speckle_data(speckle_data)


    def make(self) -> np.ndarray:
        """Produces a random speckle pattern with given parameters

        Returns:
            np.ndarray: An image array containing a speckle pattern
        """

        num_dots_x, num_dots_y, n_tot = self._optimal_dot_number()
        image = np.zeros((self.speckle_data.size_x, self.speckle_data.size_y))

        elements = self.speckle_data.size_x * self.speckle_data.size_y * n_tot
        if elements < 220000000:
            x_dot_2d, y_dot_2d = self._dot_locations_2d(num_dots_x, num_dots_y, n_tot)
            grid_shape, x_px_trans, y_px_trans = _px_locations(self.speckle_data.size_x,
                                                   self.speckle_data.size_y)
            image = self._make_in_mem(x_dot_2d,
                                      y_dot_2d,
                                      x_px_trans,
                                      y_px_trans,
                                      n_tot,
                                      grid_shape)
        else:
            x_dots, y_dots = self._dot_locations(num_dots_x, num_dots_y, n_tot)
            image = self._make_loop(image, n_tot, x_dots, y_dots)


        if self.speckle_data.gauss_blur is not None:
            image = gaussian_filter(image, self.speckle_data.gauss_blur)



        if self.speckle_data.white_bg is True:
            image = image * (-1) + 1

        bits_pp = (2**self.speckle_data.bits) - 1
        image = bits_pp * image
        image = np.floor(image)

        ratio = _colour_count(self.speckle_data.size_x, self.speckle_data.size_y, image)
        print("Final b/w ratio:", ratio)

        return image

    def _make_loop(self,
                   image: np.ndarray,
                   n_dots:int,
                   x_dots: np.ndarray,
                   y_dots:np.ndarray) -> np.ndarray:
        """Creates an image array with speckles using an iterative method to
        reduce the load on the pc's memory

        Args:
            image (np.ndarray): An image array of zeros
            n_dots (int): The number of dots in the image
            x_dots (np.ndarray): An array containing the location of the dots in the x-dir
            y_dots (np.ndarray): An array containing the location of the dots in the y-dir

        Returns:
            np.ndarray: An image array containing speckles
        """
        x_px = np.linspace(0.5, (self.speckle_data.size_x - 0.5), num=self.speckle_data.size_x)
        y_px = np.linspace(0.5, (self.speckle_data.size_y - 0.5), num=self.speckle_data.size_y)

        x_px_grid, y_px_grid = np.meshgrid(x_px, y_px)
        del (x_px, y_px)

        for i in range(n_dots):
            x_dot = x_dots[i]
            y_dot = y_dots[i]
            dist = np.sqrt(
            (x_dot - x_px_grid) ** 2 + (y_dot - y_px_grid) ** 2)
            # image[dist <= self.speckle_data.radius] = 1
            image = _threshold_image(self.speckle_data.radius, image, dist)
            del(dist)

        return image


    def _make_in_mem(self,
                     x_dot_2d:np.ndarray,
                     y_dot_2d: np.ndarray,
                     x_px_trans:np.ndarray,
                     y_px_trans:np.ndarray,
                     n_tot: int,
                     grid_shape) -> np.ndarray:
        """Creates a speckle pattern using large arrays that utilise a pc's memory
        capacity instead of computational power

        Args:
            x_dot_2d (np.ndarray): The location of the dots in the x-dir as a 2D array
            y_dot_2d (np.ndarray): The location of the dots in the y-dir as a 2D array
            x_px_trans (np.ndarray): The location of the pixel centres in the x-dir
            y_px_trans (np.ndarray): The location of the pixel centres in the y-dir
            n_tot (int): The total number of dots
            grid_shape (_type_): The shape of the array of the pixel centres

        Returns:
            np.ndarray: An image array containing a speckle pattern
        """
        x_px_same_dim = np.repeat(x_px_trans, n_tot, axis=1)
        x_dot_same_dim = np.repeat(
            x_dot_2d, (self.speckle_data.size_x * self.speckle_data.size_y), axis=0
        )
        y_px_same_dim = np.repeat(y_px_trans, n_tot, axis=1)
        y_dot_same_dim = np.repeat(
            y_dot_2d, (self.speckle_data.size_x * self.speckle_data.size_y), axis=0
        )
        del (x_px_trans, x_dot_2d, y_px_trans, y_dot_2d)

        dist = np.sqrt(
            (x_dot_same_dim - x_px_same_dim) ** 2
            + (y_dot_same_dim - y_px_same_dim) ** 2
        )

        del (x_dot_same_dim, x_px_same_dim, y_dot_same_dim, y_px_same_dim)

        image = np.zeros_like(dist)
        image = _threshold_image(self.speckle_data.radius, image, dist)
        del dist

        image = np.max(image, axis=1)
        image = image.reshape(grid_shape)

        return image

    def _optimal_dot_number(self) -> tuple[int, int, int]:
        """Function to calculate the number of dots needed in the x and y directions
        to give the required b/w ratio

        Returns:
            tuple[int, int, int]:
                num_dots_x (int): The number of dots in the x-dir
                num_dots_y (int): The number of dots in the y-dir
                n_tot (int): The total number of dots
        """

        number_dots = (
            self.speckle_data.b_w_ratio
            * self.speckle_data.size_x
            * self.speckle_data.size_y
        ) / (np.pi * self.speckle_data.radius**2)

        x_y_ratio = self.speckle_data.size_x / self.speckle_data.size_y
        area = self.speckle_data.size_x * self.speckle_data.size_y
        num_dots_y = np.sqrt(number_dots / x_y_ratio)
        num_dots_x = number_dots / num_dots_y
        num_dots_y = int(np.round(num_dots_y))
        num_dots_x = int(np.round(num_dots_x))
        n_tot = num_dots_x * num_dots_y
        b_w_ratio = round(((np.pi * self.speckle_data.radius**2 * n_tot) / (area)), 3)
        print("Initial b/w ratio", b_w_ratio)

        return (num_dots_x, num_dots_y, n_tot)

    def _dot_locations(self,
                       num_dots_x: int,
                       num_dots_y: int,
                       n_tot:int) -> tuple[np.ndarray, np.ndarray]:
        """The locations of the randomised dot centres as 1D arrays

        Args:
            num_dots_x (int): Number of dots in the x-dir for the required b/w ratio
            num_dots_y (int): Number of dots in the y-dir for the required b/w ratio
            n_tot (int): The total number of dots for the required b/w ratio

        Returns:
            tuple[np.ndarray, np.ndarray]: The dot coordinates in the x-dir and
            y-dir
        """

        x_first_dot_pos = self.speckle_data.size_x / (num_dots_x * 2)
        y_first_dot_pos = self.speckle_data.size_y / (num_dots_y * 2)
        dot_centre_x = np.linspace(
            x_first_dot_pos,
            (self.speckle_data.size_x - x_first_dot_pos),
            num=num_dots_x,
        )
        dot_centre_y = np.linspace(
            y_first_dot_pos,
            (self.speckle_data.size_y - y_first_dot_pos),
            num=num_dots_y,
        )

        dot_x_grid, dot_y_grid = np.meshgrid(dot_centre_x, dot_centre_y)
        del (dot_centre_x, dot_centre_y)
        x_dot_vec = dot_x_grid.flatten()
        y_dot_vec = dot_y_grid.flatten()
        del (dot_x_grid, dot_y_grid)
        x_dot_random = np.add(
            x_dot_vec, _random_location(self.seed, self.speckle_data.radius, n_tot)
        )
        y_dot_random = np.add(
            y_dot_vec, _random_location(self.seed, self.speckle_data.radius, n_tot)
        )
        del (x_dot_vec, y_dot_vec)

        return (x_dot_random, y_dot_random)

    def _dot_locations_2d(self,
                          num_dots_x:int,
                          num_dots_y:int,
                          n_tot:int) -> tuple[np.ndarray, np.ndarray]:
        """Method to output the centre locations of the randomised dots as 2D arrays

        Args:
            num_dots_x (int): Number of dots in the x-dir for the required b/w ratio
            num_dots_y (int): Number of dots in the y-dir for the required b/w ratio
            n_tot (int): The total number of dots for the required b/w ratio

        Returns:
            tuple[np.ndarray, np.ndarray]: 2D arrays of the randomised dot centres
        """
        x_first_dot_pos = self.speckle_data.size_x / (num_dots_x * 2)
        y_first_dot_pos = self.speckle_data.size_y / (num_dots_y * 2)
        dot_centre_x = np.linspace(
            x_first_dot_pos,
            (self.speckle_data.size_x - x_first_dot_pos),
            num=num_dots_x,
        )
        dot_centre_y = np.linspace(
            y_first_dot_pos,
            (self.speckle_data.size_y - y_first_dot_pos),
            num=num_dots_y,
        )
        dot_x_grid, dot_y_grid = np.meshgrid(dot_centre_x, dot_centre_y)
        del (dot_centre_x, dot_centre_y)
        x_dot_vec = dot_x_grid.flatten()
        y_dot_vec = dot_y_grid.flatten()
        del (dot_x_grid, dot_y_grid)
        x_dot_random = np.add(
            x_dot_vec, _random_location(self.seed, self.speckle_data.radius, n_tot)
        )
        y_dot_random = np.add(
            y_dot_vec, _random_location(self.seed, self.speckle_data.radius, n_tot)
        )
        del (x_dot_vec, y_dot_vec)
        x_dot_2d = np.atleast_2d(x_dot_random)
        y_dot_2d = np.atleast_2d((y_dot_random))
        del (x_dot_random, y_dot_random)

        return (x_dot_2d, y_dot_2d)


def _threshold_image(radius: int, image: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Takes an image and adds dots in a contrasting colour
    Different grey levels are added between the background and dots to reduce the
    spatial frequency

    Args:
        radius (int): The radius of the dots added
        image (np.ndarray): The image array to which the dots are added
        dist (np.ndarray): An array of distances from each pixel centre to
                each dot centre

    Returns:
        np.ndarray: The image array with dots added
    """

    grey_threshold = radius + 1
    mask = (dist <= grey_threshold) & (image != 1)
    image[mask] = 0.2

    grey_threshold -= 0.3
    mask = (dist <= grey_threshold) & (image != 1)
    image[mask] = 0.3

    grey_threshold -= 0.3
    mask = (dist <= grey_threshold) & (image != 1)
    image[mask] = 0.5

    grey_threshold -= 0.2
    mask = (dist <= grey_threshold) & (image != 1)
    image[mask] = 0.8

    image[dist <= radius] = 1

    return image


def _random_location(seed: int | None, radius: int, n_tot: int) -> np.ndarray:
    """Produces a vector the same size as the x and y coordinate vectors containing random values

    Args:
        seed (int | None): A value to initialise the random number generator
        radius (int): The radius of the dots, in pixels
        n_tot (int): The numer of dots, so the size the vector needs to be

    Returns:
        np.ndarray: An array of random numbers the same
                size as the dot location vector
    """

    rng = np.random.default_rng(seed)
    sigma = radius / 2.2
    random_array = rng.normal(loc=0.0, scale=sigma, size=n_tot)
    return random_array


def _colour_count(size_x: int, size_y: int, image: np.ndarray) -> float:
    """Return proportion of black pixels (with value 0) in image array as a percentage

    Args:
        size_x (int): The size of the image array in the x-dir
        size_y (int): The size of the image array in the y-dir
        image (np.ndarray): A 2D image array of specified size, containing
                only pixels of value 0 and 225

    Returns:
        float: The proportion of black in the image array
    """

    count = 0
    proportion = 0
    colours, counts = np.unique(image, return_counts=1)
    for index, colour in enumerate(colours):
        count = counts[index]
        all_proportion = (100 * count) / (size_x * size_y)
        if colour == 0.0:
            proportion = round((all_proportion / 100), 3)
    del (colours, counts)

    return proportion


def _px_locations(size_x: int, size_y: int) -> tuple[tuple, np.ndarray, np.ndarray]:
    """Returns arrays of the pixel locations in both the x and y directions

    Args:
        size_x (int): The size of the image in the x-dir
        size_y (int): The size of the image in the y-dir

    Returns:
        tuple[tuple, np.ndarray, np.ndarray]:
            x_px_grid (np.ndarray): The pixel locations in the x-dir as a 2D array the same shape as the image
            x_px_trans (np.ndarray): The pixel locations in the x-dir as a 2D array with values only in 1D
            y_px_trans (np.ndarray): The pixel locations in the y-dir as a 2D array with values only in 1D
    """

    px_centre_x = np.linspace(0.5, (size_x - 0.5), num=size_x)
    px_centre_y = np.linspace(0.5, (size_y - 0.5), num=size_y)
    x_px_grid, y_px_grid = np.meshgrid(px_centre_x, px_centre_y)
    del (px_centre_x, px_centre_y)

    grid_shape = x_px_grid.shape
    x_px_2d = np.atleast_2d(x_px_grid.flatten())
    y_px_2d = np.atleast_2d(y_px_grid.flatten())
    del (x_px_grid, y_px_grid)

    x_px_trans = x_px_2d.T
    y_px_trans = y_px_2d.T
    del (x_px_2d, y_px_2d)

    return (grid_shape, x_px_trans, y_px_trans)


def show_image(image: np.ndarray) -> None:
    """Defines figure size and formatting (no axes) and plots the image array in greyscale

    Args:
        image (np.ndarray): A 2D array to be plotted
    """

    px = 1 / plt.rcParams["figure.dpi"]
    plt.figure(figsize=((SpeckleData.size_x * px), (SpeckleData.size_y * px)))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap="grey")
    plt.show()
    plt.close()


def save_image(
    image: np.ndarray, directory: Path, filename: str, bits: int = SpeckleData.bits) -> None:
    """Saves image to specified filename and location

    Args:
        image (np.ndarray): A 2D image array to be saved to an image format
        directory (Path): The file directory to which the image should be saved
        filename (str): The filename the image should be saved to
        bits (int, optional): The bit size of the image. Defaults to SpeckleData.bits.
    """

    filename_full = filename + "." + SpeckleData.file_format.value
    filepath = Path.joinpath(directory, filename_full)
    bits_pp = 2**bits - 1

    if 2 <= bits < 8:
        image_scaled_down = image / bits_pp
        relative_scale = 8 - bits
        image = image_scaled_down.astype(np.uint8)
        scale_up_factor = 2 ** (relative_scale + bits) - 1
        image = np.multiply(image, scale_up_factor)
    elif bits == 8:
        image = image.astype(np.uint8)
    elif 8 < bits < 16:
        image_scaled_down = image / bits_pp
        relative_scale = 16 - bits
        image = image_scaled_down.astype(np.uint16)
        scale_up_factor = 2 ** (relative_scale + bits) - 1
        image = np.multiply(image, scale_up_factor)
    elif bits == 16:
        image = image.astype(np.uint16)

    tiff_image = Image.fromarray(image)
    res = SpeckleData.image_res
    tiff_image.save(filepath, SpeckleData.file_format.value, dpi=(res, res))


def mean_intensity_gradient(image: np.ndarray) -> tuple[float, float, float]:
    """Calculates the mean intensity gradient and returns the overall MIG
    as well as in the x and y directions

    Args:
        image (np.ndarray): he final image array

    Returns:
        tuple[float, float, float]:
            mig (float): The overall mean intensity gradient for the image
            mig_x (float): The mean intensity gradient in the x-direction
            mig_y (float): The mean intensity gradient in the y-direction
    """

    intensity_gradient = np.gradient(image)

    mig_x = np.round(np.abs(np.mean(intensity_gradient[0].flatten())), decimals=3)
    mig_y = np.round(np.abs(np.mean(intensity_gradient[1].flatten())), decimals=3)
    mig = np.round(np.mean(np.array([mig_x, mig_y])), decimals=3)

    return (mig, mig_x, mig_y)
