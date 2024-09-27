import os
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
    size_x: int = 500
    size_y:int = 500
    radius:int = 20
    proportion_goal:float = 0.5
    white_bg:bool = True
    image_res:int = 300
    file_format:FileFormat = FileFormat.TIFF.value
    gauss_blur: float = 0.9


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

    def make_speckle(self) ->np.ndarray:
        number_dots = ((self.speckle_data.proportion_goal * self.speckle_data.size_x * self.speckle_data.size_y)
                       / (np.pi * self.speckle_data.radius**2))
        x_y_ratio = self.speckle_data.size_x / self.speckle_data.size_y
        area = self.speckle_data.size_x * self.speckle_data.size_y
        num_dots_y = np.sqrt(number_dots / x_y_ratio)
        num_dots_x = number_dots / num_dots_y
        num_dots_y = int(np.round(num_dots_y))
        num_dots_x = int(np.round(num_dots_x))
        n_tot = num_dots_x * num_dots_y
        b_w_ratio = ((np.pi * self.speckle_data.radius**2 * n_tot) /
                     (area))
        # print(b_w_ratio)

        dot_centre_x = np.linspace(0, self.speckle_data.size_x, num=(num_dots_x + 2))
        dot_centre_x = dot_centre_x[1:-1]
        dot_centre_y = np.linspace(0, self.speckle_data.size_y, num=(num_dots_y + 2))
        dot_centre_y = dot_centre_y[1:-1]
        dot_x, dot_y = np.meshgrid(dot_centre_x, dot_centre_y)
        x_dot = dot_x.flatten()
        y_dot = dot_y.flatten()

        px_centre_x = np.linspace(0.5, (self.speckle_data.size_x - 0.5),
                                   num=self.speckle_data.size_x)
        px_centre_y = np.linspace(0.5, (self.speckle_data.size_y - 0.5),
                                   num=self.speckle_data.size_y)
        px_x, px_y = np.meshgrid(px_centre_x, px_centre_y)
        x_px = px_x.flatten()
        y_px = px_y.flatten()



        image = np.zeros((self.speckle_data.size_y, self.speckle_data.size_x))

        x_dot_2d = np.atleast_2d(x_dot)
        x_px_2d = np.atleast_2d(x_px)
        x_px_trans = np.transpose(x_px_2d)
        x_px_same_dim = np.repeat(x_px_trans, n_tot, axis=1)
        x_dot_same_dim = np.repeat(x_dot_2d, area, axis=0)

        y_dot_2d = np.atleast_2d(y_dot)
        y_px_2d = np.atleast_2d(y_px)
        y_px_trans = np.transpose(y_px_2d)
        y_px_same_dim = np.repeat(y_px_trans, n_tot, axis=1)
        y_dot_same_dim = np.repeat(y_dot_2d, area, axis=0)

        d = np.sqrt((x_dot_same_dim - x_px_same_dim)**2 + (y_dot_same_dim - y_px_same_dim)**2)


        # image[d < self.speckle_dsata.radius] = 1

        d_split = np.split(d, n_tot, axis=1)

        for i in range(n_tot):
            dot = d_split[i].reshape((self.speckle_data.size_y, self.speckle_data.size_x))
            image[dot < self.speckle_data.radius] = 1

        return image













    def generate_speckle(self) -> np.ndarray:
        '''
        Creates, displays and saves a dotted speckle pattern of specified size and black/white balance
        '''
        num_dots = 50
        proportion = 100

        image = np.zeros((self.speckle_data.size_y, self.speckle_data.size_x))

        circle = np.zeros((self.speckle_data.radius * 2, self.speckle_data.radius * 2))
        xs, ys = np.meshgrid(np.arange(2 * self.speckle_data.radius), np.arange(2* self.speckle_data.radius))
        r = ((xs - self.speckle_data.radius) ** 2 + (ys - self.speckle_data.radius) ** 2)** 0.5
        circle[r < self.speckle_data.radius] = 255


        while proportion > self.speckle_data.proportion_goal:
            for i in range(num_dots):
                image = Speckle._add_circle(self, image, circle)
            proportion = Speckle._colour_count(self, image, proportion)
            if proportion > self.speckle_data.proportion_goal:
                image = np.zeros((self.speckle_data.size_y, self.speckle_data.size_x))
            else:
                break
            num_dots += 30
            print([proportion])
        print(proportion)
        filtered = gaussian_filter(image, self.speckle_data.gauss_blur)
        if self.speckle_data.white_bg:
            filtered = filtered * -1 + 1

        return filtered

    def _add_circle(self, image: np.ndarray, circle: np.ndarray) -> np.ndarray:
        '''
        Defines a random point and (unless it is out-of-bounds) adds a circle to the image array at that position
            Parameters:
                image (array): A 2D image array of specified size
                circle (array): A 2D array the size of the radius, containing a central circle of values of 255
            Returns:
                image (array): The image array with the circle added
        '''
        pos_x = np.random.randint(self.speckle_data.radius, (self.speckle_data.size_x - self.speckle_data.radius))
        pos_y = np.random.randint(self.speckle_data.radius, (self.speckle_data.size_y - self.speckle_data.radius))

        x_start, x_end = pos_x - self.speckle_data.radius, pos_x + self.speckle_data.radius
        y_start, y_end = pos_y - self.speckle_data.radius, pos_y + self.speckle_data.radius

        if x_start >= 0 and x_end <= self.speckle_data.size_x and y_start >= 0 and y_end <= self.speckle_data.size_y:
            image[y_start:y_end, x_start:x_end] = 0
            image[y_start:y_end, x_start:x_end] += circle
        else:
            print(f"Skipping out-of-bounds circle at position ({pos_x}, {pos_y})")
        return image

    def _colour_count(self, image: np.ndarray, proportion: float) -> float:
        '''
        Return proportion of black pixels (with value 0) in image array as a percentage
            Parameters:
                image (array): A 2D image array of specified size, containing only pixels of value 0 and 225
                proportion (float): A float containing the percentage proportion of black (0) in the `image` array
            Returns:
                proportion (float): An updated percentage proportion of black in the image array
        '''
        count = 0
        colours, counts = np.unique(image, return_counts=1)
        for index, colour in enumerate(colours):
            count = counts[index]
            all_proportion = (100 * count) / (self.speckle_data.size_x * self.speckle_data.size_y)
            if colour == 0.0:
                proportion = all_proportion
        return proportion

def show_image(image: np.ndarray):
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
    plt.savefig(Path.joinpath(directory, filename_full), format=SpeckleData.file_format, bbox_inches='tight', pad_inches=0, dpi=SpeckleData.image_res)
    plt.close()
