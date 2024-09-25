import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

@dataclass
class SpeckleData:
    size_x: int = 1000
    size_y:int = 1000
    radius:int = 5
    proportion_goal:float = 50.0
    white_bg:bool = True
    image_res:int = 300


class Speckle:
    '''
    A class to generate, show and save a speckle pattern subject to input parameters
        Parameters:
            size_x (int): An integer value of the horizontal size of speckle image in pixels
            size_y (int): An integer value of the vertical size of speckle image in pixels
            radius (int): An integer value of the radius of circles used in the speckle pattern
            proportion_goal (int): An integer value of the desired black/white balance as a percentage
            filename (str): A string of the filename the image should be saved as
            file_format (str): A string of the desired file format of the saved file
            directory (str): A string of the directory where the image file should be saved
            white_bg (bool): A Boolean value for whether a white or black background is wanted
            image_res (int): An integer value of the desired image resolution in dpi
    '''
    def __init__(self,
                 filename: str,
                 file_format: str,
                 directory: Path,
                 speckle_data: SpeckleData) -> None:

        self.filename = filename
        self.file_format = file_format
        self.directory = directory
        self.speckle_data = speckle_data

        self.px = 1/plt.rcParams['figure.dpi']

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
            proportion = Speckle.colour_count(self, image, proportion)
            if proportion > self.speckle_data.proportion_goal:
                image = np.zeros((self.speckle_data.size_y, self.speckle_data.size_x))
            else:
                break
            num_dots += 30
        print(proportion)
        filtered = gaussian_filter(image, 0.9)
        if self.speckle_data.white_bg:
            filtered = filtered * -1 + 1

        Speckle.plot_image(self, filtered)

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

    def colour_count(self, image: np.ndarray, proportion: float) -> float:
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

    def plot_image(self, image: np.ndarray):
        '''
        Defines figure size and formatting (no axes), plots the image array in greyscale and saves that figure to the specified filename in the specified location
            Parameters:
                image (arrray): A 2D array to be plotted
        '''
        plt.figure(figsize=((self.speckle_data.size_x * self.px), (self.speckle_data.size_y  * self.px)))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap='grey', vmin=0, vmax=1)
        os.chdir(self.directory)
        filepath = os.getcwd()
        filename_full = self.filename + '.' + self.file_format
        plt.savefig(os.path.join(filepath, filename_full), format=self.file_format, bbox_inches='tight', pad_inches=0, dpi=self.speckle_data.image_res)
        plt.show()
        plt.close()
