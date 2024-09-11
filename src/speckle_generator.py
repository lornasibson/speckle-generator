import numpy as np
import matplotlib.pyplot as plt

def even_grid():
    width = 800
    height = 600
    x_number = 10
    y_number = 10
    px = 1/plt.rcParams['figure.dpi'] #pixel to inch conversion
    x_values = np.linspace(0, width, x_number)
    y_values = np.linspace(0, height, y_number)
    x_coords, y_coords = np.meshgrid(x_values, y_values)

    dot_size = 500
    #Making every marker a different size
    area = dot_area(dot_size, px)

    tot_area = width * height
    bw_ratio = (area / tot_area) * 100
    print('B/w ratio:', bw_ratio, '%')

    # plt.plot(x_coords, y_coords, marker='o', color='k', linestyle='none', markersize=dot_size)
    plt.figure(figsize=(width*px, height*px))
    plt.scatter(x_coords, y_coords, dot_size, marker='o', color='k')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def dot_area(dot_size, px):
    area = 0
    for dot in range(dot_size):
        diam_pts = dot**0.5
        diam_inch = diam_pts * (1/72)
        rad_inch = diam_inch / 2
        rad_px = rad_inch / px
        dot_area = np.pi * (rad_px**2)
        area += dot_area
    return area


#Trying similar code from ElsevierSoftwareX Github
def insert_circle(image, position, radius):
    circle = np.zeros(((radius*2), (radius*2)), dtype=np.uint8)
    x_values, y_values = np.meshgrid(np.arange(2*radius), np.arange(2*radius))
    r = ((x_values - radius)**2 + (y_values - radius)**2)**0.5
    circle[r < radius] = 1

    x_min = max(position[0] - radius, 0)
    x_max = min(position[0] + radius, image.shape[0])
    y_min = max(position[1] - radius, 0)
    y_max = min(position[1] + radius, image.shape[1])

    circle_x_min = radius - (position[0] - x_min)
    circle_x_max = circle_x_min + (x_max - x_min)
    circle_y_min = radius - (position[1] - y_min)
    circle_y_max = circle_y_min + (y_max - y_min)

    image[x_min:x_max, y_min:y_max] = np.maximum(image[x_min:x_max, y_min:y_max], circle[circle_x_min:circle_x_max, circle_y_min:circle_y_max])
    return image

def dot_speckle(size, n_dots, dot_radius_max, dot_radius_min):
    size_x, size_y = size
    img = np.zeros((size_x, size_y), dtype=np.uint8)

    for i in range(n_dots):
        pos_x = np.random.randint(0, size_x)
        pos_y = np.random.randint(0, size_y)

        radius = np.random.randint(dot_radius_min, dot_radius_max)

        img = insert_circle(img, (pos_x, pos_y), radius)

    return img
    
#Main script
size = (100, 100)
n_dots = 500
dot_radius_max = 10
dot_radius_min = 8
img = dot_speckle(size, n_dots, dot_radius_max, dot_radius_min)

even_grid()

