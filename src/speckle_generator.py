import numpy as np
import matplotlib.pyplot as plt

def even_grid():
    height = 40
    width = 30
    x_number = 10
    y_number = 10
    x_values = np.linspace(0, width, x_number)
    y_values = np.linspace(0, height, y_number)
    x_coords, y_coords = np.meshgrid(x_values, y_values)

    #dot_size = 20
    #Making every marker a different size
    num_dots = x_number * y_number
    dot_size = np.random.randint(20, 500, num_dots)

    # plt.plot(x_coords, y_coords, marker='o', color='k', linestyle='none', markersize=dot_size)
    plt.scatter(x_coords, y_coords, dot_size, marker='o', color='k')
    plt.xticks([])
    plt.yticks([])
    plt.show()

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

plt.imshow(img, cmap='binary', vmin=0, vmax=1)
plt.axis('off')
plt.show()

