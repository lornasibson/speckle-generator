import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def even_grid():
    width = 800
    height = 600
    x_number = int(width / 50)
    y_number = int(height / 50)
    px = 1/plt.rcParams['figure.dpi'] #pixel to inch conversion
    x_values = np.linspace(0, width, x_number)
    y_values = np.linspace(0, height, y_number)
    x_coords, y_coords = np.meshgrid(x_values, y_values)

    dot_size = 100
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

def dot_area(num_values, px):
    area = 0
    for dot in range(num_values):
        diam_pts = dot**0.5
        diam_inch = diam_pts * (1/72)
        rad_inch = diam_inch / 2
        rad_px = rad_inch / px
        dot_area = np.pi * (rad_px**2)
        area += dot_area
    return area

def random_speckle():
    width = 800
    height = 600
    x_number = int(width / 20)
    y_number = int(height / 20)
    px = 1/plt.rcParams['figure.dpi'] #pixel to inch conversion
    x_values = np.linspace(0, width, x_number)
    y_values = np.linspace(0, height, y_number) 
    
    x_coords = np.zeros((y_number, x_number))
    y_coords = np.zeros((y_number, x_number))

    for i in range(x_number):
        for j in range(y_number):
            x_coords[j][i] = np.random.randint(0, width)

    for i in range(x_number):
        for j in range(y_number):
            y_coords[j][i] = np.random.randint(0, height)   

    num_values = x_number * y_number
    area = dot_area(num_values, px)
    

    tot_area = width * height
    print(tot_area)
    bw_ratio = (area / tot_area) * 100
    print('B/w ratio:', bw_ratio, '%')

    dot_size = 100
    plt.figure(figsize=(width*px, height*px))
    plt.scatter(x_coords, y_coords, dot_size, marker='o', color='k')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('speckle_pattern.png')
    
    
    img = Image.open('speckle_pattern.png')
    im_data = np.array(img)
    h, w = im_data.shape[:2]
    colours, counts = np.unique(im_data.reshape(-1,3), axis=0, return_counts=1)
    for index, colour in enumerate(colours):
        count = counts[index]
        proportion = (100 * count) / (h * w)
        print(type(colour))
        #print(f"   Colour: {colour}, count: {count}, proportion: {proportion:.2f}%")


    
#Main script
random_speckle()

