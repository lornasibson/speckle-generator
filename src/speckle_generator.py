import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle
import io

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
    plt.show()
    # plt.savefig('speckle_pattern.png')
    

    # img = Image.open('speckle_pattern.png')
    # im_data = np.array(img)
    # h, w = im_data.shape[:2]
    # colours, counts = np.unique(im_data.reshape(-1,3), axis=0, return_counts=1)
    # for index, colour in enumerate(colours):
    #     count = counts[index]
    #     proportion = (100 * count) / (h * w)
    #     black = np.array([0, 0, 0])
    #     if np.array_equal(colour, black):
    #         print(proportion)
        #print(f"   Colour: {colour}, count: {count}, proportion: {proportion:.2f}%")

def array_speckle():
    px = 1/plt.rcParams['figure.dpi'] #pixel to inch conversion
    counter = 0
    size_x = 800
    size_y = 600
    proportion = 0
    proportion_goal = 50
    radius = 10
    image = np.zeros((size_y, size_x))
    plt.figure(figsize=(size_x*px, size_y*px))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='gray')
    fig = plt.gcf()
    ax = fig.gca()
    while proportion < proportion_goal:
        pos_x = np.random.randint(0, size_x)
        pos_y = np.random.randint(0, size_y)
        circ = plt.Circle((pos_x, pos_y), radius, color='w')
        ax.add_patch(circ)
    
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')#dpi=36)#DPI)
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        h, w = img_arr.shape[:2]
        colours, counts = np.unique(img_arr.reshape(-1,3), axis=0, return_counts=1)
        for index, colour in enumerate(colours):
            count = counts[index]
            all_proportion = (100 * count) / (h * w)
            black = np.array([0, 0, 0])
            if np.array_equal(colour, black):
                proportion = all_proportion
            counter += 1
    plt.show()

#Main script
array_speckle()

