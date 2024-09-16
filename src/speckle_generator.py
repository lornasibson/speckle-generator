import os
import io
import numpy as np
import matplotlib as mpl
from mpl_toolkits import mplot3d 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import fft
from PIL import Image
# from matplotlib.patches import Circle

# Using scatter plot
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
    area = dot_area(dot_size, px)

    tot_area = width * height
    bw_ratio = (area / tot_area) * 100
    print('B/w ratio:', bw_ratio, '%')

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

# Using Circle patch
def circle_speckle():
    mpl.rcParams['savefig.pad_inches'] = 0
    px = 1/plt.rcParams['figure.dpi'] #pixel to inch conversion
    size_x = 800
    size_y = 600
    print(size_x*px, size_y*px)
    num_dots = 600
    radius = 10
    image = np.zeros((size_y, size_x))
    plt.figure(figsize=(size_x*px, size_y*px))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='grey')
    fig = plt.gcf()
    ax = fig.gca()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    for i in range(num_dots):
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
    total_prop = 0
    for index, colour in enumerate(colours):
        count = counts[index]
        all_proportion = (100 * count) / (h * w)
        if all_proportion > 10:
            print("Colour: ", colour, "Count: ", count, "Proportion: ", all_proportion)
        total_prop += all_proportion
        # white = np.array([255, 255, 255])
        # black = np.array([0, 0, 0])
        # if np.array_equal(colour, white):
        #     proportion = all_proportion
        #     print('White proportion: ', proportion)
        # elif np.array_equal(colour, black):
        #     print('Black proportion:', all_proportion)

    plt.savefig('speckle_pattern_2', bbox_inches='tight', pad_inches=0)
    print(total_prop)
    plt.show()

# Using a 2D array
def array_speckle(size_x, size_y, radius, proportion_goal, filename, file_format, white_bg, image_res):
    mpl.rcParams['savefig.pad_inches'] = 0
    px = 1/plt.rcParams['figure.dpi'] #pixel to inch conversion
    num_dots = 50
    proportion = 100
    
    image = np.zeros((size_y, size_x))

    circle = np.zeros((radius * 2, radius * 2))
    xs, ys = np.meshgrid(np.arange(2 * radius), np.arange(2* radius))
    r = ((xs - radius) ** 2 + (ys - radius) ** 2)** 0.5
    if (circle[r < radius] == 0).all():
        circle[r < radius] = 255
    else:
        circle[r < radius] = 0
    count = 0
    
    while proportion > proportion_goal:
        for i in range(num_dots):
            pos_x = np.random.randint(radius, (size_x - radius))
            pos_y = np.random.randint(radius, (size_y - radius))
        
            x_start, x_end = pos_x - radius, pos_x + radius
            y_start, y_end = pos_y - radius, pos_y + radius

            if x_start >= 0 and x_end <= size_x and y_start >= 0 and y_end <= size_y:
                image[y_start:y_end, x_start:x_end] += circle
            else:
                print(f"Skipping out-of-bounds circle at position ({pos_x}, {pos_y})")

        h, w = image.shape[:2]
        colours, counts = np.unique(image, return_counts=1)
        for index, colour in enumerate(colours):
            count = counts[index]
            all_proportion = (100 * count) / (h * w)
            if colour == 0.0:
                proportion = all_proportion
                # print(proportion)
        num_dots += 1
    
    # filtered = gaussian_filter(image, 0.6)
    # print(image[1])
    if white_bg == 'Yes':
        image = image * -1 + 1
    plt.figure(figsize=(size_x*px, size_y*px))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='grey', vmin=0, vmax=1)
    os.chdir('/home/lorna/speckle-generator')
    filepath = os.getcwd()
    filename_full = filename + '.' + file_format
    plt.savefig(os.path.join(filepath, filename_full), format=file_format, bbox_inches='tight', pad_inches=0, dpi=image_res)
    # plt.show()
    
    fourier_transform(image)

def fourier_transform(image):
    ft = fft.fft2(image)
    size = ft.shape
    # print(ft[1])
    fft_shifted = fft.fftshift(ft)
    magnitude_spectrum = np.abs(fft_shifted)

    # Spatial frequency
    freqx = fft.fftfreq(image.shape[0])
    avg_x_freq = np.mean(freqx)

    print()
    freqy = fft.fftfreq(image.shape[1])
    avg_y_freq = np.mean(freqy)

    # plt.imshow(np.log(magnitude_spectrum), cmap='gray')
    # plt.colorbar()
    # plt.show()

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection = '3d')
    ax.grid()
    x = np.arange(1, size[1] + 1, 1)
    y = np.arange(1, size[0] + 1, 1)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, ft)
    plt.show()

    # x-dir fourier transform
    # ft_x = fft.fft(image[0])
    # print(np.mean(ft_x))


def match_id_speckle():
    img = Image.open('matchid_generated_speckle.bmp')
    img.load()
    data = np.asarray(img)
    colours, counts = np.unique(data, return_counts=1)
    for index, colour in enumerate(colours):
        count = counts[index]
        print('Colour:', colour, 'Count:', count)



#Main script
if __name__ == '__main__':
    size_x = 800
    size_y = 600
    radius = 8
    proportion_goal = 50
    filename = 'speckle_pattern_no_blur'
    file_format = 'tiff'
    white_bg = 'Yes' #Set to yes for white background with black speckles, set to 'No' for black background with white speckles
    image_res = 25
    array_speckle(size_x, size_y, radius, proportion_goal, filename, file_format, white_bg, image_res)
    # match_id_speckle()


