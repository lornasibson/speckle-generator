import numpy as np
import matplotlib.pyplot as plt

height = 40
width = 30
x_number = 10
y_number = 10
x_values = np.linspace(0, width, x_number)
y_values = np.linspace(0, height, y_number)
x_coords, y_coords = np.meshgrid(x_values, y_values)

dot_size = 20
plt.plot(x_coords, y_coords, marker='o', color='k', linestyle='none', markersize=dot_size)
plt.xticks([])
plt.yticks([])
plt.show()
