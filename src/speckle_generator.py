import numpy as np
import matplotlib.pyplot as plt

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
