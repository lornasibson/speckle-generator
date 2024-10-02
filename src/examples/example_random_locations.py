import numpy as np
from specklegenerator.specklegenerator import Speckle, SpeckleData, _px_locations

# seed = 5
# data = SpeckleData(size_x=10, size_y=10, radius=1)
# speckler = Speckle(data, seed)
# image = speckler.make()

size_x = 4
size_y = 4

grid_shape, x_px_trans, y_px_trans = _px_locations(size_x, size_y)

print(f"{grid_shape=}")
print(f"{x_px_trans=}")
print(f"{y_px_trans=}")
print(y_px_trans.shape)

comparison = x_px_trans == y_px_trans
equal_arrays = comparison.all()

print(equal_arrays)