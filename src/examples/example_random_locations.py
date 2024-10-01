from specklegenerator.specklegenerator import Speckle, SpeckleData

seed = 5
data = SpeckleData(size_x=10, size_y=10, radius=1)
speckler = Speckle(data, seed)
image = speckler.make()