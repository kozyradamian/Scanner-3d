import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

array_3d_model = np.array(np.random.random((1, 512*424, 3))*100)
array_3d_model2 = np.array(np.random.random((1,512*10,3))*100)
array_3d_model = np.append(array_3d_model, array_3d_model2, axis=0)


x = np.array([])
y = np.array([])
z = np.array([])

for i in range(0, len(array_3d_model)):
    x=np.append(x, array_3d_model[i][:][:, 0])
    y=np.append(y, array_3d_model[i][:][:, 1])
    z=np.append(z, array_3d_model[i][:][:, 2])


x, y, z = np.broadcast_arrays(x, y, z)

c = np.tile(array_3d_model.ravel()[:, None], [1, 3])

# Do the plotting in a single call.
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()