import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev

# Define the set of points in 3D space
points = np.array([
    [0, 0, 0],
    [1, 2, 1],
    [2, 3, 4],
    [3, 5, 2]
])

# Extract x, y, z coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Spline interpolation
tck, u = splprep([x, y, z], s=0)
u_fine = np.linspace(0, 1, 100)
x_fine, y_fine, z_fine = splev(u_fine, tck)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original points
ax.scatter(x, y, z, color='red', label='Original Points')

# Plot interpolated path
ax.plot(x_fine, y_fine, z_fine, color='blue', label='Spline Interpolated Path')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()