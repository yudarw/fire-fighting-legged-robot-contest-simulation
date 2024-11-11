import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the set of points in 3D space
points = np.array([
    [0, 0, 0],
    [1, 2, 1],
    [2, 3, 4],
    [3, 5, 2]
])

# Linear interpolation function
def linear_interpolation(p1, p2, num_points=100):
    return np.linspace(p1, p2, num_points)

# Generate interpolated points
interpolated_points = []
for i in range(len(points) - 1):
    interpolated_points.append(linear_interpolation(points[i], points[i+1]))

# Convert list of arrays to a single array
interpolated_points = np.vstack(interpolated_points)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', label='Original Points')

# Plot interpolated path
ax.plot(interpolated_points[:, 0], interpolated_points[:, 1], interpolated_points[:, 2], color='blue', label='Interpolated Path')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()