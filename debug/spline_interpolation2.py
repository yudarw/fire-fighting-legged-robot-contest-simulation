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

# Calculate derivatives
x_dot, y_dot, z_dot = splev(u_fine, tck, der=1)
x_ddot, y_ddot, z_ddot = splev(u_fine, tck, der=2)
x_dddot, y_dddot, z_dddot = splev(u_fine, tck, der=3)

# Time-scaling function (example: cubic polynomial)
def time_scaling(t, T):
    return 3*(t/T)**2 - 2*(t/T)**3

# Generate trajectory points with controlled motion
T = 10  # Total time
time_steps = np.linspace(0, T, 100)
trajectory = np.zeros((len(time_steps), 3))

for i, t in enumerate(time_steps):
    s = time_scaling(t, T)
    trajectory[i, 0] = np.interp(s, u_fine, x_fine)
    trajectory[i, 1] = np.interp(s, u_fine, y_fine)
    trajectory[i, 2] = np.interp(s, u_fine, z_fine)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original points
ax.scatter(x, y, z, color='red', label='Original Points')

# Plot interpolated path
ax.plot(x_fine, y_fine, z_fine, color='blue', label='Spline Interpolated Path')

# Plot trajectory
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='green', label='Controlled Trajectory')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()