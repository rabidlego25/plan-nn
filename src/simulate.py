import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# Constants
G = 1.0 # Gravitational constant (normalized)
M = 1.0 # Mass of Central Planet (normalized)
dt = .01 # Time step
T = 10.0 # Total simulation time

# Initial Conditions
x0, y0 = 1.0, 0.0
vx0, vy0 = 0.0, 1.0

def simulate_orbit(x0, y0, vx0, vy0, dt, T):
	"""
	Simulates planetary motion using Newtonian gravity

	Args:
		x0, y0: Initial condition
		vx0, vy0: Initial velocity
		dt: Time step for numerical integration
		T: Total simulation time

	Returns:
		positions: Array of (x, y) positions
		velocities: Array of (vx, vy) velocities
		times: Array of time steps
	"""

	n_steps = int(T / dt)
	positions = np.zeros((n_steps,2))
	velocities = np.zeros((n_steps,2))
	times = np.zeros(n_steps)

	x,y = x0,y0
	vx,vy = vx0,vy0

	for i in range(n_steps):
		r = np.sqrt(x**2+y**2) # distance from origin
		ax = -G * M * x / r**3 # acceleration in x
		ay = -G * M * y / r**3 # acceleration in y

		# update velocity
		vx += ax * dt
		vy += ay * dt

		# update position
		x += vx * dt
		y += vy * dt

		# store results
		positions[i] = [x,y]
		velocities[i] = [vx,vy]
		times[i] = i * dt

	return positions, velocities, times

positions, velocities, times = simulate_orbit(x0,y0,vx0,vy0,dt,T)

# plot the trajector
plt.plot(positions[:,0], positions[:,1])
plt.title('Simulated Planet Trajector')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid()
# plt.show()

# save to data folder
output_dir = '../data'
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "positions.npy"), positions)
np.save(os.path.join(output_dir, "velocities.npy"), velocities)
np.save(os.path.join(output_dir, "times.npy"), times)
print(f'data saved to {output_dir}')

output_img_dir = "../results"
os.makedirs(output_img_dir, exist_ok=True)
output_img = os.path.join(output_img_dir, "trajectory_plt.png")
plt.savefig(output_img)

# implementation of data pre-processing
# normalizing positions and velocities
scaler = MinMaxScaler()
positions_normalized = scaler.fit_transform(positions)
velocities_normalized = scaler.fit_transform(velocities)

# save normalized data
output_dir2 = '../data'
os.makedirs(output_dir2, exist_ok=True)
np.save(os.path.join(output_dir2, 'positions_normalized.npy'), positions_normalized)
np.save(os.path.join(output_dir2, 'velocities_normalized.npy'), velocities_normalized)
np.save(os.path.join(output_dir2, 'times.npy'), times)


print(f'Plot saved to {output_img_dir}')
