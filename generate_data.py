import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

# Parameters for physical system
m1, m2 = 1.0, 1.0  # masses (kg)
L1, L2 = 1.0, 1.0  # lengths of rods (m)
g = 9.81  # acceleration due to gravity (m/s^2)


# Function to compute derivatives for the double pendulum system
def double_pendulum_derivatives(t, state):
    theta1, omega1, theta2, omega2 = state

    delta = theta1 - theta2

    den1 = L1 *(2 * m1 + m2 - m2 * np.cos(2 * delta))
    den2 = L2 *(2 * m1 + m2 - m2 * np.cos(2 * delta))


    num1 = -g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(delta) * m2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(delta))
    num2 = 2 * np.sin(delta) * (omega1**2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + omega2**2 * L2 * m2 * np.cos(delta))

    alpha1 = num1 / den1
    alpha2 = num2 / den2

    return [omega1, alpha1, omega2, alpha2]


# Settings for the simulation
num_trajectories = 300
time_span = (0, 5)  # time span for the simulation (seconds)
fps = 100 
t_eval = np.linspace(time_span[0], time_span[1], int((time_span[1] - time_span[0]) * fps))

data = []

# Simulate multiple trajectories with random initial conditions
for i in range(num_trajectories):
    # Random initial angles and angular velocities
    theta1_0 = np.random.uniform(-np.pi, np.pi)
    omega1_0 = np.random.uniform(-3, 3)
    theta2_0 = np.random.uniform(-np.pi, np.pi)
    omega2_0 = np.random.uniform(-3, 3)

    initial_state = [theta1_0, omega1_0, theta2_0, omega2_0]

    # Solve the ODE for the double pendulum system
    sol = solve_ivp(double_pendulum_derivatives, time_span, initial_state, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-9)

    # Calculate the accelerations for the trajectory
    theta1, omega1, theta2, omega2 = sol.y
    alpha1_list, alpha2_list = [], []

    for j in range(len(sol.t)):
        state = [theta1[j], omega1[j], theta2[j], omega2[j]]
        _, alpha1, _, alpha2 = double_pendulum_derivatives(sol.t[j], state)
        alpha1_list.append(alpha1)
        alpha2_list.append(alpha2)

    # Store the results in a DataFrame to store in the dataset
    df = pd.DataFrame({
        'trajectory_id': i,
        'time': sol.t,
        'theta1': theta1,
        'omega1': omega1,
        'theta2': theta2,
        'omega2': omega2,
        'alpha1': alpha1_list,
        'alpha2': alpha2_list
    })
    data.append(df)

# Dataset exportation, uncomment to generate csv file
final_data = pd.concat(data, ignore_index=True)
final_data.to_csv('double_pendulum_dataset.csv', index=False)


# plotting a sample trajectory
sample_traj = final_data[final_data['trajectory_id'] == 0]
plt.figure(figsize=(12, 6))
plt.plot(sample_traj['time'], sample_traj['theta1'], label='Theta 1')
plt.plot(sample_traj['time'], sample_traj['theta2'], label='Theta 2')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Sample Trajectory of Double Pendulum')
plt.legend()
plt.savefig('double_pendulum_sample_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()

# Plotting the trajectories in Cartesian coordinates
x1 = L1 * np.sin(sample_traj['theta1'])
y1 = -L1 * np.cos(sample_traj['theta1'])
x2 = x1 + L2 * np.sin(sample_traj['theta2'])
y2 = y1 - L2 * np.cos(sample_traj['theta2'])
time_array = sample_traj['time']

plt.figure(figsize=(8, 8))
plt.plot(x1, y1, label='Limb 1', color='blue')
plt.plot(x2, y2, label='Limb 2', color='red')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('2D Sample Trajectory of Double Pendulum in Cartesian Coordinates')
plt.legend()
plt.axis('equal')
plt.savefig('double_pendulum_sample_trajectory_cartesian.png', dpi=300, bbox_inches='tight')
plt.show()

# 3D plot (X, Y, Time)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x1, y1, time_array, label='Mass 1', color='blue', alpha=0.7)
ax.plot(x2, y2, time_array, label='Mass 2', color='red', linewidth=1.5)

ax.set_title('Double Pendulum Trajectory Evolving Over Time')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Time (s)')
ax.legend()

ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.savefig('double_pendulum_sample_trajectory_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# plotting phase space for a sample trajectory
sample_traj = final_data[final_data['trajectory_id'] == 0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(sample_traj['theta1'], sample_traj['omega1'], color='blue')
ax1.set_xlabel('Theta 1 (rad)')
ax1.set_ylabel('Omega 1 (rad/s)')
ax1.set_title('Limb 1 Phase Portrait sample trajectory')

ax2.plot(sample_traj['theta2'], sample_traj['omega2'], color='red')
ax2.set_xlabel('Theta 2 (rad)')
ax2.set_ylabel('Omega 2 (rad/s)')
ax2.set_title('Limb 2 Phase Portrait sample trajectory')

plt.tight_layout()
plt.savefig('double_pendulum_phase_space.png', dpi=300, bbox_inches='tight')
plt.show()


