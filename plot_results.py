import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp

class LagrangianNN(nn.Module):
    def __init__(self):
        super(LagrangianNN, self).__init__()
        self.potential_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )

        self.mass_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 3)
        )

    def forward(self, q, q_dot):
        # embed q into sine and cosine for better learning of periodicity
        q_embed = torch.cat([torch.sin(q), torch.cos(q)], dim=-1)
        # potential energy compute
        V = self.potential_net(q_embed).squeeze(-1)

        # mass matrix compute
        mass_outs = self.mass_net(q_embed)

        # Ensure positive definiteness of the mass matrix
        L11 = torch.nn.functional.softplus(mass_outs[:, 0]) + 1e-3
        L22 = torch.nn.functional.softplus(mass_outs[:, 1]) + 1e-3
        L21 = mass_outs[:, 2]

        # Construct the mass matrix M = L L^T and Kinetic energy T = 0.5 * q_dot^T * M * q_dot
        q_dot1, q_dot2 = q_dot[:, 0], q_dot[:, 1]

        v1 = L11 * q_dot1 + L21 * q_dot2
        v2 = L22 * q_dot2

        T = 0.5 * (v1**2 + v2**2)

        return T - V  # Lagrangian L = T - V

model = LagrangianNN()
model.load_state_dict(torch.load('lnn_model.pth'))
model.eval()

# setup data and physics parameters
m1, m2 = 1.0, 1.0  # masses (kg)
L1, L2 = 1.0, 1.0  # lengths of rods (m)
g = 9.81  # acceleration due to gravity (m/s^

data = pd.read_csv('double_pendulum_dataset.csv')
sample_traj = data[data['trajectory_id'] == 0].copy()
time_array = sample_traj['time'].values
dt = time_array[1] - time_array[0]

# Derive equations of motion from the model
def get_lnn_accelerations(model, q_step, q_dot_step):
    q = torch.tensor(q_step, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    q_dot = torch.tensor(q_dot_step, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    
    # Lagrangian prediction
    L = model(q, q_dot).sum()

    dL_dq = torch.autograd.grad(L, q, create_graph=True)[0]
    dL_dqdot = torch.autograd.grad(L, q_dot, create_graph=True)[0]

    # Hessian Matrix M = d^2L/dq_dot^2
    M_rows = []
    C_rows = []

    for i in range(2):
        grad_q_dot_i = dL_dqdot[:, i].sum()

        M_row = torch.autograd.grad(grad_q_dot_i, q_dot, retain_graph=True)[0]
        C_row = torch.autograd.grad(grad_q_dot_i, q, retain_graph=True)[0]

        M_rows.append(M_row)
        C_rows.append(C_row)

    M = torch.stack(M_rows, dim=1)
    C = torch.stack(C_rows, dim=1)

    C_q_dot = torch.bmm(C, q_dot.unsqueeze(2)).squeeze(2)
    rhs = dL_dq - C_q_dot

    # Add jitter and solve
    jitter = torch.eye(2, device=M.device).unsqueeze(0) * 1e-4
    M_reg = M + jitter

    q_ddot_pred = torch.linalg.solve(M_reg, rhs.unsqueeze(2)).squeeze(2)

    return q_ddot_pred.detach().numpy()[0]

# RK45 integration using LNN predicted physics
def lnn_ode(t, y):
    # The state vector y is [theta1, theta2, omega1, omega2]
    q_step = np.array([y[0], y[1]])
    q_dot_step = np.array([y[2], y[3]])
    
    # Get the angular accelerations from the neural network
    q_ddot = get_lnn_accelerations(model, q_step, q_dot_step)
    
    # Return dy/dt = [omega1, omega2, alpha1, alpha2]
    return [y[2], y[3], q_ddot[0], q_ddot[1]]

y0 = [
    sample_traj['theta1'].iloc[0], 
    sample_traj['theta2'].iloc[0],
    sample_traj['omega1'].iloc[0], 
    sample_traj['omega2'].iloc[0]
]
# Performing RK45 integration using LNN predicted physics
sol = solve_ivp(
    fun=lnn_ode, 
    t_span=[time_array[0], time_array[-1]], 
    y0=y0, 
    t_eval=time_array, 
    method='RK45'
)

q_pred = sol.y[0:2, :].T       
q_dot_pred = sol.y[2:4, :].T

# data frame for plotting
predicted_traj = pd.DataFrame({
    'time': sol.t,
    'theta1': q_pred[:, 0],
    'theta2': q_pred[:, 1],
    'omega1': q_dot_pred[:, 0],
    'omega2': q_dot_pred[:, 1]
})


# Plotting the results

# Time domain comparison of the trajectories
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(sample_traj['time'], sample_traj['theta1'], label='Ground Truth', color='black')
ax1.plot(predicted_traj['time'], predicted_traj['theta1'], label='LNN Prediction', color='blue', linestyle='--')
ax1.set_ylabel('Theta 1 (rad)')
ax1.legend()

ax2.plot(sample_traj['time'], sample_traj['theta2'], label='Ground Truth', color='black')
ax2.plot(predicted_traj['time'], predicted_traj['theta2'], label='LNN Prediction', color='red', linestyle='--')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Theta 2 (rad)')
ax2.legend()
plt.suptitle('Time-Domain Divergence')
plt.tight_layout()
plt.show()


# phase space comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(sample_traj['theta1'], sample_traj['omega1'], label='Truth', color='gray', alpha=0.5, linewidth=3)
ax1.plot(predicted_traj['theta1'], predicted_traj['omega1'], label='LNN', color='blue', linewidth=1)
ax1.set_xlabel('Theta 1'); ax1.set_ylabel('Omega 1'); ax1.set_title('Phase Space: Limb 1'); ax1.legend()

ax2.plot(sample_traj['theta2'], sample_traj['omega2'], label='Truth', color='gray', alpha=0.5, linewidth=3)
ax2.plot(predicted_traj['theta2'], predicted_traj['omega2'], label='LNN', color='red', linewidth=1)
ax2.set_xlabel('Theta 2'); ax2.set_ylabel('Omega 2'); ax2.set_title('Phase Space: Limb 2'); ax2.legend()
plt.tight_layout()
plt.show()


# Energy comparison
def compute_energy(df):
    t1, t2 = df['theta1'], df['theta2']
    o1, o2 = df['omega1'], df['omega2']

    V = -(m1 + m2) * g * L1 * np.cos(t1) - m2 * g * L2 * np.cos(t2)
    T = 0.5 * m1 * (L1 * o1)**2 + 0.5 * m2 * ((L1 * o1)**2 + (L2 * o2)**2 + 2 * L1 * L2 * o1 * o2 * np.cos(t1 - t2))
    return T + V

energy_truth = compute_energy(sample_traj)
energy_pred = compute_energy(predicted_traj)

plt.figure(figsize=(10, 5))
plt.plot(sample_traj['time'], energy_truth, label='Ground Truth', color='black')
plt.plot(predicted_traj['time'], energy_pred, label='LNN Prediction', color='green', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Total Energy (J)')
plt.title('Total Mechanical energy over time')
plt.legend()
plt.show()
