import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('double_pendulum_dataset.csv')

# Prepare the dataset
q_tensor = torch.tensor(data[['theta1', 'theta2']].values, dtype=torch.float32)
q_dot_tensor = torch.tensor(data[['omega1', 'omega2']].values, dtype=torch.float32)
q_ddot_tensor = torch.tensor(data[['alpha1', 'alpha2']].values, dtype=torch.float32)

# Define the LNN model
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

# Euler-Lagrange Physics Loss
def euler_lagrange_residual(model, q, q_dot, q_ddot):
    q = q.clone().requires_grad_(True)
    q_dot = q_dot.clone().requires_grad_(True)

    # Predict the Lagrangian
    L = model(q, q_dot)

    # Compute the derivatives
    dL_dq = torch.autograd.grad(L, q,grad_outputs= torch.ones_like(L), create_graph=True)[0]
    dL_dqdot = torch.autograd.grad(L, q_dot,grad_outputs= torch.ones_like(L), create_graph=True)[0]

    M_rows = []
    C_rows = []

    for i in range(2):
        grad_q_dot_i = dL_dqdot[:, i].sum()

        M_row = torch.autograd.grad(grad_q_dot_i, q_dot, create_graph=True)[0]
        C_row = torch.autograd.grad(grad_q_dot_i, q, create_graph=True)[0]

        M_rows.append(M_row)
        C_rows.append(C_row)

    M = torch.stack(M_rows, dim=1)
    C = torch.stack(C_rows, dim=1)

    C_q_dot = torch.bmm(C, q_dot.unsqueeze(2)).squeeze(2)
    rhs = dL_dq - C_q_dot

    jitter = torch.eye(2, device=M.device) * 1e-4
    M_reg = M + jitter

    q_ddot_pred = torch.linalg.solve(M_reg, rhs.unsqueeze(2)).squeeze(2)

    loss = torch.nn.functional.mse_loss(q_ddot_pred, q_ddot)
    return loss

# Training the model
model = LagrangianNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
epochs = 1000
batch_size = 256

print("Starting training...")
loss_history = []
dataset = torch.utils.data.TensorDataset(q_tensor, q_dot_tensor, q_ddot_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for q_batch, q_dot_batch, q_ddot_batch in dataloader:
        optimizer.zero_grad()
        loss = euler_lagrange_residual(model, q_batch, q_dot_batch, q_ddot_batch)
        loss.backward()

        # gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    scheduler.step(avg_loss)
    if epoch % 50 == 0:
        print(f'Epoch [{epoch}/{epochs}], Physics Loss: {avg_loss:.6f}')

print("Training completed")

# Plot the training loss
torch.save(model.state_dict(), 'lnn_model.pth')
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Log MSE Loss')
plt.title('LNN Training: Euler-Lagrange Residual Loss')
plt.savefig('lnn_training_loss.png')
plt.show()
