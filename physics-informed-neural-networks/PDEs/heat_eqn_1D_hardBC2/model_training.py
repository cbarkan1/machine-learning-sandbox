"""
Solve the 1D heat equation, d/dt u(x,t) = alpha * d2/dx2 u(x,t)

Boundary conditions:
u(x,t) = 0 at all boundaries.

"""

import numpy as np
import torch
import torch.optim as optim
from model_architecture import PINN
import matplotlib.pyplot as plt


def initial_condition(x):
    # Initial condition u(x,t=0)
    return torch.sin(np.pi * x) + torch.sin(2*np.pi * x)


# PDE parameters
alpha = 0.1  # Thermal diffusivity
x_range = (0, 1)
t_range = (0, 1)


# Random seed
torch.manual_seed(10)


model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
n_epochs = 5000

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # Sample random points in the domain
    x = torch.rand(1000, 1) * (x_range[1] - x_range[0]) + x_range[0]
    t = torch.rand(1000, 1) * (t_range[1] - t_range[0]) + t_range[0]
    
    # Model predictions
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    
    # Derivatives for PDE loss
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    pde_residual = u_t - alpha * u_xx
    
    # Initial condition residual
    x_ic = torch.rand(100, 1) * (x_range[1] - x_range[0]) + x_range[0]
    t_ic = torch.zeros_like(x_ic)
    u_ic = model(x_ic, t_ic)
    ic_residual = u_ic - initial_condition(x_ic)
    

    # Compute the total loss
    loss = torch.mean(pde_residual**2) + torch.mean(ic_residual**2)
    
    # Backpropagation and optimization step
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}')


# Plotting model solution
model.eval()
xs = torch.linspace(x_range[0], x_range[1], 100)
ts = torch.linspace(t_range[0], t_range[1], 100)
x_mesh, t_mesh = torch.meshgrid(xs, ts)
x_flat = x_mesh.reshape(-1, 1)
t_flat = t_mesh.reshape(-1, 1)

with torch.no_grad():
    u_pred = model(x_flat, t_flat).reshape(x_mesh.shape)

plt.figure(figsize=(10, 8))
plt.pcolormesh(x_mesh, t_mesh, u_pred)
plt.colorbar(label='Temperature')
plt.xlabel('x')
plt.ylabel('t')
plt.title('PINN solution')
plt.show()


# Save the model parameters
torch.save(model.state_dict(), 'weights1.pth')
