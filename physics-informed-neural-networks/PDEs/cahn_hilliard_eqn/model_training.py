"""
Solve the 1D Cahn-Hilliard equation.

This is a non-linear PDE that describes liquid-liquid phase separation.

Much slower due to the complexity of the equation

"""


import numpy as np
import torch
import torch.optim as optim
from model_architecture import PINN
import matplotlib.pyplot as plt


def initial_condition(x):
    # Initial condition u(x,t=0)
    return torch.sin(np.pi * x)


# PDE parameters
D = 0.1  # Diffusion coefficient
gamma = 0.1  # phase boundary thickness parameter
x_range = (0, 1)
t_range = (0, 0.5)


# Random seed
torch.manual_seed(10)

model = PINN()
#model.load_state_dict(torch.load('weights_gamma0p1_ver2.pth'))
optimizer = optim.Adam(model.parameters(), lr=0.001)
n_epochs = 10000

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # Sample random points in the domain
    x = torch.rand(1000, 1) * (x_range[1] - x_range[0]) + x_range[0]
    t = torch.rand(1000, 1) * (t_range[1] - t_range[0]) + t_range[0]
    
    # Model predictions
    x.requires_grad = True
    t.requires_grad = True
    c = model(x, t)
    
    # Derivatives for PDE loss
    c_t = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    mu = c**3 - c - gamma*c_xx  # Chemical potential field
    mu_x = torch.autograd.grad(mu, x, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
    mu_xx = torch.autograd.grad(mu_x, x, grad_outputs=torch.ones_like(mu_x), create_graph=True)[0]
    
    pde_residual = c_t - D * mu_xx
    
    # Initial condition residual
    x_ic = torch.rand(100, 1) * (x_range[1] - x_range[0]) + x_range[0]
    t_ic = torch.zeros_like(x_ic)
    c_ic = model(x_ic, t_ic)
    ic_residual = c_ic - initial_condition(x_ic)
    
    # Left boundary condition
    t_left = torch.rand(100, 1) * (t_range[1] - t_range[0]) + t_range[0]
    x_left = torch.ones_like(t_left) * x_range[0]
    t_left.requires_grad = True
    x_left.requires_grad = True
    c_left = model(x_left, t_left)
    c_x_left = torch.autograd.grad(c_left, x_left, grad_outputs=torch.ones_like(c_left), create_graph=True)[0]
    c_xx_left = torch.autograd.grad(c_x_left, x_left, grad_outputs=torch.ones_like(c_x_left), create_graph=True)[0]
    c_xxx_left = torch.autograd.grad(c_xx_left, x_left, grad_outputs=torch.ones_like(c_xx_left), create_graph=True)[0]

    # Right boundary condition residual
    t_right = torch.rand(100, 1) * (t_range[1] - t_range[0]) + t_range[0]
    x_right = torch.ones_like(t_right) * x_range[0]
    t_right.requires_grad = True
    x_right.requires_grad = True
    c_right = model(x_right, t_right)
    c_x_right = torch.autograd.grad(c_right, x_right, grad_outputs=torch.ones_like(c_right), create_graph=True)[0]
    c_xx_right = torch.autograd.grad(c_x_right, x_right, grad_outputs=torch.ones_like(c_x_right), create_graph=True)[0]
    c_xxx_right = torch.autograd.grad(c_xx_right, x_right, grad_outputs=torch.ones_like(c_xx_right), create_graph=True)[0]

    # Total loss
    loss = torch.mean(pde_residual**2) + torch.mean(ic_residual**2) \
        + torch.mean(c_x_left**2) + torch.mean(c_x_right**2) \
        + torch.mean(c_xxx_left**2) + torch.mean(c_xxx_right**2)
    
    # Backpropagation and optimization step
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')


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
metadata = {'D': D, 'gamma': gamma, 'x_range': x_range, 't_range': t_range}
save_dict = {'model_state': model.state_dict(), 'metadata':, metadata}
torch.save(save_dict, 'weights_gamma0p1_ver1.pth')
