import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# Set random seed for reproducibility
torch.manual_seed(1234)

# Define the problem parameters
D = 1.0  # Diffusion coefficient
gamma = 0.01  # Interface parameter
x_range = (0, 1)
t_range = (0, 1)

# Create the PINN model
model = PINN()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of training iterations
n_iterations = 1000

# Training loop
for epoch in range(n_iterations):
    optimizer.zero_grad()
    
    # Sample random points in the domain
    x = torch.rand(1000, 1) * (x_range[1] - x_range[0]) + x_range[0]
    t = torch.rand(1000, 1) * (t_range[1] - t_range[0]) + t_range[0]
    
    # Compute the model predictions
    x.requires_grad = True
    t.requires_grad = True
    c = model(x, t)
    
    # Compute the derivatives
    c_t = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    c_xxx = torch.autograd.grad(c_xx, x, grad_outputs=torch.ones_like(c_xx), create_graph=True)[0]
    c_xxxx = torch.autograd.grad(c_xxx, x, grad_outputs=torch.ones_like(c_xxx), create_graph=True)[0]
    
    # Compute the PDE residual
    f = c**3 - c - gamma * c_xx
    f_xx = torch.autograd.grad(torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0],
                                x, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    pde_residual = c_t - D * f_xx
    
    # Compute the initial condition residual
    x_ic = torch.rand(100, 1) * (x_range[1] - x_range[0]) + x_range[0]
    t_ic = torch.zeros_like(x_ic)
    c_ic = model(x_ic, t_ic)
    ic_residual = c_ic - 0.5 * (1 + torch.tanh((x_ic - 0.5) / (2 * gamma**0.5)))  # Example initial condition
    
    # Compute the boundary condition residuals
    x_bc = torch.tensor([[0.0], [1.0]])  # Boundary points
    t_bc = torch.rand(2, 1) * (t_range[1] - t_range[0]) + t_range[0]
    x_bc.requires_grad = True
    c_bc = model(x_bc, t_bc)
    c_x_bc = torch.autograd.grad(c_bc, x_bc, grad_outputs=torch.ones_like(c_bc), create_graph=True)[0]
    c_xxx_bc = torch.autograd.grad(c_xx, x_bc, grad_outputs=torch.ones_like(c_xx), create_graph=True, allow_unused=True)[0]
    bc_residual = torch.cat([c_x_bc, c_xxx_bc])
    
    # Compute the total loss
    loss = torch.mean(pde_residual**2) + torch.mean(ic_residual**2) + torch.mean(bc_residual**2)
    
    # Backpropagation and optimization step
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{n_iterations}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
x_eval = torch.linspace(x_range[0], x_range[1], 100).unsqueeze(1)
t_eval = torch.linspace(t_range[0], t_range[1], 100).unsqueeze(1)
X, T = torch.meshgrid(x_eval.squeeze(), t_eval.squeeze(), indexing='ij')
X_flat = X.reshape(-1, 1)
T_flat = T.reshape(-1, 1)

with torch.no_grad():
    c_pred = model(X_flat, T_flat).reshape(X.shape)

# Plot the results
plt.figure(figsize=(10, 8))
plt.pcolormesh(X, T, c_pred, shading='auto')
plt.colorbar(label='Concentration')
plt.xlabel('x')
plt.ylabel('t')
plt.title('PINN Solution of Cahn-Hilliard Equation')
plt.show()