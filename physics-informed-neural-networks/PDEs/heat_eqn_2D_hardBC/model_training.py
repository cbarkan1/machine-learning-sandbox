import torch
import torch.optim as optim
from model_architecture import PINN
import numpy as np
import matplotlib.pyplot as plt
from time import time
pi = np.pi


def initial_condition(x, y):
    return torch.sin(pi*x) * torch.sin(2*pi*y)


def exact_solution(x, y, t):
    return torch.sin(pi*x) * torch.sin(2*pi*y) * torch.exp(-5*alpha*pi**2*t)


# Set random seed for reproducibility
torch.manual_seed(1234)

# Define the problem parameters
alpha = 0.05  # Thermal diffusivity


# Create the PINN model
model = PINN()
model.load_state_dict(torch.load('weights1.pth'))

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of training iterations
n_epochs = 1000

time0 = time()

# Training loop
for epoch in range(n_epochs):

    optimizer.zero_grad()
    
    # Evaluate model at random points in domain
    x = torch.rand(1000, 1, requires_grad=True)
    y = torch.rand(1000, 1, requires_grad=True)
    t = torch.rand(1000, 1, requires_grad=True)
    u = model(x, y, t)
    
    # PDE residual
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    pde_residual = u_t - alpha * (u_xx + u_yy)
    
    # initial condition residual
    x_ic = torch.rand(200, 1)
    y_ic = torch.rand(200, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = model(x_ic, y_ic, t_ic)
    ic_residual = u_ic - initial_condition(x_ic, y_ic)
    
    # Total loss
    loss = torch.mean(pde_residual**2) + torch.mean(ic_residual**2)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}, Time/epoc: {(time()-time0)/epoch:.4f}s')

# Evaluation
model.eval()
x_eval = torch.linspace(0, 1, 50)
y_eval = torch.linspace(0, 1, 50)
t_eval = torch.tensor([0.0, 0.5, 1.0])

X, Y = torch.meshgrid(x_eval, y_eval, indexing='ij')

fig, axes = plt.subplots(2, len(t_eval), figsize=(12, 6))
for i, t in enumerate(t_eval):
    T = t * torch.ones_like(X)
    with torch.no_grad():
        U = model(X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1)).reshape(X.shape)
    
    im = axes[0, i].pcolormesh(X, Y, U)
    axes[0, i].set_title(f't = {t.item():.1f}')
    fig.colorbar(im, ax=axes[0, i])

    U_exact = exact_solution(X, Y, T)
    im = axes[1, i].pcolormesh(X, Y, U-U_exact)
    axes[1, i].set_title(f't = {t.item():.1f}')
    fig.colorbar(im, ax=axes[1, i])

plt.tight_layout()
plt.show()

# Save the model parameters
torch.save(model.state_dict(), 'weights1.pth')
