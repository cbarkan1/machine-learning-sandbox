import torch
import torch.optim as optim
from model_architecture import PINN
from time import time
pi = 3.1415926535898


def initial_condition(x, y, z):
    return torch.sin(pi*x) * torch.sin(2*pi*y) + torch.sin(pi*z)


def exact_solution(x, y, z, t):
    return torch.sin(pi*x) * torch.sin(2*pi*y) * torch.exp(-5*alpha*pi**2*t) \
        + torch.sin(pi*z) * torch.exp(-alpha*pi**2*t)


# Set random seed for reproducibility
torch.manual_seed(1234)

# Define the problem parameters
alpha = 0.05  # Thermal diffusivity


# Create the PINN model
model = PINN()
model.load_state_dict(torch.load('weights2.pth'))

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Number of training iterations
n_epochs = 10000

time0 = time()

# Training loop
for epoch in range(n_epochs):

    optimizer.zero_grad()
    
    # Evaluate model at random points in domain
    x = torch.rand(2000, 1, requires_grad=True)
    y = torch.rand(2000, 1, requires_grad=True)
    z = torch.rand(2000, 1, requires_grad=True)
    t = torch.rand(2000, 1, requires_grad=True)
    u = model(x, y, z, t)
    
    # PDE residual
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    pde_residual = u_t - alpha * (u_xx + u_yy + u_zz)
    
    # initial condition residual
    x_ic = torch.rand(200, 1)
    y_ic = torch.rand(200, 1)
    z_ic = torch.rand(200, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = model(x_ic, y_ic, z_ic, t_ic)
    ic_residual = u_ic - initial_condition(x_ic, y_ic, z_ic)
    
    # boundary condition residual
    x_bc = torch.cat([torch.rand(200, 1),
                      torch.zeros(200, 1),
                      torch.zeros(200, 1),
                      torch.rand(200, 1),
                      torch.ones(200, 1),
                      torch.ones(200, 1)])
    y_bc = torch.cat([torch.zeros(200, 1),
                      torch.rand(200, 1),
                      torch.zeros(200, 1),
                      torch.ones(200, 1),
                      torch.rand(200, 1),
                      torch.ones(200, 1)])
    z_bc = torch.cat([torch.zeros(200, 1),
                      torch.zeros(200, 1),
                      torch.rand(200, 1),
                      torch.ones(200, 1),
                      torch.ones(200, 1),
                      torch.rand(200, 1)])
    t_bc = torch.rand(1200, 1)
    u_bc = model(x_bc, y_bc, z_bc, t_bc)
    bc_residual = u_bc
    
    # Compute the total loss
    # It's actually much faster to compute loss all at one like this, than to 
    # use loss += ... to add in all the terms separately. Not sure why.
    loss = torch.mean(pde_residual**2) + torch.mean(ic_residual**2) + torch.mean(bc_residual**2)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Time/epoc: {(time()-time0)/epoch:.4f}s')

# Save the model parameters
torch.save(model.state_dict(), 'weights2.pth')
