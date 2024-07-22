import torch
from model_architecture import PINN
import matplotlib.pyplot as plt

x_range = (0, 1)
t_range = (0, 0.5)

# Loading model with trained weights
model = PINN()
model.load_state_dict(torch.load('weights_gamma0p1_ver2.pth'))
model.eval()

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


plt.figure(figsize=(10, 8))
plt.plot(x_mesh[:,0],u_pred[:,0])


plt.figure(figsize=(10, 8))
plt.plot(x_mesh[:,-1],u_pred[:,-1])

plt.show()
