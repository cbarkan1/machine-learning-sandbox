import torch
import torch.nn as nn
pi = 3.14159265359

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),  # 3 inputs: x, y, t
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)   # 1 output: u
        )

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return torch.sin(pi*x)*torch.sin(pi*y)*self.net(inputs)
