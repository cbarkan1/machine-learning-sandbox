import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),  # 4 inputs: x, y, z, t
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)   # 1 output: u
        )

    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        return self.net(inputs)
