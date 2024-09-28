import torch
import torch.nn as nn
pi = 3.14159265359

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x, t):

        # Concatenate x and t
        inputs = torch.cat([x, t], dim=1)

        #sin(x) factor forces model to obey BC
        return torch.sin(pi*x) * self.net(inputs)
