import torch
import torch.nn as nn

"""
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

    def forward(self, x_list, t):
        inputs = torch.cat([*x_list, t], dim=1)
        return self.net(inputs)
"""


class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super(PINN, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.Tanh()

    def forward(self, x_list, t):
        X = torch.cat([*x_list, t], dim=1)
        X = self.activation(self.input_layer(X))
        for layer in self.hidden_layers:
            X = self.activation(layer(X))
        return self.output_layer(X)
