import torch
import torch.nn as nn
pi = 3.1415926535898


class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super(PINN, self).__init__()
        
        self.input_dim = input_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.Tanh()

    def forward(self, x_list, t):
        X = torch.cat([*x_list, t], dim=1)
        X = self.activation(self.input_layer(X))
        for layer in self.hidden_layers:
            X = self.activation(layer(X))
        X = self.output_layer(X)
        for dim in range(self.input_dim - 1):
            X *= torch.sin(pi*x_list[dim]) # Now I'm hard coding the distinction between space and time.
        return X
