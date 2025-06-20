import torch
import torch.nn as nn

class NARXModel(nn.Module):
    def __init__(self, input_size=37, hidden_sizes=[64, 64, 64], output_size=6):
        super(NARXModel, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def get_model():
    return NARXModel()


def get_loss():
    return nn.MSELoss()


def get_optimizer(model, lr=1e-3):
    return torch.optim.Adam(model.parameters(), lr=lr)
