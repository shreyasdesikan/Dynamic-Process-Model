import torch
import torch.nn as nn
from torch.optim import AdamW


class SimpleANNModel(nn.Module):
    def __init__(self, input_size=65, hidden_sizes=[64, 64, 64], output_size=6):
        super(SimpleANNModel, self).__init__()
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


class LSTMModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=1, output_size=6):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # Input x shape: (batch_size, window_size * 13)
        seq = x.reshape(x.size(0), -1, 13)  # (batch, window_size, 13)
        out, (hn, _) = self.lstm(seq)
        last_hidden = hn[-1]
        return self.fc(last_hidden)


class StackedLSTMModel(nn.Module):
    def __init__(self, input_size=13, output_size=6):
        super(StackedLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # Input x shape: (batch_size, window_size * 13)
        seq = x.reshape(x.size(0), -1, 13)  # (batch, window_size, 13)
        out1, _ = self.lstm1(seq)
        out2, (hn2, _) = self.lstm2(out1)
        last_hidden = hn2[-1]
        return self.fc(last_hidden)


def get_model(input_size=65, model_type="ann", window_size=5):
    if model_type == "lstm":
        return LSTMModel(input_size=13)
    elif model_type == "stacked_lstm":
        return StackedLSTMModel(input_size=13)
    else:
        return SimpleANNModel(input_size=input_size)


def get_loss(use_huber=False):
    return nn.SmoothL1Loss() if use_huber else nn.MSELoss()


def get_optimizer(model, lr=1e-3, use_adamw=False):
    if use_adamw:
        return torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr)

