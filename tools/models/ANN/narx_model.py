import torch
import torch.nn as nn
import torch.nn.functional as F
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

class BidirectionalLSTMWithAttention(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, fc_dim=64, output_dim=6, num_layers=2):
        super(BidirectionalLSTMWithAttention, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Attention over sequence output
        self.attn_layer = nn.Linear(hidden_dim * 2, 1)

        # Residual projection to match FC dimension
        self.res_proj = nn.Linear(hidden_dim * 2, fc_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.out = nn.Linear(fc_dim, output_dim)

    def forward(self, x):
        # x: (batch, window_size * input_dim)
        batch_size, flat_len = x.size()
        window_size = flat_len // 13  # hardcoded input_dim = 13
        x = x.view(batch_size, window_size, 13)

        lstm_out, _ = self.lstm(x)  # (batch, window, 2 * hidden_dim)

        attn_scores = self.attn_layer(lstm_out)           # (batch, window, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, window, 1)

        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, 2 * hidden_dim)

        # Residual block with projection
        fc1_out = F.relu(self.fc1(context))               # (batch, fc_dim)
        residual = self.res_proj(context)                 # (batch, fc_dim)
        fc2_out = F.relu(self.fc2(fc1_out + residual))    # (batch, fc_dim)

        return self.out(fc2_out)                          # (batch, output_dim)


def get_model(input_size=65, model_type="ann", window_size=5):
    if model_type == "lstm":
        return LSTMModel(input_size=13)
    elif model_type == "stacked_lstm":
        return StackedLSTMModel(input_size=13)
    elif model_type == "bilstm_attn":
        return BidirectionalLSTMWithAttention(input_dim=13, output_dim=6)
    else:
        return SimpleANNModel(input_size=input_size)


def get_loss(use_huber=False):
    return nn.SmoothL1Loss() if use_huber else nn.MSELoss()


def get_optimizer(model, lr=1e-3, use_adamw=False):
    if use_adamw:
        return torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr)

