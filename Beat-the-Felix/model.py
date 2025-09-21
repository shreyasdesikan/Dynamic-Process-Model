import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
    
class BidirectionalLSTMWithMultiHead(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, fc_dim=64, output_dim=6, num_layers=2):
        super(BidirectionalLSTMWithMultiHead, self).__init__()

        # BiLSTM block
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Attention layer
        self.attn_layer = nn.Linear(hidden_dim * 2, 1)

        # Shared residual FC block
        self.fc1 = nn.Linear(hidden_dim * 2, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.res_proj = nn.Linear(hidden_dim * 2, fc_dim)

        # Output heads matching order: ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']
        self.head_ctpm = nn.Linear(fc_dim, 2)    # c, T_PM
        self.head_dxyz = nn.Linear(fc_dim, 3)    # d50, d90, d10
        self.head_ttm = nn.Linear(fc_dim, 1)     # T_TM

    def forward(self, x):
        # x: (batch, window_size * input_dim)
        batch_size, flat_len = x.size()
        window_size = flat_len // 13
        x = x.view(batch_size, window_size, 13)

        # BiLSTM + Attention
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn_layer(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Shared residual FC block
        fc1_out = F.relu(self.fc1(context))
        residual = self.res_proj(context)
        shared = F.relu(self.fc2(fc1_out + residual))

        # Outputs in the correct state order
        out_ctpm = self.head_ctpm(shared)     # c, T_PM
        out_dxyz = self.head_dxyz(shared)     # d50, d90, d10
        out_ttm = self.head_ttm(shared)       # T_TM

        # Concatenate in correct order
        return torch.cat([out_ctpm, out_dxyz, out_ttm], dim=1)


def get_loss(use_huber=False):
    return nn.SmoothL1Loss() if use_huber else nn.MSELoss()


def get_optimizer(model, lr=1e-3, use_adamw=False):
    if use_adamw:
        return torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr)

