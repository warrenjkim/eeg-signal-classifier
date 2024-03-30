import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Positional(nn.Module):
    def __init__(self, d_model=200, dropout=0.5, max_len=5000):
        super(Positional, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, 1, d_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(1), 0, :].unsqueeze(0)
        return self.dropout(x)


class CTransformer(nn.Module):
    def __init__(self,
                 num_classes=4,
                 hidden_dims=256,
                 dropout=0.3647999634687216,
                 kernel1=5,
                 kernel2=(1, 25),
                 pool_kernel=3,
                 in_channels=22,
                 depth=25,
                 time_bins=400,
                 channels=22,
                 d_model=200,
                 nhead=4,
                 num_encoder_layers=3):
        super(CTransformer, self).__init__()

        self.hidden_dims = time_bins + kernel2[1] - 1

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                             out_channels=depth,
                                             kernel_size=kernel1),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(depth),
                                   nn.Dropout(dropout))

        self.L_pool = time_bins - kernel1 + 1

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=depth,
                                             out_channels=depth*8,
                                             kernel_size=kernel2),
                                   nn.ELU(),
                                   nn.BatchNorm2d(depth * 8),
                                   nn.MaxPool2d(kernel_size=(1, pool_kernel)),
                                   nn.Dropout(dropout))

        self.L_pool = self.L_pool - kernel2[1] + 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1
        self.hidden_dims = self.L_pool * self.hidden_dims

        self.flatten = nn.Flatten(start_dim=2)

        self.encoding = Positional(d_model, dropout=0.2)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model,
                                                    nhead=nhead,
                                                    dim_feedforward=hidden_dims,
                                                    dropout=0.2,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_layers=num_encoder_layers)

        self.dense = nn.Linear(in_features=depth*8,
                               out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = x.unsqueeze(2)
        x = self.conv2(x)
        x = x.squeeze(2)

        x = x.transpose(1, 2)
        x = self.encoding(x)
        x = self.transformer_encoder(x)

        x = x[:, -1, :]
        x = self.dense(x)

        return F.log_softmax(x, dim=1)
