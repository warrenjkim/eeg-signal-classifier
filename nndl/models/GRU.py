import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nndl.models.TimeDistributed import TimeDistributed

class GRU(nn.Module):
    def __init__(self,
                 num_classes=4,
                 hidden_dims=128,
                 num_layers=2,
                 dropout=0.5,
                 kernel=5,
                 pool_kernel=2,
                 time_bins=400,
                 channels=22,
                 depth=32):
        super(GRU, self).__init__()

        self.height = np.sqrt(depth)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels,
                      out_channels=depth,
                      kernel_size=kernel**2),
            nn.ReLU(),
            nn.BatchNorm1d(depth),
            nn.Dropout(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=depth,
                      out_channels=depth * 2,
                      kernel_size=kernel**2),
            nn.ELU(),
            nn.BatchNorm1d(depth * 2),
            nn.MaxPool1d(kernel_size=pool_kernel),
            nn.Dropout(dropout)
        )

        self.gru1 = nn.GRU(input_size=depth * 2,
                           hidden_size=hidden_dims // 2,
                           num_layers=num_layers,
                           bias=True,
                           batch_first=True,
                           bidirectional=True)
        self.gru2 = nn.GRU(input_size=hidden_dims,
                           hidden_size=hidden_dims,
                           num_layers=num_layers,
                           bias=True,
                           batch_first=True,
                           bidirectional=True)

        self.dense = TimeDistributed(nn.Linear(in_features=hidden_dims * 2,
                                               out_features=num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.dense(x)
        x = x[:, -1, :]
        return F.log_softmax(x, dim=1)


