# hours wasted: 2

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self,
                 num_classes,
                 hidden_dims,
                 dropout,
                 kernel1,
                 kernel2,
                 kernel3,
                 kernel4,
                 pool_kernel,
                 time_bins=400,
                 channels=22,
                 depth=25,
                 scale=2):
        super(CNNLSTM, self).__init__()

        self.magic_number = time_bins + kernel3 - 1

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels,
                      out_channels=depth,
                      kernel_size=kernel1),
            nn.ReLU(),
            nn.BatchNorm1d(depth),
            nn.Dropout(dropout)
        )

        self.L_pool = time_bins - kernel1 + 1

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=depth,
                      out_channels=depth,
                      kernel_size=kernel2),
            nn.ELU(),
            nn.BatchNorm2d(depth),
            nn.MaxPool2d(kernel_size=(1, pool_kernel)),
            nn.Dropout(dropout)
        )

        self.L_pool = self.L_pool - kernel2[1] + 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=depth,
                      out_channels=depth * scale,
                      kernel_size=kernel3),
            nn.ELU(),
            nn.BatchNorm1d(depth * scale),
            nn.MaxPool1d(kernel_size=pool_kernel),
            nn.Dropout(dropout)
        )

        self.L_pool = self.L_pool - kernel3 + 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=depth * scale,
                      out_channels=self.magic_number,
                      kernel_size=kernel4),
            nn.ReLU(),
            nn.BatchNorm1d(self.magic_number),
            nn.MaxPool1d(kernel_size=pool_kernel),
            nn.Dropout(dropout)
        )

        self.L_pool = self.L_pool - kernel4 + 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1

        self.lstm = nn.LSTM(input_size=self.L_pool,
                            hidden_size=hidden_dims,
                            num_layers=1,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(in_features=hidden_dims, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.unsqueeze(2)
        x = self.conv2(x)
        x = x.squeeze(2)
        x = self.conv3(x)
        x = self.conv4(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.dense(x[:, -1])

        return F.log_softmax(x, dim=1)
