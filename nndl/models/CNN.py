# hours wasted: 2

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,
                 num_classes=4,
                 dropout=0.5,
                 kernel=7,
                 pool_kernel=3,
                 time_bins=400,
                 channels=22,
                 depth=25):
        super(CNN, self).__init__()

        self.magic_number = time_bins + kernel - 1

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels,
                      out_channels=depth,
                      kernel_size=kernel),
            nn.ReLU(),
            nn.BatchNorm1d(depth),
            nn.Dropout(dropout)
        )

        self.L_pool = time_bins - kernel + 1

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=depth,
                      out_channels=depth * 2,
                      kernel_size=(1, kernel)),
            nn.ELU(),
            nn.BatchNorm2d(depth * 2),
            nn.MaxPool2d(kernel_size=(1, pool_kernel)),
            nn.Dropout(dropout)
        )

        self.L_pool = self.L_pool - kernel + 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=depth * 2,
                      out_channels=depth * 4,
                      kernel_size=kernel),
            nn.ELU(),
            nn.BatchNorm1d(depth * 4),
            nn.MaxPool1d(kernel_size=pool_kernel),
            nn.Dropout(dropout)
        )

        self.L_pool = self.L_pool - kernel + 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=depth * 4,
                      out_channels=self.magic_number,
                      kernel_size=kernel),
            nn.ELU(),
            nn.BatchNorm1d(self.magic_number),
            nn.MaxPool1d(kernel_size=pool_kernel),
            nn.Dropout(dropout)
        )

        self.L_pool = self.L_pool - kernel + 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1

        self.dense = nn.Linear(in_features=self.L_pool * self.magic_number,
                               out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.unsqueeze(2)      # match dims for Conv2d().
        x = self.conv2(x)
        x = x.squeeze(2)        # match dims for Conv1d().
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1) # match dims for affine.
        x = self.dense(x)

        return F.log_softmax(x, dim=1)
