# hours wasted: too many

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self,
                 num_classes=4,
                 hidden_dims=256,
                 dropout=0.5,
                 kernel_size=5,
                 pool_kernel=2,
                 time_bins=400,
                 channels=22,
                 depth=64):
        super(CNN_LSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels,
                      out_channels=depth,
                      kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm1d(depth),
            nn.Dropout(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=depth,
                      out_channels=depth * 2,
                      kernel_size=(1, depth)),
            nn.ELU(),
            nn.BatchNorm2d(depth * 2),
            nn.MaxPool2d(kernel_size=(1, depth)),
            nn.Dropout(dropout)
        )

        self.L_pool = self.L_pool - kernel_size+ 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=depth * 2,
                      out_channels=depth * 4,
                      kernel_size=kernel_size),
            nn.ELU(),
            nn.BatchNorm1d(depth * 4),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(input_size=5,
                            hidden_size=hidden_size // 2,
                            num_layers=3,
                            batch_first=True,
                           bidirectional=True)

        self.dense = nn.Linear(in_features=hidden_size,
                               out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.unsqueeze(2)      # match dims for Conv2d().
        x = self.conv2(x)
        x = x.squeeze(2)        # match dims for Conv1d().
        x = self.conv3(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return F.log_softmax(x, dim=1)


class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        tmp = x.congituous().view(-1, x.size(-1))
        y = x.layer(tmp)
        y = y.contiguous().view(x.size(0), -1, y.size(-1))

        return y

class OTHER_CNN_LSTM:
    def __init__(self,
                 num_classes=4,
                 hidden_dims=256,
                 dropout=0.5,
                 kernel_size=5,
                 pool_kernel=2,
                 time_bins=400,
                 channels=22,
                 depth=64):
        super(OTHER_CNN_LSTM, self).__init__()

        self.height = np.sqrt(depth)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels,
                      out_channels=depth,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(depth),
            nn.Dropout(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=depth,
                      out_channels=depth * 2,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ELU(),
            nn.BatchNorm1d(depth * 2),
            nn.MaxPool1d(kernel_size=pool_kernel),
            nn.Dropout(dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=depth * 2,
                      out_channels=depth * 4,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ELU(),
            nn.BatchNorm1d(depth * 4),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(input_size=depth * 4,
                            hidden_size=hidden_size,
                            num_layers=3,
                            batch_first=True,
                           bidirectional=True)

        self.dense = TimeDistributed(nn.Linear(in_features=hidden_size * 2,
                                               out_features=num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return F.log_softmax(x, dim=1)


