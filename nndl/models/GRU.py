import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nndl.models.TimeDistributed import TimeDistributed

# class GRU(nn.Module):
#     def __init__(self,
#                  num_classes=4,
#                  hidden_dims=128,
#                  num_layers=2,
#                  dropout=0.5,
#                  kernel=5,
#                  pool_kernel=2,
#                  time_bins=400,
#                  channels=22,
#                  depth=32):
#         super(GRU, self).__init__()
#
#         self.height = np.sqrt(depth)
#
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels=channels,
#                       out_channels=depth,
#                       kernel_size=kernel),
#             nn.ReLU(),
#             nn.BatchNorm1d(depth),
#             nn.MaxPool1d(kernel_size=pool_kernel),
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(in_channels=depth,
#                       out_channels=depth * 2,
#                       kernel_size=kernel),
#             nn.ELU(),
#             nn.BatchNorm1d(depth * 2),
#             nn.MaxPool1d(kernel_size=pool_kernel),
#             nn.Dropout(dropout)
#         )
#
#         self.gru = nn.GRU(input_size=depth * 2,
#                            hidden_size=hidden_dims // 2,
#                            num_layers=num_layers,
#                            bias=True,
#                            batch_first=True,
#                            bidirectional=True)
#
#         self.td = nn.Sequential(
#             TimeDistributed(nn.Linear(in_features=hidden_dims,
#                                       out_features=hidden_dims // 2)),
#             TimeDistributed(nn.ReLU()),
#             TimeDistributed(nn.BatchNorm1d(hidden_dims // 2)),
#             TimeDistributed(nn.Dropout(dropout)),
#         )
#
#         self.dense = nn.Linear(in_features=hidden_dims // 2,
#                                out_features=num_classes)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.transpose(1, 2)
#         x, _ = self.gru(x)
#         x = self.td(x)
#         x = x[:, -1, :]
#         x = self.dense(x)
#         return F.log_softmax(x, dim=1)


# class GRU(nn.Module):
#     def __init__(self,
#                  num_classes=4,
#                  hidden_dims=128,
#                  num_layers=2,
#                  dropout=0.5,
#                  kernel=5,
#                  pool_kernel=2,
#                  time_bins=400,
#                  channels=22,
#                  depth=32):
#         super(GRU, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels=channels,
#                       out_channels=depth,
#                       kernel_size=kernel,
#                       stride=stride,
#                       padding=padding),
#             nn.ReLU(),
#             nn.BatchNorm1d(depth),
#             nn.Dropout(dropout)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=depth,
#                       out_channels=depth * 2,
#                       kernel_size=(1, depth),
#                       stride=stride,
#                       padding=padding),
#             nn.ELU(),
#             nn.BatchNorm2d(depth * 2),
#             nn.MaxPool2d(kernel_size=(1, pool_kernel)),
#             nn.Dropout(dropout)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(in_channels=depth * 2,
#                       out_channels=depth * 4,
#                       kernel_size=kernel,
#                       stride=stride,
#                       padding=padding + 1),
#             nn.ELU(),
#             nn.BatchNorm1d(depth * 4),
#             nn.MaxPool1d(kernel_size=pool_kernel),
#             nn.Dropout(dropout)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv1d(in_channels=depth * 4,
#                       out_channels=depth * 4,
#                       kernel_size=kernel,
#                       stride=stride,
#                       padding=padding + 3),
#             nn.ELU(),
#             nn.BatchNorm1d(depth * 4),
#             nn.MaxPool1d(kernel_size=pool_kernel),
#             nn.Dropout(dropout)
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv1d(in_channels=depth * 4,
#                       out_channels=depth * 8,
#                       kernel_size=kernel,
#                       stride=stride,
#                       padding=padding + 5),
#             nn.ELU(),
#             nn.BatchNorm1d(depth * 8),
#             nn.MaxPool1d(kernel_size=pool_kernel),
#             nn.Dropout(dropout)
#         )
#
#         self.nd = nn.Sequential(
#             nn.Linear(in_features=depth * 8, out_features=depth * 4),
#             nn.ReLU(),
#             nn.BatchNorm1d(depth * 4),
#             nn.Dropout(dropout),
#             nn.Linear(in_features=depth * 4, out_features=depth * 2),
#             nn.ReLU(),
#             nn.BatchNorm1d(depth * 2),
#             nn.Dropout(dropout),
#             nn.Linear(in_features=depth * 2, out_features=depth),
#             nn.ReLU(),
#             nn.BatchNorm1d(depth),
#             nn.Dropout(dropout),
#         )
#
#         self.gru = nn.GRU(input_size=depth * 8,
#                            hidden_size=depth * 4,
#                            num_layers=num_layers,
#                            bias=True,
#                            batch_first=True,
#                            bidirectional=True)
#
#         self.td = nn.Sequential(
#             TimeDistributed(nn.Linear(in_features=depth * 8, out_features=depth * 4)),
#             TimeDistributed(nn.ReLU()),
#             TimeDistributed(nn.BatchNorm1d(depth * 4)),
#             TimeDistributed(nn.Dropout(dropout)),
#             TimeDistributed(nn.Linear(in_features=depth * 4, out_features=depth * 2)),
#             TimeDistributed(nn.ReLU()),
#             TimeDistributed(nn.BatchNorm1d(depth * 2)),
#             TimeDistributed(nn.Dropout(dropout)),
#             TimeDistributed(nn.Linear(in_features=depth * 2, out_features=depth)),
#             TimeDistributed(nn.ReLU()),
#             TimeDistributed(nn.BatchNorm1d(depth)),
#             TimeDistributed(nn.Dropout(dropout)),
#         )
#         self.dense = nn.Linear(in_features=depth, out_features=num_classes)
#
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = x.unsqueeze(2)
#         x = self.conv2(x)
#         x = x.squeeze(2)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = x.transpose(1, 2)
#         x, _ = self.gru(x)
#         x = self.td(x)
#         x = x[:, -1, :]
#         x = self.dense(x)
#         return F.log_softmax(x, dim=1)


class GRU(nn.Module):
    def __init__(self,
                 num_classes=4,
                 hidden_dims=128,
                 num_layers=2,
                 dropout=0.5,
                 kernel=10,
                 stride=1,
                 padding=0,
                 pool_kernel=3,
                 time_bins=400,
                 channels=22,
                 depth=25):
        super(GRU, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels,
                      out_channels=depth,
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(depth),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=depth,
                      out_channels=depth * 2,
                      kernel_size=(1, depth),
                      stride=stride,
                      padding=padding),
            nn.ELU(),
            nn.BatchNorm2d(depth * 2),
            nn.MaxPool2d(kernel_size=(1, pool_kernel)),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=depth * 2,
                      out_channels=depth * 4,
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding + 1),
            nn.ELU(),
            nn.BatchNorm1d(depth * 4),
            nn.MaxPool1d(kernel_size=pool_kernel),
            nn.Dropout(dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=depth * 4,
                      out_channels=depth * 4,
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding + 3),
            nn.ELU(),
            nn.BatchNorm1d(depth * 4),
            nn.MaxPool1d(kernel_size=pool_kernel),
            nn.Dropout(dropout)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=depth * 4,
                      out_channels=depth * 8,
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding + 5),
            nn.ELU(),
            nn.BatchNorm1d(depth * 8),
            nn.MaxPool1d(kernel_size=pool_kernel),
            nn.Dropout(dropout)
        )

        self.nd = nn.Sequential(
            nn.Linear(in_features=depth * 8, out_features=depth * 4),
            nn.ReLU(),
            nn.BatchNorm1d(depth * 4),
            nn.Dropout(dropout),
            nn.Linear(in_features=depth * 4, out_features=depth * 2),
            nn.ReLU(),
            nn.BatchNorm1d(depth * 2),
            nn.Dropout(dropout),
            nn.Linear(in_features=depth * 2, out_features=depth),
            nn.ReLU(),
            nn.BatchNorm1d(depth),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(input_size=depth * 8,
                           hidden_size=depth * 4,
                           num_layers=num_layers,
                           bias=True,
                           batch_first=True,
                           bidirectional=True)

        self.td = nn.Sequential(
            TimeDistributed(nn.Linear(in_features=depth * 8, out_features=depth * 4)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.BatchNorm1d(depth * 4)),
            TimeDistributed(nn.Dropout(dropout)),
            TimeDistributed(nn.Linear(in_features=depth * 4, out_features=depth * 2)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.BatchNorm1d(depth * 2)),
            TimeDistributed(nn.Dropout(dropout)),
            TimeDistributed(nn.Linear(in_features=depth * 2, out_features=depth)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.BatchNorm1d(depth)),
            TimeDistributed(nn.Dropout(dropout)),
        )
        self.dense = nn.Linear(in_features=depth, out_features=num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = x.unsqueeze(2)
        x = self.conv2(x)
        x = x.squeeze(2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = self.td(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return F.log_softmax(x, dim=1)
