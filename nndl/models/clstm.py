import torch
import torch.nn as nn
import torch.nn.functional as F

class CLSTM(nn.Module):
    def __init__(self,
                 num_classes=4,
                 dropout=0.3647999634687216,
                 kernel1=5,
                 kernel2=(1, 25),
                 kernel3=6,
                 kernel4=5,
                 pool_kernel=3,
                 in_channels=22,
                 depth=25,
                 num_layers=2,
                 time_bins=400):
        super(CLSTM, self).__init__()

        self.hidden_dims = time_bins + kernel3 - 1

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                             out_channels=depth,
                                             kernel_size=kernel1),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(depth),
                                   nn.Dropout(dropout))

        self.L_pool = time_bins - kernel1 + 1

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=depth,
                                             out_channels=depth*2,
                                             kernel_size=kernel2),
                                   nn.ELU(),
                                   nn.BatchNorm2d(depth * 2),
                                   nn.MaxPool2d(kernel_size=(1, pool_kernel)),
                                   nn.Dropout(dropout))

        self.L_pool = self.L_pool - kernel2[1] + 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1

        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=depth*2,
                                             out_channels=depth*4,
                                             kernel_size=kernel3),
                                   nn.ELU(),
                                   nn.BatchNorm1d(depth * 4),
                                   nn.MaxPool1d(kernel_size=pool_kernel),
                                   nn.Dropout(dropout))

        self.L_pool = self.L_pool - kernel3 + 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1

        self.conv4 = nn.Sequential(nn.Conv1d(in_channels=depth*4,
                                             out_channels=self.hidden_dims,
                                             kernel_size=kernel4),
                                   nn.ELU(),
                                   nn.BatchNorm1d(self.hidden_dims),
                                   nn.MaxPool1d(kernel_size=pool_kernel),
                                   nn.Dropout(dropout))

        self.L_pool = self.L_pool - kernel4 + 1
        self.L_pool = (self.L_pool - pool_kernel) // pool_kernel + 1
        self.hidden_dims = self.L_pool * self.hidden_dims

        self.linear = nn.Sequential(nn.Linear(in_features=self.hidden_dims,
                                              out_features=depth*8),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(depth * 8),
                                   nn.Dropout(dropout))

        self.lstm = nn.LSTM(input_size=depth*8,
                            hidden_size=depth*4,
                            num_layers=num_layers,
                            bias=True,
                            batch_first=True,
                            bidirectional=True)

        self.dense = nn.Linear(in_features=depth*8,
                               out_features=num_classes)


    def forward(self, x):
        x = self.conv1(x)

        x = x.unsqueeze(2)
        x = self.conv2(x)
        x = x.squeeze(2)

        x = self.conv3(x)
        x = self.conv4(x)

        x = torch.flatten(x, 1)
        x = self.linear(x)

        x, _ = self.lstm(x)
        x = self.dense(x)

        return F.log_softmax(x, dim=1)
