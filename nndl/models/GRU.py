import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 num_classes,
                 in_channels=22):
        super(GRU, self).__init__()

        self.gru = nn.GRU(input_size=in_channels,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bias=True,
                          batch_first=True)

        self.dropout = nn.Dropout(0.8)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.affine = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x, _ = self.gru(x, None)
        x = self.dropout(x)
        x = self.dense(x[:, -1, :])
        x = self.dropout(x)
        x = self.affine(x)

        return F.log_softmax(x, dim=1)
