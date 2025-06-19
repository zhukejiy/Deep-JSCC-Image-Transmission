import torch
import torch.nn as nn
import torch.nn.functional as F


class AFModule(nn.Module):
    def __init__(self, channel):
        super(AFModule, self).__init__()
        self.fc1 = nn.Linear(channel + 1, channel)
        self.fc2 = nn.Linear(channel, channel)

    def forward(self, x, snr):  # x: [B, C, H, W], snr: [B, 1]
        B, C, H, W = x.shape
        # Global average pooling
        context = torch.mean(x, dim=(2, 3))  # [B, C]
        # Concatenate SNR
        context = torch.cat([context, snr], dim=1)  # [B, C+1]
        # Fully connected layers
        scale = self.fc1(context)
        scale = F.relu(scale)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale).view(B, C, 1, 1)
        return x * scale
