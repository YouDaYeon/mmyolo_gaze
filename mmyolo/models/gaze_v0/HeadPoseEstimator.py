import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadPoseEstimator(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_dim=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.fc = nn.Linear(hidden_dim, out_dim)  # yaw, pitch, roll (or just yaw, pitch)

    def forward(self, x):
        x = self.pool(x)               # [B, C, 1, 1]
        x = F.relu(self.conv(x))      # [B, hidden_dim, 1, 1]
        x = x.view(x.size(0), -1)     # [B, hidden_dim]
        out = self.fc(x)              # [B, 3]
        return out
