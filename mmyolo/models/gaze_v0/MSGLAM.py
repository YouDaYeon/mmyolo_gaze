import torch
import torch.nn as nn
import torch.nn.functional as F

from .CBAM import CBAM

class GMWNonLocal(nn.Module):
    def __init__(self, in_channels, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        F_q = self.q_conv(x).view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        F_k = self.k_conv(x).view(B, C, -1).permute(0, 2, 1)
        F_v = self.v_conv(x).view(B, C, -1).permute(0, 2, 1)

        # Gaussian similarity
        q_expand = F_q.unsqueeze(2)  # [B, HW, 1, C]
        k_expand = F_k.unsqueeze(1)  # [B, 1, HW, C]
        dist = torch.norm(q_expand - k_expand, dim=3) ** 2  # [B, HW, HW]
        A = torch.exp(-dist / (2 * self.sigma ** 2))
        A = F.softmax(A, dim=-1)

        out = torch.bmm(A, F_v)  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out + x  # residual

class MSGLAM(nn.Module):
    def __init__(self, in_channels, groups=4, rounds=4):
        super().__init__()
        self.groups = groups
        self.rounds = rounds
        self.cbams = nn.ModuleList()
        # self.gmws = nn.ModuleList()
        # self.tail_convs = nn.ModuleList()

        for _ in range(groups):
            self.cbams.append(CBAM(in_channels // groups))
            # self.gmws.append(GMWNonLocal(in_channels // groups))
            # self.tail_convs.append(nn.Conv2d(2 * (in_channels // groups), in_channels // groups, kernel_size=1))

        # self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        # self.downsample = nn.Sequential(
        #     nn.Conv2d((in_channels // groups), (in_channels // groups), kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d((in_channels // groups)),
        #     nn.ReLU()
        # )

    def forward(self, x):
        for _ in range(self.rounds):
            splits = torch.chunk(x, self.groups, dim=1)

            # # downsample for gmw
            # splits_down = []
            # for i in range(self.groups):
            #     splits_down.append(self.downsample(splits[i]))

            # outputs = []
            cbam_out = []

            for i in range(self.groups):
                # cbam_out = self.cbams[i](splits[i])
                # gmw_out = self.gmws[i](splits_down[i])
                # gmw_out_up = F.interpolate(gmw_out, size=splits[i].size()[2:], mode='bilinear')
                # combined = torch.cat([cbam_out, gmw_out_up], dim=1)
                # outputs.append(self.tail_convs[i](combined))

                cbam_out.append(self.cbams[i](splits[i]))

            # x = torch.cat(outputs, dim=1)
            x = torch.cat(cbam_out, dim=1)
        return x
