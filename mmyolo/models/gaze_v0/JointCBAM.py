import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'min']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            # min 실험 추가
            elif pool_type=='min':
                min_pool = -F.max_pool2d( -x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( min_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)  # b,c,1,1 -> b,c,h,w
        # return x * scale
        return channel_att_sum

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)        # b,2,h,w
        x_out = self.spatial(x_compress)     # b,1,h,w
        # scale = F.sigmoid(x_out) # broadcasting
        # return x * scale
        return x_out

class JointCBAM(nn.Module):
    def __init__(self, gate_channels, size, reduction_ratio=16, pool_types=['avg', 'max', 'min']):
        super(JointCBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels + size*size, (gate_channels + size*size) // reduction_ratio),
            nn.ReLU(),
            nn.Dropout(0.3),  ##### 추가
            nn.Linear((gate_channels + size*size) // reduction_ratio, gate_channels + size*size)
        )
    def forward(self, x):

        B, C, H, W = x.size()

        x_ch = self.ChannelGate(x)
        x_sp = self.SpatialGate(x)

        x_ch_falt = x_ch.view(x_ch.size(0), -1)  # (B, C)
        x_sp_flat = x_sp.view(x_sp.size(0), -1)  # (B, H*W)
        interaction_input = torch.cat([x_ch_falt, x_sp_flat], dim=1)  # (B, C + H*W)

        interaction_out = self.mlp(interaction_input)  # (B, C + H*W)
        ch_attn, sp_attn = torch.split(interaction_out, [C, H*W], dim=1)

        ch_attn = torch.sigmoid(ch_attn).view(B, C, 1, 1)       # Channel weight
        sp_attn = torch.sigmoid(sp_attn).view(B, 1, H, W)       # Spatial weight

        out1 = x * ch_attn     # Channel-wise weighting
        out2 = x * sp_attn     # Spatial-wise weighting

        out = out1 + out2 

        return out