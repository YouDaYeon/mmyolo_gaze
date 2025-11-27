import torch
import torch.nn as nn
import torch.nn.functional as F

class SPCModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(SPCModule, self).__init__()
        self.branches = nn.ModuleList()
        self.kernel_sizes = kernel_sizes
        branch_out_channels = out_channels // len(kernel_sizes)

        for k in kernel_sizes:
            group = 2 ** ((k - 1) // 2)
            self.branches.append(
                nn.Conv2d(
                    in_channels, branch_out_channels,
                    kernel_size=k, padding=k//2, groups=group, bias=False
                )
            )

    def forward(self, x):
        return [branch(x) for branch in self.branches]  # list of feature maps


class SEWeightModule(nn.Module):
    def __init__(self, in_channels_list, reduction=16):
        super(SEWeightModule, self).__init__()
        self.se_blocks = nn.ModuleList()
        for c in in_channels_list:
            self.se_blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(c, c // reduction, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c // reduction, 1, kernel_size=1, bias=False),  # Single value per branch
                )
            )

    def forward(self, feats):  # feats: list of feature maps
        weights = []
        for i, feat in enumerate(feats):
            w = self.se_blocks[i](feat)  # (B, 1, 1, 1)
            weights.append(w.squeeze(-1).squeeze(-1))  # (B, 1) → scalar per batch
        weights = torch.cat(weights, dim=1)  # (B, S)
        return weights  # no softmax yet


class PSAModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], reduction=16):
        super(PSAModule, self).__init__()
        self.spc = SPCModule(in_channels, out_channels, kernel_sizes)
        branch_out_channels = out_channels // len(kernel_sizes)
        self.se = SEWeightModule(
            [branch_out_channels] * len(kernel_sizes),
            reduction=reduction
        )

    def forward(self, x):
        # Step 1: Multiscale feature maps
        feats = self.spc(x)  # [H0, H1, ..., Hs-1] channel 192 -> 64(k=3), 64(k=5), 64(k=7)

        # Step 2: Raw attention weights
        attn_logits = self.se(feats)  # shape: (B, S)

        # Step 3: Softmax over scale dimension
        attn_weights = F.softmax(attn_logits, dim=1)  # shape: (B, S)

        # Step 4: Apply weight to each feature map
        weighted_feats = []
        for i, feat in enumerate(feats):
            weight = attn_weights[:, i].view(-1, 1, 1, 1)  # (B,1,1,1)
            weighted_feats.append(feat * weight)

        # Step 5: Concatenate along channel dimension
        out = torch.cat(weighted_feats, dim=1)
        return out
