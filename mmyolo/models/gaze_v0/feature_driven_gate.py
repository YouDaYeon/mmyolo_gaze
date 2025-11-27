import torch
import torch.nn as nn

class GazeFusionDynamicGate(nn.Module):
    def __init__(self, in_channels, proj_dim=96, scorer_hidden=64):
        super().__init__()

        # 레벨별 채널 정규화(projection)
        self.proj3 = nn.Sequential(nn.Linear(in_channels[0], proj_dim), nn.ReLU(inplace=True))
        self.proj4 = nn.Sequential(nn.Linear(in_channels[1], proj_dim), nn.ReLU(inplace=True))
        self.proj5 = nn.Sequential(nn.Linear(in_channels[2], proj_dim), nn.ReLU(inplace=True))

        # 공유 scorer: 동일 기준으로 레벨 간 비교
        self.scorer = nn.Sequential(
            nn.Linear(proj_dim, scorer_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(scorer_hidden, 1)
        )

    def forward(self, x, pred_p3p4p5):
        '''
        x: p3p4p5 feature map
        pred_p3p4p5: gaze 예측값
        '''
        
        p3, p4, p5 = x
        
        p3 = self.proj3(p3.view(p3.shape[0], -1))  # (B, proj_dim)
        p4 = self.proj4(p4.view(p4.shape[0], -1)) 
        p5 = self.proj5(p5.view(p5.shape[0], -1)) 

        logit3 = self.scorer(p3)  # (B, 1)
        logit4 = self.scorer(p4)
        logit5 = self.scorer(p5)
        logits = torch.cat([logit3, logit4, logit5], dim=1) # (B, 3)

        weights = torch.softmax(logits, dim=1) # (B, 3)

        # (B,3levels,3xyz) 형태로 스택
        preds = torch.stack(pred_p3p4p5, dim=1)  # (B, 3, 3)

        # 1. 가중합
        pred_xyz = (preds * weights.unsqueeze(-1)).sum(dim=1)  # (B, 3)

        # # 2. feature 하나만 선택
        # max_idx = torch.argmax(weights, dim=1) # (B,)
        # pred_xyz = preds[torch.arange(preds.shape[0]), max_idx, :] # (B, 3)

        return pred_xyz

