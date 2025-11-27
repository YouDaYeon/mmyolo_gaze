import torch
import torch.nn as nn

import numpy as np
from PIL import Image

from mmyolo.registry import MODELS

class OhemCELoss(nn.Module):
    def __init__(self, thresh: float, min_kept: int) -> None:
        super().__init__()
        self.thresh = torch.log(torch.tensor(1 / thresh, dtype=torch.float))
        self.min_kept = min_kept
        self.criteria = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.min_kept] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.min_kept]
        return torch.mean(loss)
    
# ### 추가 ###   
# class Normalize:
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, image, target):
#         image = F.normalize(image, mean=self.mean, std=self.std)
#         return image, target


# class ToTensor:
#     def __call__(self, image, target):
#         image = F.to_tensor(image)
#         return image, target


# class Compose:
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, image, target):
#         for transform in self.transforms:
#             image, target = transform(image, target)
#         return image, target
# ###########   

@MODELS.register_module()
class OhemLossWrapper:
    def __init__(self, thresh: float, min_kept: int, loss_weight: float) -> None:
        self.loss = OhemCELoss(thresh=thresh, min_kept=min_kept)
        self.loss_weight = loss_weight

    def __call__(self, output, labels):
        
        # 추가) labels 전처리
        new_labels = []
        for label_path in labels:
            label = Image.open(label_path).convert('P')
            label = np.array(label).astype(np.int64)
            new_labels.append(label)
        new_labels = torch.stack([torch.from_numpy(label) for label in new_labels]).to(output[0].device)

        out, out16, out32 = output

        loss1 = self.loss(out, new_labels)
        loss2 = self.loss(out16, new_labels)
        loss3 = self.loss(out32, new_labels)

        loss = loss1 + loss2 + loss3
        return loss * self.loss_weight                     # loss1: 2.8760 loss2:3.0288 loss3: 2.8494
