from turtle import heading
import numpy as np
import torch
from torch import nn

class GaussianHeatmap(nn.Module):
    def __init__(
            self,
            sigma = 3
    ):
        """
        size: (H, W) = 히트맵 크기
        center: (x, y) = gt 시선 좌표 
        sigma: 가우시안 표준 편차
        """
        super().__init__()

        self.sigma = sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, size, gaze):
        B = gaze.shape[0]
        H, W = size

        center = min(H, W) // 2
        gaze_len = center*1.0
        x_re_gaze = torch.round(center-gaze_len*gaze[:, 0]) # 0~W 범위
        y_re_gaze = torch.round(center-gaze_len*gaze[:, 1]) # 0~H 범위
        # z_gaze = gaze[:, 2]  # z 값 추출 (-1 ~ 1)

        # 2D meshgrid 생성 (PyTorch 사용) 
        y, x = torch.meshgrid(torch.arange(H, device=self.device),                   # 모든 행이 동일
                              torch.arange(W, device=self.device), indexing='ij')    # 모든 열이 동일
        x = x.unsqueeze(0).expand(B, -1, -1)
        y = y.unsqueeze(0).expand(B, -1, -1)

        # 배치별 gaze 좌표 추출
        x_g = x_re_gaze.view(B, 1, 1).expand(B, H, W)
        y_g = y_re_gaze.view(B, 1, 1).expand(B, H, W)

        # # z 값을 활용한 가중치 설정 (0~1 범위로 변환)
        # weight = (z_gaze + 1) / 2 + 1e-5   # -1~1 -> 0~1, 0 방지
        # weight = weight.view(B, 1, 1).expand(B, H, W).to(self.device)
        # z 값을 활용한 가우시안 sigma 조절 (1 - z 사용하여 정면일수록 sharp)
        # sigma = self.sigma * (2 + z_gaze).view(B, 1, 1).expand(B, H, W)

        # 2D 가우시안 분포 계산
        # 각 픽셀과 시선 좌표와의 거리 계산. 이를 가우시안 함수로 변환하여 heatmap 생성.
        heatmap = torch.exp(-((x - x_g) ** 2 + (y - y_g) ** 2) / (2 * self.sigma ** 2))   #(0, 1] 치역 범위
        # heatmap = torch.exp(-((x - x_g) ** 2 + (y - y_g) ** 2) / (2 * sigma ** 2)) 
        # heatmap = heatmap * weight
        # Numpy → PyTorch Tensor 변환
        if not torch.is_tensor(heatmap):
            heatmap = torch.tensor(heatmap, dtype=torch.float32)
        else:
            heatmap = heatmap.clone().detach().float()

        # # Normalize (0~255 범위로 변환하여 OpenCV에서 사용 가능하도록)
        # heatmap = heatmap / heatmap.max() * 255
        # heatmap = heatmap.cpu().numpy().astype(np.uint8)  # OpenCV 호환을 위해 uint8 변환

        return heatmap

# heatmap 시각화
import os, cv2
import json
if __name__ == "__main__":
    data_root = '/mnt/terror/gaze360_dataset/gaze_annotations/'
    image_root = '/mnt/terror/gaze360_dataset/original/imgs/'
    test_file_path = os.path.join(data_root, "gaze_test_all_bbox.json")

    with open(test_file_path, 'r') as f:
        test_label = json.load(f)

    test_images = test_label['images']
    test_3Dgaze = test_label['3Dgaze']

    gaze_coordinates = [gaze['coordinate'] for gaze in test_3Dgaze]
    image_id = [gaze['image_id'] for gaze in test_3Dgaze]
    size = [(image['height'], image['width']) for image in test_images]
    file_name = [image['file_name'] for image in test_images]

    gaze_x = [gaze[0] for gaze in gaze_coordinates]
    gaze_y = [gaze[1] for gaze in gaze_coordinates]
    gaze_z = [gaze[2] for gaze in gaze_coordinates]
    gaze = torch.tensor([gaze_x, gaze_y, gaze_z]).T  # shape: (B, 3)

    gh = GaussianHeatmap()
    for i, (size, gaze) in enumerate(zip(size, gaze)):
        gaze = gaze.expand(1, -1).to(torch.device("cuda"))
        # heatmap = gh(size, gaze)
        heatmap = gh((64, 64), gaze)
        
        # Load original image
        img_path = os.path.join(image_root, file_name[i])
        img = cv2.imread(img_path)

        # Convert heatmap to uint8 and apply colormap
        # heatmap_vis = (heatmap.cpu().numpy() * 255).astype(np.uint8).squeeze()

        heatmap_np = heatmap.cpu().numpy().squeeze()
        heatmap_normalized = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())
        heatmap_vis = (heatmap_normalized * 255).astype(np.uint8)

        heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap_colored, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Blend original image with heatmap
        alpha = 0.7  # Transparency factor
        overlay = cv2.addWeighted(img, 1-alpha, heatmap_resized, alpha, 0)
        overlay = overlay.astype(np.uint8)

        # Concatenate original and overlay images side by side
        combined = np.concatenate((img, overlay), axis=1)

        # Save the result
        save_dir = 'DY_TEST/heatmap2/simga3'
        save_path = os.path.join(save_dir, file_name[i])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, combined)