# Pytorch 버전이 2.0 이상이라 발생하는 에러 해결을 위함.
# 체크포인트 파일을 불러올 때 state_dict 키를 포함하지 않는 구조이기 때문에 발생.
# 새로운 체크포인트 파일을 생성하여 해결.

import torch

checkpoint = torch.load('/mnt/terror/gaze360_dataset/mm_yolov8m_coco.pth')

# state_dict 키가 없으면 직접 state_dict로 사용
if 'state_dict' not in checkpoint:
    new_checkpoint = {
        'state_dict': checkpoint,
        'meta': {}
    }
    torch.save(new_checkpoint, '/mnt/terror/gaze360_dataset/mm_yolov8m_coco_state_dict.pth')
    print("체크포인트 파일이 변환되었습니다.")