import cv2
import numpy as np

import torch


ATTRIBUTES = [
    'skin',
    'l_brow',
    'r_brow',
    'l_eye',
    'r_eye',
    'eye_g',
    'l_ear',
    'r_ear',
    'ear_r',
    'nose',
    'mouth',
    'u_lip',
    'l_lip',
    'neck',
    'neck_l',
    'cloth',
    'hair',
    'hat'
]

COLOR_LIST = [
    [0, 0, 0],
    [0, 0, 255],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0], 
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [255, 85, 0],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
]


def vis_parsing_maps(image, bboxes, seg_masks, save_image=False, save_dir="visualize"):
    """
    얼굴 파싱 결과를 시각화하는 함수

    Args:
        image: 원본 이미지 경로 리스트
        seg_masks: 모델이 예측한 세그멘테이션 마스크 리스트 (각 마스크는 numpy array)
        save_image: 결과 이미지 저장 여부 (default: False)
        save_dir: 저장할 디렉토리 경로 (default: "visualize")

    Returns:
        blended_images: 원본 이미지와 세그멘테이션 마스크를 블렌딩한 결과 이미지 리스트
    """
    blended_images = []
    
    for img_path, bbox, seg_mask in zip(image, bboxes, seg_masks):
        # 이미지 읽기
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 세그멘테이션 마스크를 uint8로 변환
        # seg_mask = seg_mask.squeeze().cpu().numpy().astype(np.uint8)
        if isinstance(seg_mask, torch.Tensor):
            seg_mask = seg_mask.squeeze().cpu().numpy()
        else:
            seg_mask = np.squeeze(seg_mask)
        
        # 세그멘테이션 마스크와 동일한 크기의 빈 컬러 마스크 생성 (H,W,3)
        seg_mask_color = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3))

        # 마스크에서 가장 큰 클래스 인덱스 값을 구함
        num_classes = np.max(seg_mask)

        # 각 클래스별로 미리 정의된 색상을 할당
        for class_index in range(1, num_classes + 1):
            # 현재 클래스에 해당하는 픽셀 위치 찾기
            class_pixels = np.where(seg_mask == class_index)
            # 해당 픽셀들에 클래스별 색상 할당
            seg_mask_color[class_pixels[0], class_pixels[1], :] = COLOR_LIST[class_index]

        # 컬러 마스크를 uint8로 변환
        seg_mask_color = seg_mask_color.astype(np.uint8)

        # RGB 이미지를 OpenCV의 BGR 형식으로 변환
        bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 원본 이미지와 컬러 마스크를 6:4 비율로 블렌딩
        blended_image = cv2.addWeighted(bgr_image, 0.6, seg_mask_color, 0.4, 0)
        # blended_images.append(blended_image)


        ### --- bbox 시각화 추가 ---
        # Draw bbox on image if exists
        if bbox != [0, 0, 0, 0]:
            bx, by, bx2, by2 = bbox
            cv2.rectangle(blended_image, (bx, by), (bx2, by2), (0, 255, 0), 2)

        blended_images.append(blended_image)

        # save_image가 True인 경우 결과 저장
        if save_image:
            # 파일명 추출
            img_name = '/'.join(img_path.split('/')[6:])
            base_name = img_name.split('.')[0]
            base_name = base_name.replace("/", "_")
            
            # 저장 경로 생성
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # 마스크와 블렌딩된 이미지 저장
            # cv2.imwrite(os.path.join(save_dir, f'{base_name}_mask.jpg'), seg_mask)
            cv2.imwrite(os.path.join(save_dir, f'{base_name}_blend.jpg'), blended_image, 
                       [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return blended_images
