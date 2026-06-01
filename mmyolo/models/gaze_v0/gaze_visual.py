from mmyolo.registry import VISUALIZERS

from mmdet.visualization import DetLocalVisualizer

from typing import Dict, List, Optional, Tuple, Union
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmengine.dist import master_only
import numpy as np
import cv2
import mmcv
import torch
import copy
from mmdet.visualization.palette import _get_adaptive_scales, get_palette

import math


############### bbox 시각화 ###############
def drawBoxes(boxes, image, id):

    # torch.Tensor 또는 numpy array 형태라면 numpy 배열로 변환해야 함
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    startX = boxes[0]
    startY = boxes[1]
    endX = boxes[2]
    endY = boxes[3]

    if id==0:
        cv2.rectangle(image, (int(startX), int(startY)), (int(endX), int(endY)), (255,0,0), 2) # 파랑 -> face
    if id==1:
        cv2.rectangle(image, (int(startX), int(startY)), (int(endX), int(endY)), (0,255,0), 1) # 라임 -> eye_left, eye_right

@VISUALIZERS.register_module()
class DetLocalVisualizerGaze(DetLocalVisualizer):
    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.line_width = line_width
        self.alpha = alpha
        # Set default value. When calling
        # `DetLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)

        gt_img_data = None
        pred_img_data = None

        ##################에러 발생 구간#################

        if data_sample is None:
            return data_sample
            
        try:
            # 각 필드별로 개별적으로 CPU로 이동
            for key in data_sample.keys():
                field = getattr(data_sample, key)
                
                # InstanceData 객체인 경우
                if hasattr(field, 'keys') and hasattr(field, 'to'):
                    try:
                        # InstanceData 자체를 CPU로 이동 시도
                        setattr(data_sample, key, field.to('cpu'))
                    except AssertionError:
                        # InstanceData 내부 필드들을 개별 처리
                        for sub_key in field.keys():
                            sub_field = getattr(field, sub_key)
                            if hasattr(sub_field, 'to'):
                                try:
                                    setattr(field, sub_key, sub_field.to('cpu'))
                                except Exception as e:
                                    print(f"Warning: Failed to move {key}.{sub_key} to CPU: {e}")
                                    # 빈 텐서인 경우 건너뛰기
                                    continue
                
                # 일반 텐서인 경우
                elif hasattr(field, 'to'):
                    try:
                        setattr(data_sample, key, field.to('cpu'))
                    except Exception as e:
                        print(f"Warning: Failed to move {key} to CPU: {e}")
                        
        except Exception as e:
            print(f"Warning: Failed to move data_sample to CPU: {e}")

        if draw_gt and data_sample is not None:
            gt_img_data = copy.deepcopy(image)                 
            if 'gt_instances' in data_sample:
                gt_instances = data_sample.gt_instances
                gt_img_data = self._draw_instances(gt_img_data, gt_instances, color = (230,253,11)) # 한 사진에 같이 시각화했을 때는 빨강 gt 이었는데 노랑으로 변경.
            

        if draw_pred and data_sample is not None:
            pred_img_data = copy.deepcopy(image)                        
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                # pred_instances = pred_instances[
                #     pred_instances.scores > pred_score_thr]
                pred_img_data = self._draw_instances(pred_img_data, pred_instances, color = (230,253,11)) # 노랑 pred
        
        ############## bbox 시각화 ###############

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

        cv2.imwrite('gaze_img.jpg', drawn_img)
        
        return drawn_img    
    
    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'], color) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        # image = image.copy()                        ### bbox 시각화 시

        if 'gazes' in instances:
            # bbox가 비어있으면 gaze가 torch.Size([3])임.
            if instances.bboxes.numel() > 0:
                gaze = instances.gazes[0]
            else:
                gaze = instances.gazes

            h, w = image.shape[:2]
            image_center = (w // 2, h // 2)

            # 화살표 길이와 두께 설정
            l = min(w, h) // 2
            gaze_len = l*1.0
            thick = 2
            # Gaze 화살표 그리기 arrowedLine(im, 출발좌표(x,y), 종료좌표(x,y), color, thickness)
            cv2.arrowedLine(image, (image_center[0],image_center[1]),
                            (int(image_center[0]-gaze_len*gaze[0]),int(image_center[1]-gaze_len*gaze[1])),
                            color,thickness=thick)
            # cv2.arrowedLine(image, (image_center[0],130),
            #                 (int(image_center[0]-gaze_len*gaze[0]),int(130-gaze_len*gaze[1])),
            #                 color,thickness=thick)
            
            # # gaze 좌표 print
            # print("gaze: ", gaze)
            # print("yaw_pitch", vector_to_yaw_pitch(gaze))
            # print("yaw_pitch_angular", compute_yaw_angular(vector_to_yaw_pitch(gaze)))

        ############### bbox 시각화 ###############
        if 'bboxes' in instances and instances.bboxes.sum() > 0:
            bboxes = instances.bboxes
            labels = instances.labels

            for label, bbox in zip(labels, bboxes):
                if label in (0, 1):
                    drawBoxes(bbox, image, label)

        cv2.imwrite('gaze_img.jpg', image)

        return image
    
#################################################
# bbox와 gaze 시각화 #
#################################################

# def draw_bbox_and_gaze(image: np.ndarray, eye_2d: np.ndarray, eye_3d: np.ndarray, gaze: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
def draw_bbox_and_gaze(image: np.ndarray, head_bbox: np.ndarray, face_bbox: np.ndarray, eye_bbox1: np.ndarray, eye_bbox2: np.ndarray, gaze: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
    """Draw bbox and gaze vector on image.
    
    Args:
        image (np.ndarray): Input image
        bbox (np.ndarray): Bounding box coordinates [x1, y1, x2, y2]
        gaze (np.ndarray): Gaze vector [x, y, z]
        save_path (str, optional): Path to save the drawn image. Defaults to None.
        
    Returns:
        np.ndarray: Image with drawn bbox and gaze vector
    """
    # Draw bbox
    cv2.rectangle(image, 
                 (int(head_bbox[0]), int(head_bbox[1])), 
                 (int(head_bbox[2]+head_bbox[0]), int(head_bbox[3]+head_bbox[1])), 
                 (255,0,0), 2)  # Green color for bbox

    cv2.rectangle(image, 
                 (int(face_bbox[0]), int(face_bbox[1])), 
                 (int(face_bbox[2]+face_bbox[0]), int(face_bbox[3]+face_bbox[1])), 
                 (0,0,255), 2)  # Blue color for bbox
    
    cv2.rectangle(image, 
                 (int(eye_bbox1[0]), int(eye_bbox1[1])), 
                 (int(eye_bbox1[2]+eye_bbox1[0]), int(eye_bbox1[3]+eye_bbox1[1])), 
                 (0,255,0), 2)  # Green color for bbox
    
    cv2.rectangle(image, 
                 (int(eye_bbox2[0]), int(eye_bbox2[1])), 
                 (int(eye_bbox2[2]+eye_bbox2[0]), int(eye_bbox2[3]+eye_bbox2[1])), 
                 (0,255,0), 2)  # Green color for bbox
    
    # Calculate center point for gaze arrow
    h, w = image.shape[:2]
    center_x = w // 2
    center_y = h // 2
    
    # Draw gaze vector
    arrow_length = min(w, h) // 2
    end_x = int(center_x - arrow_length * gaze[0])
    end_y = int(center_y - arrow_length * gaze[1])
    
    cv2.arrowedLine(image, 
                    (center_x, center_y-250),
                    (end_x, end_y-250), 
                    (0,255,255),  # Yellow color for gaze
                    thickness=2)
    
    # # Draw eye 2d
    # eye_2d = eye_2d * np.array([w, h])

    # cv2.circle(image, 
    #            (int(eye_2d[0]), int(eye_2d[1])), 
    #            5,  # 반지름
    #            (0, 255, 0),  # 녹색
    #            -1)  # 채워진 원
    
    # eye_3d = eye_3d * np.array([w, h, 1])
    # cv2.circle(image, 
    #            (int(eye_3d[0]), int(eye_3d[1])), 
    #            5,  # 반지름
    #            (0, 0, 255),  # 빨강색
    #            -1)  # 채워진 원
    if save_path is not None:
        cv2.imwrite(save_path, image)
        
    return image

def vector_to_yaw_pitch(x):
    output = np.zeros((2))
    output[0] = np.arctan2(x[0], -x[2])
    output[1] = np.arcsin(x[1])
    return output

# def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0, save_path=None):
#     """Draw gaze angle on given image with a given eye positions."""
#     image_out = image_in
#     (h, w) = image_in.shape[:2]
#     length = c
#     pos = (int(a+c / 2.0), int(b+d / 2.0))
#     if len(image_out.shape) == 2 or image_out.shape[2] == 1:
#         image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
#     dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])

#     dy = -length * np.sin(pitchyaw[1])
#     cv2.arrowedLine(image_out, 
#                     (int(pos[0]), int(pos[1])),
#                     (int(pos[0] + dx), int(pos[1] + dy)), 
#                     color,
#                     thickness, 
#                     cv2.LINE_AA, 
#                     tipLength=0.18)
#     if save_path is not None:
#         cv2.imwrite(save_path, image_out)
#     return image_out

def compute_yaw_angular(target):

    output_yaw = 180 * (target[0]) / math.pi
    
    return output_yaw

# image = cv2.imread('/mnt/terror/gaze360_dataset/original/imgs/rec_078/body/000035/000673.jpg')

# # Example bbox and gaze coordinates
# head_bbox = np.array([46,158,355,356])  # [x1, y1, x2, y2]
# face_bbox = np.array([177,247,119,176])  # [x1, y1, x2, y2]
# eye_bbox1 = np.array([276,309,11,13])  # [x1, y1, x2, y2]
# eye_bbox2 = np.array([239,311,20,14])  # [x1, y1, x2, y2]
# gaze = np.array([-0.6038025407684535,-0.17154828552366802,-0.7784559573254289])  # [x, y, z]
# # yaw_pitch = vector_to_yaw_pitch(gaze)
# # print("yaw: ", yaw_pitch[0], "pitch: ", yaw_pitch[1])
# # yaw_pitch_angular = compute_yaw_angular(yaw_pitch)
# # print("yaw_angular: ", yaw_pitch_angular)

# # yaw_pitch = vector_to_yaw_pitch(gaze)

# # output_yaw = compute_yaw_angular(yaw_pitch)
# # print("yaw: ", yaw_pitch[0], "pitch: ", yaw_pitch[1])
# # print("output_yaw: ", output_yaw)

# # result_image = draw_gaze(
# #     a=face_bbox[0],
# #     b=face_bbox[1],
# #     c=face_bbox[2],
# #     d=face_bbox[3],
# #     image_in=image,
# #     pitchyaw=yaw_pitch,
# #     save_path='demo/bboxgaze_visual/test.jpg'
# # )

# # eye_2d = np.array([0.32571266, 0.25452647])
# # eye_3d = np.array([0.713, -0.185,  1.663])

# # Draw bbox and gaze
# result_image = draw_bbox_and_gaze(
#     image=image,
#     head_bbox=head_bbox,
#     face_bbox=face_bbox,
#     eye_bbox1=eye_bbox1,
#     eye_bbox2=eye_bbox2,
#     gaze=gaze,
#     save_path='DY_TEST/bboxgaze_visual.jpg'
# )

