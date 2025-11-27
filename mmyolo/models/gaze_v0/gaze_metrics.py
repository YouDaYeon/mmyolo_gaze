import torch
import torch.nn.functional as F
import math
from typing import Dict, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmdet.datasets.api_wrappers import COCO
from mmengine.fileio import get_local_path
from mmyolo.registry import METRICS
from mmengine.evaluator.metric import _to_cpu
from mmengine.dist import (collect_results,is_main_process)

@METRICS.register_module()
class AngularError(BaseMetric):
    def __init__(self,
                 ann_file: Optional[str] = None,
                 outfile_prefix: Optional[str] = None,
                 backend_args: dict = None):
        super().__init__()

        # self.results = []
        self.results_360 = []
        self.results_90 = []
        self.results_20 = []
        self.results_other = []
        self.outfile_prefix = outfile_prefix
        self.backend_args = backend_args

        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)

        else:
            self._coco_api = None


    def process(self, data_batch: dict, data_samples: Sequence[dict]):
        # 데이터 배치와 샘플에서 예측(pred)과 목표(target) 값을 추출
        # results에 배치 별 pred, gt 함께 저장
        # data_batch 사용 x, data_samples gt, pred 값 존재 -> 배치만큼 for문 돌면서 results 에 (gt,pred) 값 같이 저장

        for data_sample in data_samples: 
            
            ######################### front 180 or front facing (20도)
            device = data_sample['gt_instances']['gazes'][0].device 
            front_gaze = torch.tensor([0.,0.,-1.]).to(device)
            check_gt = data_sample['gt_instances']['gazes'][0]
            check_gt = check_gt.view(-1,3,1) 
            front_gaze = front_gaze.view(-1,1,3)
            output_dot = torch.bmm(front_gaze,check_gt) # tensor([[[0.9855]]], device='cuda:0') shape = 1,1,1
            output_dot = output_dot.view(-1) # tensor([0.9855], device='cuda:0') shape = 1
            output_dot = torch.acos(output_dot) # tensor([1.3996], device='cuda:0') shape = 1
            output_dot = output_dot.data # tensor([1.3996], device='cuda:0')
            check_angular = 180*torch.mean(output_dot)/math.pi
            
            ### 360, 90, 20 각도로 MAE계산
            for threshold in [360, 90, 20]:
 
                if check_angular <= threshold:
                #########################

                    result = dict()
                    pred = data_sample['pred_instances']
                    result['img_id'] = data_sample['img_id']
                    # result['pred_gaze'] = pred['gazes'] #.cpu().numpy()                   ####### gazes로 수정
                    
                    ### metric 수정 후 pred['gazes']에 첫번째 값만 가져오도록 하였음.
                    if pred['gazes'].dim() == 1:
                        result['pred_gaze'] = pred['gazes']
                    else:
                        result['pred_gaze'] = pred['gazes'][0] #.cpu().numpy()

                    gt = dict()
                    # gt['gt_gaze'] = torch.tensor([[0.5763,0.06721,-0.8144]]).to(device='cuda')
                    gt['gt_gaze'] = data_sample['gt_instances']['gazes'][0] # dataloader 수정 후 gaze gt 가져오는 거 수정 필요

                    if self._coco_api is None:
                        # TODO: Need to refactor to support LoadAnnotations
                        assert 'instances' in data_sample, \
                            'ground truth is required for evaluation when ' \
                            '`ann_file` is not provided'
                        gt['anns'] = data_sample['instances']

                    # self.results.append((gt, result))
                    ### 임계값에 따라 다른 변수에 저장
                    if threshold == 360:
                        self.results_360.append((gt, result))
                        if check_angular > 90:
                            self.results_other.append((gt, result))
                    elif threshold == 90:
                        self.results_90.append((gt, result))
                    elif threshold == 20:
                        self.results_20.append((gt, result))



    def evaluate(self, size: int) -> Dict:
        # 결과 수집 및 처리
        def collect_and_compute(results_list):
            if self.collect_device == 'cpu':
                results = collect_results(
                    results_list,
                    size,
                    self.collect_device,
                    tmpdir=self.collect_dir)
            else:
                results = collect_results(results_list, size, self.collect_device)

            if is_main_process():
                results = _to_cpu(results)
                return self.compute_metrics(results)

        # 각 임계값별 메트릭 계산
        gaze_result_360 = collect_and_compute(self.results_360)    # 25969 (testset)
        gaze_result_90 = collect_and_compute(self.results_90)      # 20322 (testset)
        gaze_result_20 = collect_and_compute(self.results_20)      # 3995  (testset)
        gaze_result_other = collect_and_compute(self.results_other)  # 5647 (testset)
        # 결과 초기화
        self.results_360 = []
        self.results_90 = []
        self.results_20 = []
        self.results_other = []
        # 결과 반환
        return {
            'MAE_360': float(gaze_result_360['mean_angular_error']),
            'MAE_90': float(gaze_result_90['mean_angular_error']),
            'MAE_20': float(gaze_result_20['mean_angular_error']),
            'MAE_other': float(gaze_result_other['mean_angular_error'])
        }

    def compute_metrics(self, results) -> Dict[str, float]: 
        
        gaze_result = dict()
        if results:
            for batch_c in range(len(results)):                 # 17038

                if batch_c == 0:
                    target_stack = results[batch_c][0]['gt_gaze'].unsqueeze(0)  # 차원 추가
                    pred_stack = results[batch_c][1]['pred_gaze']
                else:
                    target_tensor = results[batch_c][0]['gt_gaze'].unsqueeze(0)  # 차원 추가
                    pred_tensor = results[batch_c][1]['pred_gaze']

                    target_stack = torch.cat((target_stack,target_tensor),dim=0)
                    pred_stack = torch.cat((pred_stack,pred_tensor),dim=0)
        
            target = target_stack.unsqueeze(1)
            pred = pred_stack.unsqueeze(1)

            device = pred.device                     # pitch yaw 할 때 추가함.
            target = target.to(device)
            
            # 정규화. 두 벡터 각도 차이 계산할 때 단위벡터로 변환해야 함. 벡터 크기를 1로 만들어 방향만 남도록.
            # target 은 정규화 되어있음.
            # #### gaze_head에서 no_normalize일 때 pred값을 noramlize하는 코드 추가.
            # pred = pred.view(-1, 3)
            # pred = F.normalize(pred, p=2, dim=1)

            pred = pred.view(-1,3,1) 
            target = target.view(-1,1,3)
            output_dot = torch.bmm(target,pred).view(-1) # 배치 행렬 곱셈 = 내적. 단위벡터가 아니면 내적 값이 -1~1 범위를 벗어날 수 있음.
            output_dot = torch.clamp(output_dot, -1.0 + 1e-6, 1.0 - 1e-6) # 내적 결과가 -1 ~ 1을 넘길 수 없게.
            output_dot = torch.acos(output_dot) # 각도(라디안 단위)  입력값의 범위가 -1~1. 그렇지 않으면 nan값 발생.
            output_dot = output_dot.data

            # ### MAE360 계산만.
            # # 각 이미지에 대한 각도 오차 log
            # gaze_error = 180*output_dot/math.pi
            # # gaze_error 값과 batch_c를 튜플로 묶어서 리스트로 만들기
            # gaze_error_with_index = [(gaze_error[batch_c], batch_c) for batch_c in range(len(results))]
            # # gaze_error를 큰 순서대로 정렬
            # gaze_error_with_index.sort(reverse=True, key=lambda x: x[0])
            # with open('work_dirs/yolov8_s_fast_1xb12-40e_gaze_v0/20250902_PositionMap_Eye/gaze_error_log.txt', 'w') as f:
            #     for error, batch_c in gaze_error_with_index:
            #         f.write(f"Image {batch_c+1} MAE: {error:.4f} degrees pred: {pred[batch_c]} target: {target[batch_c]}\n")

            gaze_result['mean_angular_error']  = 180*torch.mean(output_dot)/math.pi # radian -> degree 로 변환하여 평균 각도 오차 계산
        else:
            gaze_result['mean_angular_error'] = 0.06
        return gaze_result