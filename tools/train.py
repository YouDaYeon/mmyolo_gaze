# Copyright (c) OpenMMLab. All rights reserved.
import sys
import argparse
import logging
import os
import os.path as osp

# See tools/test.py: sys.path must include repo root so `mmyolo` resolves to `<repo>/mmyolo`.
_TOOLS_DIR = osp.dirname(osp.abspath(__file__))
_REPO_ROOT = osp.abspath(osp.join(_TOOLS_DIR, '..'))
for _p in (
        _REPO_ROOT,
        osp.join(_REPO_ROOT, 'mmyolo', 'mmdetection'),
        osp.join(_REPO_ROOT, 'mmdetection'),
):
    if osp.isdir(_p):
        sys.path.insert(0, _p)

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # ######## 모델 파라미터 계산 함수 ########
    # def calculate_model_parameters(model):

    #     total_params = sum(p.numel() for p in model.parameters())
    #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #     # # 디버깅: 파라미터 정보 출력
    #     # for name, param in model.named_parameters():
    #     #     print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

    #     return total_params, trainable_params

    # ######## 모델 파라미터 계산 ########
    # model = runner.model
    # total_params, trainable_params = calculate_model_parameters(model)

    # print_log(
    #     f"Model Parameter Info:\n"
    #     f"  Total parameters: {total_params:,}\n"
    #     f"  Trainable parameters: {trainable_params:,}"
    # )

    # # 서브모듈별 파라미터 수 확인
    # for name, module in model.named_children():
    #     module_params = sum(p.numel() for p in module.parameters())
    #     print(f"{name}: {module_params:,} parameters")

    # # 모델 구조
    # print(runner.model)

    # # 모델 로드 및 추론 시 VRAM 사용량
    # import torch
    # import torchvision.models as models
    # import time

    # torch.cuda.empty_cache()  # 캐시 비우기
    # torch.cuda.reset_peak_memory_stats()

    # # 모델 로드
    # model = runner.model

    # used_mb = torch.cuda.memory_allocated() / 1024**2        # 1024 × 1024 Bytes = 1MB
    # print(f"모델 로드 후 VRAM: {used_mb:.2f} MB")

    # # 정확하게 파라미터+버퍼 크기만 계산
    # param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    # size_all_mb = (param_size + buffer_size) / 1024**2
    # print(f"모델 파라미터+버퍼 메모리: {size_all_mb:.2f} MB")

    # torch.cuda.reset_peak_memory_stats()
    # x = torch.randn(1, 3, 320, 320).cuda()
    # torch.cuda.synchronize()                 # 이전 GPU 연산 끝날 때까지 대기
    # start = time.time()

    # with torch.no_grad():
    #     y = model(x)
    # torch.cuda.synchronize()                 # 이전 GPU 연산 끝날 때까지 대기
    # end = time.time()
    
    # peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    # print(f"추론 시 peak VRAM 사용량: {peak_mb:.2f} MB")
    # print(f"Inference time: {end - start:.4f} sec")

    # start training
    runner.train()


if __name__ == '__main__':
    main()
