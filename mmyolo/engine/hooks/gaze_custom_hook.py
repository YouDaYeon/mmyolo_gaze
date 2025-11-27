from mmengine.hooks import Hook
from mmyolo.registry import HOOKS


@HOOKS.register_module()
class EpochUpdateHook(Hook):
    def before_train_epoch(self, runner):
        # model의 on_train_epoch_start 호출
        if hasattr(runner.model, 'module'):
            runner.model.module.on_train_epoch_start(runner.epoch)
        else:
            runner.model.on_train_epoch_start(runner.epoch)
