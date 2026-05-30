"""NaN/Inf 自动早停 Hook。

训练中一旦 loss 变成 nan/inf,立即抛异常终止训练,避免在已经崩坏的模型上
白烧几小时 GPU(P2 调试期间多次因 NaN 没及时发现而浪费多机算力)。

用法(config 里加):
    custom_hooks = [
        dict(type="NaNStopHook", check_grad=True, patience=0),
    ]

参数:
    check_loss: 检查 runner.outputs['log_vars']['loss'] 是否 nan/inf(默认 True)
    check_grad: 额外检查 log_vars 里的 grad_norm(默认 False;有些版本不记录)
    patience:   连续多少次 nan 才停(默认 0 = 第一次就停)。设 >0 可容忍
                偶发单步 nan(配合 fp16 dynamic loss scale 时有用)。
"""
import math

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class NaNStopHook(Hook):
    def __init__(self, check_loss=True, check_grad=False, patience=0):
        self.check_loss = bool(check_loss)
        self.check_grad = bool(check_grad)
        self.patience = int(patience)
        self._nan_streak = 0

    @staticmethod
    def _is_bad(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False
        return math.isnan(f) or math.isinf(f)

    def after_train_iter(self, runner):
        outputs = getattr(runner, "outputs", None)
        if not outputs:
            return
        log_vars = outputs.get("log_vars", {})

        bad_keys = []
        if self.check_loss:
            for k in ("loss",):
                if k in log_vars and self._is_bad(log_vars[k]):
                    bad_keys.append(k)
        if self.check_grad:
            for k in ("grad_norm",):
                if k in log_vars and self._is_bad(log_vars[k]):
                    bad_keys.append(k)

        if not bad_keys:
            self._nan_streak = 0
            return

        self._nan_streak += 1
        runner.logger.error(
            f"[NaNStopHook] detected nan/inf in {bad_keys} at iter "
            f"{runner.iter + 1} (streak={self._nan_streak}/{self.patience + 1}). "
            f"log_vars={ {k: log_vars.get(k) for k in bad_keys} }"
        )
        if self._nan_streak > self.patience:
            # 抛异常让 torchrun/launcher 立即终止所有 rank,不再白烧 GPU
            raise RuntimeError(
                f"[NaNStopHook] training loss became nan/inf at iter "
                f"{runner.iter + 1}; aborting to save GPU time. "
                f"Check [NaN-DEBUG] lines (set model.debug_nan_source=True) "
                f"to locate the source branch."
            )
