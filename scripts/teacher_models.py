"""고성능 교사 모델 (Retinexformer). 학습용 교사이며 배포·선택 후보가 아니다.

학생 모델과 같은 방식으로 쓰기 위해 출력을 튜플로 감싼다: model(x)[-1]이 보정 이미지.
입력 크기가 8의 배수가 아니면 reflect padding 후 원래 크기로 자른다.
"""
import importlib.util
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ARCH_FILE = ROOT / "vendor_retinexformer" / "RetinexFormer_arch.py"

# 원 저장소 Options/RetinexFormer_LOL_v2_real.yml 의 network_g 설정
TEACHER_CONFIGS = {
    "retinexformer": {"in_channels": 3, "out_channels": 3, "n_feat": 40, "stage": 1, "num_blocks": [1, 2, 2]},
}

_ARCH = None


def _arch():
    global _ARCH
    if _ARCH is None:
        spec = importlib.util.spec_from_file_location("retinexformer_arch", ARCH_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ARCH = module
    return _ARCH


class PaddedTeacher(nn.Module):
    def __init__(self, net, multiple=8):
        super().__init__()
        self.net = net
        self.multiple = multiple

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % self.multiple, (-w) % self.multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        y = self.net(x)[..., :h, :w]
        return (y,)


def build_teacher(name):
    if name not in TEACHER_CONFIGS:
        raise ValueError(f"Unknown teacher {name!r}; choose from {sorted(TEACHER_CONFIGS)}")
    return PaddedTeacher(_arch().RetinexFormer(**TEACHER_CONFIGS[name]))


def is_teacher(name):
    return name in TEACHER_CONFIGS
