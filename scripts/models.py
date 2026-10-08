"""Model zoo for the project: Mini / Max (pinned upstream code) and Max+ (project-defined).

All models return three outputs (H/4, H/2, H) so the upstream multi-level
loss applies unchanged; the last output is the enhanced image.
"""
import torch
import torch.nn as nn

from common import ROOT  # noqa: F401  (puts vendor/ on sys.path)
from model import UltraFastLiNET, ultrafast_linet_max, ultrafast_linet_mini


class Refiner(nn.Module):
    """Full-resolution RGB residual from [input, Max output] (6 -> hidden -> 3).

    The last layer starts at zero, so an untrained Max+ reproduces its Max
    branch exactly; training only adds a correction on top.
    """

    def __init__(self, hidden=13):
        super().__init__()
        self.conv1 = nn.Conv2d(6, hidden, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, 3, kernel_size=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, enhanced):
        return enhanced + self.conv2(self.act(self.conv1(torch.cat([x, enhanced], dim=1))))


class MaxPlus(nn.Module):
    """Max + small full-resolution refiner (~950 parameters in total by default)."""

    def __init__(self, hidden=13):
        super().__init__()
        self.base = ultrafast_linet_max()
        self.refiner = Refiner(hidden)

    def forward(self, x):
        d1, d2, d3 = self.base(x)
        return d1, d2, self.refiner(x, d3)


BUILDERS = {"mini": ultrafast_linet_mini, "max": ultrafast_linet_max, "maxplus": MaxPlus}


def build_model(name):
    if name not in BUILDERS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(BUILDERS)}")
    return BUILDERS[name]()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["BUILDERS", "MaxPlus", "Refiner", "UltraFastLiNET", "build_model", "count_parameters"]
