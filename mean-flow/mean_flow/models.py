import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MLP(nn.Module):
    def __init__(self, dim, out_dim=None, w=64):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 2, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, r, t):
        x = torch.cat([x, r[:, None], t[:, None]], dim=-1)
        return self.net(x)
