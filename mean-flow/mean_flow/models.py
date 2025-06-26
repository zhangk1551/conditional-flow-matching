import torch
import torch.nn as nn

from torchcfm.models.unet.nn import timestep_embedding, linear
from torchcfm.models.unet.unet import UNetModel


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


class UNet(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        time_embed_out_dim = self.time_embed[-1].out_features
        self.combined_time_embed = linear(time_embed_out_dim * 2, time_embed_out_dim)

    def forward(self, x, r, t, y=None):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        r_emb = self.time_embed(timestep_embedding(r, self.model_channels))
        t_emb = self.time_embed(timestep_embedding(t, self.model_channels))
        emb = self.combined_time_embed(torch.cat([r_emb, t_emb], dim=-1))

#        if self.num_classes is not None:
#            assert y.shape == (x.shape[0],)
#            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
