import torch
import pytorch_lightning as pl
from torchcfm.optimal_transport import OTPlanSampler


class MeanFlow(pl.LightningModule):
  def __init__(self, mean_flow_config, net):
    super().__init__()
    self.net = net
    self.config = mean_flow_config
    if self.config.use_optimal_transport:
        self.ot_sampler = OTPlanSampler(method="exact")

  def sample_conditional_pt(self, e, x, t, sigma):
    t = t.reshape(-1, *([1] * (x.dim() - 1)))
    mu_t = t * e + (1 - t) * x
    epsilon = torch.randn_like(x)
    return mu_t + sigma * epsilon

  def compute_conditional_vector_field(self, e, x):
    return e - x

  def compute_target(self, v, r, t, dudt):
    r = r.reshape(-1, *([1] * (v.dim() - 1)))
    t = t.reshape(-1, *([1] * (v.dim() - 1)))
    u_tgt = v - (t - r) * dudt
    return u_tgt

  def sample(self, e, step=1):
    device = next(self.net.parameters()).device
    if step == 1:
        r = torch.zeros(e.shape[0], device=device)
        t = torch.ones(e.shape[0], device=device)
        return e - self.net(e, r, t)

    ts = torch.linspace(1, 0, step + 1, device=device)
    x = e.clone()

    for i in range(step):
        t = ts[i].expand(x.shape[0])
        r = ts[i + 1].expand(x.shape[0])
        u = self.net(x, r, t)
        dt = ts[i] - ts[i + 1]
        x = x - dt * u
    return x


  def sample_r_t(self, batch_size, device, dist='logit_normal', mu=-0.4, sigma=1.0):
    if dist == 'uniform':
        rt = torch.rand(batch_size, 2, device=device)
    elif dist == 'logit_normal':
        normal_sample = torch.randn(batch_size, 2, device=device) * sigma + mu
        rt = torch.sigmoid(normal_sample)
    else:
        raise ValueError("dist must be 'uniform' or 'logit_normal'")

    r, _ = rt.min(dim=-1)
    t, _ = rt.max(dim=-1)

    equal_ratio = 1 - self.config.unequal_ratio
    if equal_ratio > 0.0:
        num_equal = int(batch_size * equal_ratio)
        indices = torch.randperm(batch_size)[:num_equal]
        t[indices] = r[indices]

    return r, t


  def adaptive_weighted_loss(self, error, p=1, c=1e-3):
    L = torch.sum(error ** 2, dim=1)
    w = (1.0 / (L + c) ** p).detach()
    loss = (w * L).mean()
    return loss


  def loss(self, e, x):

    if self.config.use_optimal_transport:
        e, x = self.ot_sampler.sample_plan(e, x)

    r, t = self.sample_r_t(e.shape[0], device=e.device)

    z = self.sample_conditional_pt(e, x, t, sigma=0.01)
    v = self.compute_conditional_vector_field(e, x)

    _, dudt = torch.autograd.functional.jvp(self.net, (z, r, t), (v, torch.zeros(r.shape, device=e.device), torch.ones(t.shape, device=e.device)))

    u = self.net(z, r, t)
    u_tgt = self.compute_target(v, r, t, dudt)

    loss = self.adaptive_weighted_loss(u - u_tgt.detach())
    return loss
