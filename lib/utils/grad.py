import torch
from torch import Tensor
import math

class NormClipper(torch.nn.Module):
    def __init__(self, initial_max_norm = 100, rel_thresh = 0.7, momentum = 0.1) -> None:
        super().__init__()
        self.initial_max_norm = initial_max_norm
        self.register_buffer('max_norm', torch.tensor(-1, dtype=torch.float))
        self.momentum = momentum
        self.scale = 1+rel_thresh
        self.last_norm = 0
        self.start = True

    def forward(self, grads, norm_type = 2):
        return self.clip(grads, norm_type)


    def clip(self, grads, norm_type = 2):
        if self.start and self.max_norm <= 0:   #check first if just start, then max_norm, to avoid unnecessary CPU-GPU sync
            new_norm, clipped = clip_norm(grads, self.initial_max_norm, norm_type=norm_type)
            self.max_norm = new_norm * self.scale
        else:
            self.start = False
            new_norm, clipped = clip_norm(grads, self.max_norm, norm_type=norm_type)
            self.max_norm = self.max_norm * (1-self.momentum) \
                + self.momentum * self.scale * new_norm.clamp_max(self.max_norm * self.scale)

        self.last_norm = new_norm
        return clipped[0] if isinstance(grads, Tensor) else clipped


def clip_norm(
        grads, max_norm, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(grads, torch.Tensor):
        grads = [grads]
    # max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == math.inf:
        norms = [p.detach().abs().max().to(device) for p in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        if len(grads) > 1:
            total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in grads]), norm_type)
        else:
            total_norm = torch.norm(grads[0].detach(), norm_type).to(device)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    clipped =[p.mul(clip_coef_clamped.to(p.device)) for p in grads]
    return total_norm, clipped
