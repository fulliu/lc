import torch
from . import rotation_conversions as rcvt
from torch import Tensor


def quaternion_rep_to_RT(quaternion_reps:Tensor):
    return _quaternion_rep_to_RT(quaternion_reps) if quaternion_reps.requires_grad else _quaternion_rep_to_RT_script(quaternion_reps)
    
def _quaternion_rep_to_RT(quaternion_reps:Tensor):
    '''
    Parameters:
    quaternion_reps: shape(*, 7)
    Returns:
    Rs: shape(*, 3, 3)
    ts: shape(*, 3)
    '''
    Rs=rcvt.quaternion_to_matrix(quaternions=quaternion_reps[...,:4])
    ts = quaternion_reps[...,4:7]
    return Rs, ts

@torch.jit.script
def _quaternion_rep_to_RT_script(quaternion_reps:Tensor):
    '''
    Parameters:
    quaternion_reps: shape(*, 7)
    Returns:
    Rs: shape(*, 3, 3)
    ts: shape(*, 3)
    '''
    Rs=rcvt.quaternion_to_matrix(quaternions=quaternion_reps[...,:4])
    ts = quaternion_reps[...,4:7]
    return Rs, ts


def RT_to_quaternion_rep(Rs:Tensor, ts:Tensor):
    '''
    Parameters:
    Rs: shape(*, 3, 3)
    ts: shape(*, 3)
    Returns:
    quaternion_reps: shape(*, 7)
    '''
    quaternions = rcvt.matrix_to_quaternion(Rs)
    return torch.cat((quaternions, ts), dim = -1)


def project_apply(cam_K, pts_3d, R = None, t = None, min_z = 0.1) -> Tensor:
    '''
    fit numpy or torch
    Args:
    cam_K: np.ndarray or Tensor, shape (*, 3, 3)
    pts_3d: np.ndarray or Tensor, shape (*, N, 3)
    R: shape(*, 3, 3)
    t: shape(*, 3) or (*, 3, 1)
    Return:
    pts_2d: np.ndarray or Tensor, shape (*, N, 2)
    '''
    if R is not None:
        pts_3d = pts_3d @ R.mT + t.squeeze(-1)[...,None,:]
    xformed = pts_3d @ cam_K.transpose(-1,-2)
    z=xformed[...,2:3]
    z=z.clamp(min=min_z) if isinstance(z, Tensor) else z.clip(min=min_z)
    return xformed[...,:2]/z


def gen_uv(shape_hw, device=None, dtype=None):
    '''
    Return: (H, W, 2)
    '''
    H,W = shape_hw[-2:]
    xs = torch.arange(0,W-0.5,device=device, dtype=dtype)
    ys = torch.arange(0,H-0.5,device=device, dtype=dtype)
    x,y = torch.meshgrid((xs,ys),indexing='xy')
    return torch.stack((x,y),dim=-1)



