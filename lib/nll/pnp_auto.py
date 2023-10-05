import torch
import functorch
from torch import Tensor
from ..transforms import rotation_conversions as rcvt
from .pnp_utils import (
    axis_angle_rotate_point_jac_near_zero,
    weighted_hess_jac,
    safe_cholesky, 
    nll_update,
    make_sure_symmetric
)

def residual_with_jac6d(
    pose_gt:Tensor, cam_K:Tensor, pts3d:Tensor, pts2d:Tensor,\
    d_aax_xyz:Tensor = None):
    """
    jac is computed w.r.t. a pose update at right side of quat_xyz around zero in axis_angle representation,
    which means we are using some kind of right perturbation method
    slightly diffenent from the right perturb in SLAM, here:
        R_{perturbed} = R_{original} @ R_{perturb},
        t_{perturbed} = t_{original} + t_{perturb}
    Args:
        pose_gt: Shape (*, 7) operating point, wijk_xyz
        cam_K  : shape (*, 3, 3), camera matrix
        pts3d  : Shape (*, N, 3), 3d points in model frame
        pts2d  : Shape (*, N, 2), measured 2d point positions
        d_aax_xyz: (Tensor | None): shape (*, 6), delta pose, useful if we want to get Hessian, values should always be zeros
    Return:
        r : Shape (*, N, 2) residual
        dr_dpose: Shape (*, N, 2, 6) jacobian of residual w.r.t. 6d pose update
    """

    xyz = pose_gt[...,4:7]
    post_rot = rcvt.quaternion_to_matrix(pose_gt[...,:4])  #(*, 3, 3)

    xyz = xyz + d_aax_xyz[...,3:6]
    xformed_before_post_rot, jac_before_rot = \
        axis_angle_rotate_point_jac_near_zero(d_aax_xyz[...,:3], pts3d)#(*, N, 3), (*, N, 3, 3)

    dxyz_daax = post_rot.unsqueeze(-3) @ jac_before_rot   #(*, 1, 3, 3) @ (*, N, 3, 3)

    xformed = xformed_before_post_rot @ post_rot.mT + xyz.unsqueeze(-2)    # (*, N, 3) = ((*, N, 3) @ (*, 3, 3) + (*, 1, 3))
    inv_z = 1.0/xformed[...,2] # (*, N)
    uv0 = xformed[...,0:2] * inv_z[...,None] # (*, N, 2)
    eye_blk = torch.eye(2, dtype=uv0.dtype, device=uv0.device).expand(uv0.shape[:-1]+(-1,-1))
    duv0_dxyz=(inv_z[...,None, None] * torch.cat((eye_blk, -uv0.unsqueeze(-1)), dim = -1)) # (*, N, 2, 3)

    uv = uv0 @ cam_K[...,:2,:2].mT + cam_K[...,None,:2,2] # (*, N, 2)
    duv_duv0 = cam_K[...,:2,:2].unsqueeze(-3) # (*, 1, 2, 2)
    
    r = uv - pts2d # (*, N, 2)

    duv0_dpose = torch.cat((duv0_dxyz@dxyz_daax, duv0_dxyz), dim = -1) #(*, N, 2, 6)
    dr_dpose = duv_duv0 @ duv0_dpose

    return r, dr_dpose


def _elem_jac_fn(d_aax_xyz, quat_xyz, cam_K, pts3d, pts2d):
    r, dr_dpose = residual_with_jac6d(quat_xyz, cam_K, pts3d, pts2d, d_aax_xyz)
    jac = r.unsqueeze(-1) * dr_dpose # (*, N, 2, 6)=(*, N, 2, 1) * (*, N, 2, 6)
    return jac.flatten(start_dim=-3), (jac, r) #(*, N*2*6), (*, N, 2, 6)

_elem_hess_fn = functorch.jacfwd(_elem_jac_fn, has_aux=True)


def hessian_6d_elem(quat_xyz:Tensor, cam_K:Tensor, pts3d:Tensor, pts2d:Tensor):
    """
    Args:
        quat_xyz : Shape (*, 7) operating point, wijk_xyz, function works only if operating point is at optimal point
        cam_K : shape (*, 3, 3), camera matrix
        pts3d : Shape (*, N, 3), 3d points in model frame
        pts2d : Shape (*, N, 2), measured 2d point positions
        
    Return:
        hess (*, N, 2, 6, 6) coordinate wise hessian matrix w.r.t. 6d pose update
        jac  (*, N, 2, 6)    coordinate wise jacobian matrix w.r.t. 6d pose update
    """
    hess_fn = functorch.vmap(_elem_hess_fn) if len(pts3d.shape)==3 else _elem_hess_fn
    d_aax_xyz = pts2d.new_zeros(pts2d.shape[:-2]+(6,))
    hess_flatten, (jac, r) = hess_fn(d_aax_xyz, quat_xyz, cam_K, pts3d, pts2d)  #(*, N*2*6, 6), (*, N, 2, 6), (*, N, 2)
    hess:Tensor = hess_flatten.view(jac.shape[:-1]+(6,6))   #(*, N, 2, 6, 6)
    return hess, jac,


def diff_pnp_perturb(quat_xyz: Tensor, cam_K:Tensor, pts3d: Tensor, pts2d: Tensor, icov2 : Tensor, with_cov = True):
    """
    Args:
        quat_xyz : Shape (*, 7) operating point, wijk_xyz, function works only if operating point is at optimal point
        cam_K : shape (*, 3, 3), camera matrix
        pts3d : Shape (*, N, 3), 3d points in model frame
        pts2d : Shape (*, N, 2), measured 2d point positions
        icov2 : Shape (*, N, 2, 2) or (*, N, 1) or (*, N) inverse of covariance

    Return:
        invalid : (*) invalid masks
        right_update: (*, 6) pose update with aax_xyz rep, gradient information attached
        cov: (*, 6, 6) or None,   precise covariance matrix if with_cov is True, else None.
    """
    hessians, jacs, = hessian_6d_elem(quat_xyz, cam_K, pts3d, pts2d)
    hessians, jacs, = weighted_hess_jac(icov2, hessians, jacs)
    hessians = make_sure_symmetric(hessians)

    hess_sqrtL, invalid = safe_cholesky(hessians)
    right_update = nll_update(hess_sqrtL, jacs.unsqueeze(-1))

    cov = torch.cholesky_inverse(hess_sqrtL) if with_cov else None
    return invalid, right_update, cov


def weighted_pnp_jac_wrt_pts2d(pts2d: Tensor, state_gt: Tensor, cam_K: Tensor, pts3d: Tensor, weights: Tensor, with_cov = False):
    """
    Args:
        pts2d : Shape (*, N, 2), measured 2d point positions
        quat_xyz : Shape (*, 7) operating point, wijk_xyz, function works only if operating point is at optimal point
        cam_K : shape (*, 3, 3), camera matrix
        pts3d : Shape (*, N, 3), 3d points in model frame
        weights : Shape (*, N, 2, 2) or (*, N, 1) or (*, N) inverse of covariance
    """
    batched = len(pts3d.shape) != 2
    create_graph = any(t.requires_grad for t in (state_gt, cam_K, pts3d, pts2d, weights))

    if not pts2d.requires_grad:
        pts2d = pts2d.detach().requires_grad_(True)
        
    _, update_6d, cov = diff_pnp_perturb(state_gt, cam_K, pts3d, pts2d, weights, with_cov=with_cov)
    outs = update_6d.sum(-2) if batched else update_6d

    def _jac_fn(grad_out):
        return torch.autograd.grad(outs, pts2d, grad_out, create_graph=create_graph, allow_unused=True)[0]
    jac_fn = functorch.vmap(_jac_fn)

    jac:Tensor = jac_fn(torch.eye(6, dtype=outs.dtype,device=outs.device))
    jac = jac.permute(1,0,2,3) if batched else jac # (*, 6, N, 2) or (6, N, 2)
    return (jac, cov) if with_cov else jac
    

