import torch
import functorch
from torch import Tensor
from .nll import pnp_auto
from .nll import pnp_utils as nll_pnp_utils
from . import transforms as xforms
from typing import Callable


def twice_huber(val_abs:Tensor, delta):
    delta = delta.detach() if isinstance(delta, Tensor) else delta
    large = val_abs > delta
    return torch.where(large, delta*(2*val_abs - delta), val_abs**2)


def clamp_error(error:Tensor, max_err_len):
    with torch.no_grad():
        max_err_len = max_err_len[...,None] if isinstance(max_err_len, Tensor) else max_err_len
        err_len = torch.linalg.vector_norm(error, dim=-1) + 1e-6
        f = ((err_len - max_err_len)/err_len).unsqueeze(-1)
        large = f > 0
        delta = f * error * large

    return error - delta


def robust_weights_cov(inv_std2d_pred:Tensor, error2d:Tensor, valid_mask:Tensor, rel_thresh = 3, w_e_thresh = 4):
    error = error2d.abs()
    with torch.no_grad():
        vmsk, vcnt = (valid_mask.unsqueeze(-1), valid_mask.sum(-1,keepdim=True)) if valid_mask is not None else (None, None)
        mean_abs = (error * vmsk).sum(-2)/vcnt if vmsk is not None else error.mean(-2)
    cov = twice_huber(error, mean_abs.unsqueeze(-2) * rel_thresh)

    with torch.no_grad():
        w_e = (inv_std2d_pred ** 2) * cov
        mean_w_e = (w_e * vmsk).sum(-2)/vcnt if vmsk is not None else w_e.mean(-2)
        delta_inv_std = torch.sqrt((mean_w_e.unsqueeze(-2) * w_e_thresh)/(cov + 1e-6))
    weights = twice_huber(inv_std2d_pred, delta_inv_std)
    return weights, cov


def jac_update2alter(state:Tensor, xform_fn:Callable[[Tensor], Tensor]):
    update = state.new_zeros(state.shape[:-1]+(6,),requires_grad=True)
    quat_xyz = nll_pnp_utils.apply_perturb(state.detach(), update)
    xformed_rep = xform_fn(quat_xyz)

    outputs = xformed_rep.sum(-2)
    grad_outs = torch.eye(xformed_rep.shape[-1],device=quat_xyz.device, dtype=quat_xyz.dtype)
    jac:Tensor = functorch.vmap(lambda grad_out:torch.autograd.grad(outputs, update, grad_out,\
        retain_graph=False, create_graph=False, allow_unused=True)[0])(grad_outs).permute(1,0,2)    #(*, N, 6)
    return jac


def transformed_cov_from_jac(*update_covs:Tensor, jac:Tensor):
    results = [(jac @ c * jac).sum(-1) if isinstance(c, Tensor) else 0 for c in update_covs]  #((*, N, 6)@(*, 6, 6)*(*, N, 6)).sum(-1)=(*, N)
    return results if len(results)!=1 else results[0]


def xform_3d(state:Tensor, bbox3d:Tensor):
    R,t = xforms.quaternion_rep_to_RT(state)
    return (bbox3d @ R.mT + t.unsqueeze(-2)).flatten(start_dim=-2,end_dim=-1)


def xform_2d(state:Tensor, cam_K:Tensor, bbox3d:Tensor):
    R,t = xforms.quaternion_rep_to_RT(state)
    return xforms.project_apply(cam_K, bbox3d, R, t).flatten(start_dim=-2,end_dim=-1)


def loss_cov_3d(cov_xformed_diag:Tensor, diameter:Tensor = None):
    B = len(cov_xformed_diag)
    good = (cov_xformed_diag>0).all(dim=-1, keepdim=True)
    point_wise = cov_xformed_diag.reshape(B,-1,3)
    avg_distance = torch.where(good,point_wise.sum(-1),1).sqrt().mean(-1)
    avg_distance = avg_distance / diameter if diameter is not None else avg_distance
    return avg_distance


def loss_cov_2d(cov_xformed_diag:Tensor):
    B = len(cov_xformed_diag)
    good = (cov_xformed_diag>0).all(dim=-1, keepdim=True)
    point_wise = cov_xformed_diag.reshape(B,-1,2)
    avg_distance = torch.where(good,point_wise.sum(-1),1).sqrt().mean(-1)
    return avg_distance


def Loss_cov_mixed(K_out:Tensor, pose_gt:Tensor, pts3d:Tensor, pts2d_out:Tensor, inv_std2d:Tensor, valid_factor:Tensor, **kwargs)->Tensor:
    '''
    kwarg_dict includes:
        bbox_3d:Tensor,
        max_err_len = 32
        rel_thresh = 3
        w_e_thresh = 4
    '''
    bbox_3d = kwargs['bbox_3d']
    max_err_len = kwargs.get('max_err_len', 32)
    rel_thresh = kwargs.get('rel_thresh', 3)
    w_e_thresh = kwargs.get('w_e_thresh', 4)
    cov_2d = kwargs.get('cov_2d', False)
    
    R, t = xforms.quaternion_rep_to_RT(pose_gt)
    pts2d_proj = xforms.project_apply(K_out, pts3d, R, t)
    err_2d = pts2d_out - pts2d_proj
    error_clamped = clamp_error(err_2d, max_err_len)
    weights, cov_est = robust_weights_cov(inv_std2d, error_clamped, valid_factor, rel_thresh=rel_thresh, w_e_thresh=w_e_thresh)

    jac_pts2update, prior_update_cov = pnp_auto.weighted_pnp_jac_wrt_pts2d(
        pts2d_proj.detach(), pose_gt.detach(), K_out.detach(), pts3d.detach(), weights, with_cov=True)

    jac_pts2update = jac_pts2update.flatten(start_dim=-2,end_dim=-1) # (*, 6, N, 2) -> (*, 6, 2N)

    if cov_2d:
        xform_fn = lambda st: xform_2d(st, K_out, bbox_3d)
        loss_fn, err_dim = loss_cov_2d, 2
    else:
        xform_fn = lambda st: xform_3d(st, bbox_3d)
        loss_fn, err_dim = loss_cov_3d, 3
    
    jac_up2alter = jac_update2alter(pose_gt, xform_fn = xform_fn)

    prior_alter_cov = transformed_cov_from_jac(prior_update_cov, jac = jac_up2alter)
    prior_error = loss_fn(prior_alter_cov)

    jac_d =  jac_pts2update

    cov_cal = cov_est.flatten(start_dim=-2,end_dim=-1) # (*, 2N)
    half_update_cov = jac_d * cov_cal.unsqueeze(-2) @ jac_d.mT * 0.5

    update_cov = half_update_cov + half_update_cov.mT
    alter_cov = transformed_cov_from_jac(update_cov, jac = jac_up2alter)
    cov_err = loss_fn(alter_cov)

    delta = (jac_up2alter @ (jac_pts2update @ error_clamped.detach().flatten(start_dim=-2).unsqueeze(-1))).squeeze(-1)
    linear_err = torch.linalg.vector_norm(delta.reshape(delta.shape[:-1]+(8,err_dim)),dim=-1).mean(dim=-1)

    loss_pose = prior_error.log() + 0.5 * (cov_err + linear_err)/prior_error
    return loss_pose
