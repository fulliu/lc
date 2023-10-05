from . import pnp_ceres
import torch
from torch import Tensor


def solve(
    cam_mat: Tensor,    # (*, 3, 3) or list[(3, 3)]
    pts3d: Tensor,      # (*, N, 3) or list[(N, 3)]
    pts2d: Tensor,      # (*, N, 2) or list[(N, 2)]
    icovs: Tensor,      # (*, N, 2, 2) or (*, N, 2) or list[(N, 2, 2)|(N, 2)]
    start: Tensor,      # (*, 7)    or list[(7)]
    n_points: Tensor = None,   # (*)       or list[(0)|float]
    *,
    optimal_start = False,
    max_iter_count = 50,
    num_workers = 1,
    filter_input_nan = False,
    **kwargs,
    ):

    dev = pts3d[0].device
    if isinstance(pts3d, (list, tuple)):
        if n_points is None:
            n_points = [len(p3d) for p3d in pts3d]
        cam_mat, pts3d, pts2d, icovs, start, n_points = _batch_tensors(
            cam_mat, pts3d, pts2d, icovs, start, n_points, device=dev)
    if filter_input_nan:
        cam_mat, pts3d, pts2d, icovs, start, n_points = \
            (torch.nan_to_num(t) if t is not None else None for t in (cam_mat, pts3d, pts2d, icovs, start, n_points))
    invalid_dict = dict()
    
    start = start.detach()
    if optimal_start:
        solutions = start
    else:
        with torch.no_grad():
            if len(pts2d.shape)==len(icovs.shape):  #(*, N, 2)
                icovs_sqrtL = torch.diag_embed(icovs.sqrt())
            else:   #(*, N, 2, 2)
                icovs_sqrtL = torch.linalg.cholesky_ex(icovs)[0]

            opt_states, trust_regions, solver_invalids = pnp_ceres.solve(
                cam_mat, pts3d, pts2d, icovs_sqrtL, start, n_points, 
                max_iter_count=max_iter_count, num_workers=num_workers, **kwargs)

        solutions = opt_states.to(dev, non_blocking=True)
        invalid_dict['solver_invalids']=solver_invalids.to(device=dev, dtype=torch.bool, non_blocking=True)

    states = solutions

    invalid_dict, invalids = _combine_invalids(invalid_dict, states)
    states = torch.where(invalids[...,None], start.to(states.device), states)
    return invalid_dict, states


def _combine_invalids(invalid_dict, states):
    if not invalid_dict:
        shape = (len(states),) if isinstance(states, list) else states.shape[:-1]
        invalids = torch.zeros(shape, dtype=torch.bool, device=states[0].device)
        return dict(invalids = invalids), invalids

    invalids = torch.stack(tuple(invalid_dict.values())).any(dim=0)
    invalid_dict['invalids'] = invalids
    return invalid_dict, invalids


def _batch_tensors(*tensor_lsts,device=None):
    batched_all=[]
    for tensor_lst in tensor_lsts:
        if tensor_lst is None:
            batched_all.append(None)
            continue

        sample = tensor_lst[0]
        if isinstance(sample, Tensor):
            max_shape=max(t.shape[:1] for t in tensor_lst)
            batched = sample.new_zeros((len(tensor_lst),) + max_shape + sample.shape[1:])
            if len(batched.shape)>1:
                for i,t in enumerate(tensor_lst):
                    batched[i,:len(t)]=t
            else:
                for i,t in enumerate(tensor_lst):
                    batched[i]=t
        else:
            batched = torch.tensor(tensor_lst, device=device)
        batched_all.append(batched)
    return batched_all

