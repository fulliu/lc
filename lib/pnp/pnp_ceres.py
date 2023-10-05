from ._ext import lib
from ._ext import ffi
import numpy as np
import torch

def solve(
    cam_mat,    # (*, 3, 3) or list of (3, 3)
    pts3d,      # (*, N, 3) or list of (N, 3)
    pts2d,      # (*, N, 2) or list of (N, 2)
    pts2d_icov_sqrtL, # (*, N, 2, 2) or list of (N, 2, 2)
    start,      # (*, 7) or list of (7)
    n_points = None,   # (*) or list of scalar
    *,
    max_iter_count = 50,
    num_workers = 1,
    **kwargs,
):
    '''
    Args:
    start_state:Tensor or list(Tensor) (*, 7) or list of (7)
    coord_3d:   Tensor or list(Tensor) (*, N, 3) or list of (N, 3)
    coord_2d:   Tensor or list(Tensor) (*, N, 2) or list of (N, 2)
    icov_sqrtL: Tensor or list(Tensor) (*, N, 2, 2) or list of (N, 2, 2)
    cam_mat:    Tensor or list(Tensor) (*, 3, 3) or list of (3, 3)

    Returns:
    states: tuple of (list(states), list(trust regions))
    '''
    single = False
    state = start
    if not isinstance(state, (list, tuple)) and len(state.shape) == 1:   # not list and not batched
        single = True
        if pts2d[0].shape == pts2d_icov_sqrtL[0].shape:
            pts2d_icov_sqrtL = torch.diag_embed(pts2d_icov_sqrtL)
        state = [_to_np(state)]
        pts3d = [_to_np(pts3d)]
        pts2d = [_to_np(pts2d)]
        pts2d_icov_sqrtL = [_to_np(pts2d_icov_sqrtL)]
        cam_mat = [_to_np(cam_mat)]
        n_points = [int(n_points)] if n_points is not None else [pts2d.shape[0]]
    else:
        if pts2d[0].shape == pts2d_icov_sqrtL[0].shape:
            pts2d_icov_sqrtL = [torch.diag_embed(c) for c in pts2d_icov_sqrtL]
        state = [_to_np(s) for s in state]
        pts3d = [_to_np(c) for c in pts3d]
        pts2d = [_to_np(c) for c in pts2d]
        pts2d_icov_sqrtL = [_to_np(c) for c in pts2d_icov_sqrtL]
        cam_mat = [_to_np(c) for c in cam_mat]
        n_points = [int(n) for n in n_points] if n_points is not None else [int(coords.shape[0]) for coords in pts2d]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dtype = pts3d[0].dtype
    state = [_make_cont(s, dtype) for s in state]
    pts3d = [_make_cont(c, dtype) for c in pts3d]
    pts2d = [_make_cont(c, dtype) for c in pts2d]
    pts2d_icov_sqrtL = [_make_cont(c, dtype) for c in pts2d_icov_sqrtL]
    cam_mat = [_make_cont(c, dtype) for c in cam_mat]
    outputs = _pnp_ceres_omp_f32(state, pts3d, pts2d, pts2d_icov_sqrtL, cam_mat, n_points, \
        worker_count= num_workers,max_iter_count=max_iter_count, **kwargs,)
    outputs = [torch.from_numpy(o) for o in outputs]
    return outputs if not single else [out[0] for out in outputs]


def _to_np(val):
    if not isinstance(val, np.ndarray):
        val = val.detach().to(device='cpu',non_blocking=True).numpy()
    return val

def _make_cont(val, dtype=None):
    dtype = val.dtype if dtype is None else dtype
    return np.ascontiguousarray(val, dtype)


def _pnp_ceres_omp_f32(state,   #shape(B, 7) or list((7))
                    coord_3d,   #shape(B, N, 3) or list((N,3))
                    coord_2d,   #shape(B, N, 2) or list((N,2))
                    icov_sqrtL, #shape(B, N, 2, 2) or list((N,2,2))
                    cam_mat,    #shape(B, 3, 3) or list((3,3))
                    point_counts,         #shape(B)   or list(int)
                    worker_count=1,#default single threaded
                    max_iter_count=300,
                    **kwargs,
                    ):
        if isinstance(state, list):
            state = np.stack(state).astype(np.float32)
        else:
            state = state.astype(np.float32).copy()

        print_summary = int(kwargs.get('print_summary',0))
        function_tolerance = float(kwargs.get('function_tolerance', 1e-6))
        max_iter_count=int(max_iter_count)
        job_cnt = len(state)
        data_ptr_array=f'float*[{job_cnt}]'
        data_ptr = 'float*'
        states = ffi.new(data_ptr_array)
        coord_3ds = ffi.new(data_ptr_array)
        coord_2ds = ffi.new(data_ptr_array)
        icov_sqrtLs = ffi.new(data_ptr_array)
        cam_mats = ffi.new(data_ptr_array)

        result_tr = np.zeros((job_cnt), np.float32)  # trust region radius
        result_tr_ptr = ffi.cast(data_ptr, result_tr.ctypes.data)

        result_invalid_flags = np.zeros((job_cnt), np.int32)
        result_ret_ptr = ffi.cast('int*', result_invalid_flags.ctypes.data)
        point_counts=np.array(point_counts, dtype=np.int32)
        pt_cnt_ptr = ffi.cast('int*', point_counts.ctypes.data)
        
        for i in range(job_cnt):

            coord_3d_ptr = ffi.cast(data_ptr, coord_3d[i].ctypes.data)
            coord_2d_ptr = ffi.cast(data_ptr, coord_2d[i].ctypes.data)
            coord_2d_icov_sqrtL = ffi.cast(data_ptr, icov_sqrtL[i].ctypes.data)
            cam_mat_ptr = ffi.cast(data_ptr, cam_mat[i].ctypes.data)
            state_ptr = ffi.cast(data_ptr, state[i].ctypes.data)

            states[i]=state_ptr
            coord_3ds[i]=coord_3d_ptr
            coord_2ds[i]=coord_2d_ptr
            icov_sqrtLs[i]=coord_2d_icov_sqrtL
            cam_mats[i]=cam_mat_ptr
            '''
            void pnp_ceres_f32_omp(
              float ** init_states,
              float ** cam_Ks,
              float ** pts2ds,
              float ** pts3ds,
              float ** icov_sqrtLs,
              int * ptCnts, int printSummary,
              float* result_trs, int* rets,
              int job_count,
              int num_threads
            );
            '''
        num_threads = int(worker_count)
        lib.pnp_ceres_f32_omp(
            states, cam_mats, coord_2ds, coord_3ds, icov_sqrtLs, pt_cnt_ptr,max_iter_count,\
            function_tolerance, print_summary, result_tr_ptr, result_ret_ptr, job_cnt, num_threads
        )
        return state, result_tr, result_invalid_flags
        



