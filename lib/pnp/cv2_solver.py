import torch
import cv2
import numpy as np
import torch
import multiprocessing
from .. transforms import rotation_conversions as rcvt

def solve(
    cam_mat,    # (*, 3, 3) or list of (3, 3)
    coord_3d,   # (*, N, 3) or list of (N, 3)
    coord_2d,   # (*, N, 2) or list of (N, 2)
    *,
    reprojectionError = 3.0,
    confidence=0.99,
    num_workers = 1,
    **kwargs,
):
    '''
    Args:
    cam_mat:    Tensor or list(Tensor) (*, 3, 3) or list of (3, 3)
    coord_3d:   Tensor or list(Tensor) (*, N, 3) or list of (N, 3)
    coord_2d:   Tensor or list(Tensor) (*, N, 2) or list of (N, 2)

    Returns:
    invalids,   Invalid indicator
    states,     Solutions
    inliers     Inliers from PnPRansac
    '''

    single = False
    if not isinstance(coord_2d, (list,tuple)) and len(coord_2d.shape) == 2:   # not list and not batched
        single = True
        coord_2d = [_to_np(coord_2d)]
        coord_3d = [_to_np(coord_3d)]
        cam_mat = [_to_np(cam_mat)]
    else:
        coord_2d = [_to_np(c) for c in coord_2d]
        coord_3d = [_to_np(c) for c in coord_3d]
        cam_mat = [_to_np(c) for c in cam_mat]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dtype = coord_2d[0].dtype
    coord_2d = [_make_cont(c, dtype) for c in coord_2d]
    coord_3d = [_make_cont(c, dtype) for c in coord_3d]

    cam_mat = [_make_cont(c, dtype) for c in cam_mat]
    reprojectionError = [reprojectionError]*len(coord_2d) if isinstance(reprojectionError,(int,float)) else reprojectionError.tolist()
    inputs = list(zip(coord_3d, coord_2d, cam_mat,[confidence]*len(coord_2d), reprojectionError))

    if len(inputs) > 2 and num_workers > 1:
        results = get_workers().map(_cv2_solve_from_tuple, inputs)
    else:
        results = map(_cv2_solve_from_tuple, inputs)
    outputs = list(results)

    invalids, states, inliers = list(zip(*outputs)) if not single else outputs[0]
    return invalids, states, inliers


def _to_np(val):
    if not isinstance(val, np.ndarray):
        val = val.detach().to(device='cpu',non_blocking=True).numpy()
    return val

def _make_cont(val, dtype=None):
    dtype = val.dtype if dtype is None else dtype
    return np.ascontiguousarray(val, dtype)

def _cv2_solve_from_tuple(input_tuple):
    coord_3d, coord_2d, cam_mat, confidence, reprojectionError = input_tuple
    try:
        retval, rvec, tvec, inliers = \
            cv2.solvePnPRansac(coord_3d, coord_2d, cam_mat,
                None, flags=cv2.SOLVEPNP_EPNP, confidence=confidence, iterationsCount=150,
                reprojectionError=reprojectionError)
    except:
        retval=False
        rvec=np.zeros((3,1),dtype=np.float64)
        tvec = rvec
        inliers = np.zeros((0,1),dtype=np.int32)
    states = torch.cat(
        (
            rcvt.axis_angle_to_quaternion(torch.from_numpy(rvec)[...,0]),
            torch.from_numpy(tvec)[...,0]
        ), dim=-1)
    if retval == False:
        inliers = np.zeros((0,1),dtype=np.int32)
    return not retval, states, torch.from_numpy(inliers[...,0]).to(torch.int64)


import atexit

worker_count = 6
worker_pool = None
def exit_handler():
    global worker_pool
    if worker_pool is not None:
        worker_pool.close()
        worker_pool.join()

atexit.register(exit_handler)


def get_workers()->multiprocessing.Pool:
    global worker_pool
    if worker_pool is None:
        worker_pool = multiprocessing.Pool(worker_count)
    return worker_pool

