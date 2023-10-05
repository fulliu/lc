import json
import torch
import numpy as np
import math
from scipy.spatial.transform import Rotation as sciR
from torch import Tensor

@torch.jit.script
def select_pose_2d(cam_K:Tensor, pts3d:Tensor, pts2d:Tensor, pose_candi:Tensor):
    '''
    cam_K: (B, 3, 3)
    pts3d: (B, N, 3)
    pts2d: (B, N, 2)
    pose_candi: (B, K, 3, 4)
    '''
    if pose_candi.shape[-3]==1:
        return pose_candi.squeeze(-3)

    # cam_K, pts3d, pts2d = (t.unsqueeze(-3) for t in (cam_K, pts3d, pts2d))  #(B, 1, ...)
    cam_K, pts3d, pts2d = cam_K.unsqueeze(-3),pts3d.unsqueeze(-3),pts2d.unsqueeze(-3) #(B, 1, ...)
    xformed = pts3d @ pose_candi[...,:3,:3].mT + pose_candi[...,None,:3,3] #(B,K,N,3)
    homo = xformed @ cam_K.mT
    uv = homo[...,:2]/homo[...,2:3] #(B,K,N,2)

    err = torch.linalg.vector_norm(uv - pts2d,dim=-1).mean(dim=-1)  #(B,K)
    min_idx = torch.argmin(err,dim=-1)  #(B,)

    batch_idx = torch.arange(len(min_idx),device=min_idx.device)
    best_pose = pose_candi[batch_idx, min_idx]  #(B, 3, 4)
    return best_pose
    
@torch.jit.script
def select_pose_3d(cam_K:Tensor, pts3d_out:Tensor, homo_z:Tensor, pose_candi:Tensor):
    '''
    cam_K: (B, 3, 3)
    pts3d_out: (B, N, 3)
    homo_z:    (B, N, 3)   # ground truth homography coordinate
    pose_candi: (B, K, 3, 4)
    '''
    if pose_candi.shape[-3]==1:
        return pose_candi.squeeze(-3)
    inv_K = torch.linalg.inv_ex(cam_K)[0].unsqueeze(-3)

    # pts3d_out, homo_z = (t.unsqueeze(-3) for t in (pts3d_out, homo_z))  #(B, 1, ...)
    pts3d_out, homo_z = pts3d_out.unsqueeze(-3),homo_z.unsqueeze(-3) #(B, 1, ...)
    xR = pose_candi[...,:3,:3].mT
    neg_xt = pose_candi[...,None,:3,3] @ xR.mT

    pts3d_ref = homo_z @ (inv_K.mT @ xR.mT) - neg_xt

    err = torch.linalg.vector_norm(pts3d_out - pts3d_ref, dim=-1).mean(dim=-1)  #(B,K)
    min_idx = torch.argmin(err,dim=-1)  #(B,)

    batch_idx = torch.arange(len(min_idx),device=min_idx.device)
    best_pose = pose_candi[batch_idx, min_idx]  #(B, 3, 4)
    return best_pose

def symmetry_pose_candidates(base_R, base_t, model_info, continuous_steps = 384):
    continuous = 'symmetries_continuous' in model_info
    discrete = 'symmetries_discrete' in model_info
    if continuous and discrete:
        '''
        implement by changing: pose_candi: (B, K, 3, 4) to pose_candi:(B, D, C, 3, 4), 
        D stands for discrete symmetry counts, C stands for discretized continuous symmetries
        '''
        raise NotImplementedError
    elif discrete:
        xform_Rs = [np.eye(3)]
        xform_ts = [np.zeros(3)]
        for sym in model_info['symmetries_discrete']:
            sym_44 = np.reshape(sym,(4, 4))
            xform_Rs.append(sym_44[:3,:3])
            xform_ts.append(sym_44[:3,3])

        xform_Rs = np.stack(xform_Rs)
        xform_ts = np.stack(xform_ts)
    elif continuous:
        cont_syms = model_info['symmetries_continuous']
        assert len(cont_syms) == 1
        sym = cont_syms[0]
        axis = np.array(sym['axis'])
        offset = np.array(sym['offset'])
        rotvecs = np.linspace(0, 2*math.pi, continuous_steps, endpoint=False)[...,None]*axis
        xform_Rs = sciR.from_rotvec(rotvecs).as_matrix()
        xform_ts = np.matmul(xform_Rs, -offset) + offset
    else:
        xform_Rs = np.eye(3)[None]
        xform_ts = np.zeros(3)[None]

    candi_Rs = base_R @ xform_Rs
    candi_ts = (base_R @ xform_ts[...,None])[...,0] + base_t
    candi_Rts = np.concatenate((candi_Rs, candi_ts[...,None]),axis=-1)
    return candi_Rts

