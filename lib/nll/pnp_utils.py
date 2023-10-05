import torch
from torch import Tensor
from typing import Union, Callable

# @torch.jit.script
def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a, b:  (*, 4): wijk
    Returns:
    (*, 4): wijk
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)
    

# @torch.jit.script
def axis_angle_to_quaternion_near_zero(axis_angle:Tensor):
    '''
    use tayler expansion to ensure 2nd order differentiable
    it is actually 3rd order differentiable because of the special structure
    of tayler expansion of sin(x) and cos(x) around zero
    '''
    angles_sq = axis_angle.pow(2).sum(-1, keepdim=True)
    axis_angle_times_sin_half_angles_over_angles =axis_angle*(0.5 - angles_sq / 48)
    cos_half_angles = 1 - angles_sq / 8
    quaternions = torch.cat((cos_half_angles, axis_angle_times_sin_half_angles_over_angles),dim=-1)
    return quaternions

def make_skew_symm(vec3:Tensor, zeros:Tensor = None):
    '''
    vec3: shape(*, 3)
    return skew symmetric matrix built from vec3
    '''
    aa0, aa1, aa2 = vec3.unbind(dim=-1)
    zeros = zeros if zeros is not None else aa0.new_zeros(())
    zeros = zeros.expand_as(aa0)
    skew_symm = torch.stack(
        (zeros, -aa2, aa1,
        aa2, zeros, -aa0,
        -aa1, aa0, zeros),
        dim=-1
    ).reshape(zeros.shape+(3,3))    #(*, 3, 3)
    return skew_symm


# @torch.jit.script
def axis_angle_rotate_point_jac_near_zero(axis_angle:Tensor, pts3d:Tensor):
    '''
    At most 3rd order approximation for axis_angle rotation with very little angles,
    used for hessian computation to build compute graph
    Parameters:
    axis_angle : shape (*, 3) axis_angle rotations, should always be zero
    pts3d : Shape (*, N, 3), 3d points in model frame
    Returns:
    pts : shape (*, N, 3) rotated points
    jac : shape (*, N, 3, 3) jacobian matrix
    '''
    zero = axis_angle.new_zeros(())
    skew_symm = make_skew_symm(axis_angle, zero) #(*, 3, 3)

    # jac_0 = zeros((3,3))
    # pts3d_0 = pts3d
    
    jac_1 = make_skew_symm(pts3d, zero).mT # jac_{n+1}=skew_symm.unsqueeze(-3) @ jac_{n}+make_skew_symm(pts3d_{n}).mT
    pts3d_1 = pts3d @ skew_symm.mT

    jac_2 = skew_symm.unsqueeze(-3) @ jac_1 + make_skew_symm(pts3d_1, zero).mT
    # pts3d_2 = pts3d_1 @ skew_symm.mT

    pts = pts3d + pts3d_1# + pts3d_2 * 0.5
    jac = jac_1 + jac_2 * 0.5

    return pts, jac


# @torch.jit.script
def weighted_hess_jac(icov2:Union[Tensor,None], hess:Tensor, jac:Tensor = None, sum_up = True):
    '''
    icov2:(*, N, 2, 2) or (*, N, 2) or (*, N, 1) or (*, N)
    hess: (*, N, 2, 6, 6)
    jac:  (*, N, 2, 6)
    '''
    if icov2 is not None:
        if len(hess.shape)==len(icov2.shape)+1: #hess (*, N, 2, 2)
            hess = icov2 @ hess.reshape(jac.shape[:-1]+(36,))   #(*, N, 2, 2)@(*, N, 2, 36)=(*, N, 2, 36)
            hess = hess.view(jac.shape[:-1]+(6,6))  #(*, N, 2, 6, 6)
            jac = icov2 @ jac

        else: #(*, N, 2) or (*, N, 1) or (*, N)
            icov2 = icov2.view(jac.shape[:-2]+(-1,)).expand(jac.shape[:-1]) #(*, N, 2)
            hess = icov2[..., None, None] * hess
            jac = icov2[..., None] * jac

    if sum_up:
        hess = hess.sum(dim=(-4,-3))
        jac = jac.sum(dim=(-3,-2))
            
    return hess, jac


# @torch.jit.script
def apply_perturb(quat_rep : Tensor, aax_xyz_perturb : Tensor):
    '''
    R_{perturbed} = R_{original} @ R_{perturb},
    t_{perturbed} = t_{original} + t_{perturb}
    '''
    a, b = quat_rep[...,:4], axis_angle_to_quaternion_near_zero(aax_xyz_perturb[...,:3])
    new_quat = quaternion_raw_multiply(a, b)
    new_xyz = quat_rep[...,4:7] + aax_xyz_perturb[...,3:6]
    return torch.cat((new_quat, new_xyz), dim = -1)


class _nll_update(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hess_sqrtL:Tensor, jtr:Tensor): # hess_sqrtL: (*, 6, 6), jtr: (*, 6, 1)
        ctx.save_for_backward(hess_sqrtL,)
        return hess_sqrtL.new_zeros(hess_sqrtL.shape[:-2]+(6,)) #(*, 6)

    @staticmethod
    def backward(ctx, grad_outputs:Tensor): # grad_outputs: (*, 6)
        vjp_res = - torch.cholesky_solve(grad_outputs.unsqueeze(-1), ctx.saved_tensors[0])  #(*, 6, 1)
        return None, vjp_res

nll_update:Callable[[Tensor,Tensor], Tensor] = _nll_update.apply    
'''nll_update(hess_sqrtL : Tensor, jtr : Tensor) -> Tensor
'''


@torch.jit.script
def make_sure_symmetric(sym : Tensor):
    half_sym = sym * 0.5
    return half_sym + half_sym.mT


@torch.jit.script
def make_sure_SPD(spd:Tensor):
    '''
    make sure all matrices are symmetric positive-definite,
    if not, substitute it with an Identity matrix, and mark in invalid.
    '''
    '''
    Args:
    spd: (*, N, N)  batch of matrices supposed to be symmetric positive-definite
    Outputs:
    fixed:(*, N, N) batch of matrices with non-symmetric-positive-definite subsitited by identity matrix
    info: (*)   info returned from cholesky_ex, zero value if matrix is SPD,
                else the order of the leading minor that is not positive-definite
    '''
    info:Tensor = torch.linalg.cholesky_ex(spd.detach())[1]    #_, (*)
    cond = (info!=0).view(info.shape+(1,1)) #(*, 1, 1)
    eye = torch.eye(spd.shape[-1], dtype=spd.dtype, device=spd.device)    #(N, N)
    fixed = torch.where(cond, eye, spd) #(*, N, N)
    return fixed, info

@torch.jit.script
def safe_cholesky(spd:Tensor):
    '''
    output: sqrtL, info
    '''
    fixed, info = make_sure_SPD(spd)
    sqrtL:Tensor = torch.linalg.cholesky_ex(fixed)[0]
    return sqrtL, info
    