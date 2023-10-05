import torch
import numpy as np
from torch import Tensor
from typing import Union, List
from operator import itemgetter
from lib.utils.grad import NormClipper
import lib.transforms as xforms
import torch.nn.functional as F
import torch.nn as nn

import symmetry
import floatbits

from lib.cov_mixed import Loss_cov_mixed


def nn_out_to_xyz(
    nn_out:Tensor = None, 
    noc_scale_xfd:Tensor = None, *,
    raw_bits_gt:Tensor = None, 
    noc_mask:Tensor = None, 
    model_transform:Tensor = None, 
    bit_cnt:Union[int, List[int]] = None, inference = False):
    '''
    Args:
    nn_out: (B, C, H, W)
    nn_out_gt: (B, C, H, W)
    model_xform:(B, 4, 4)

    Output:
    xyz: (B, H_out, W_out, 3)
    '''
    if bit_cnt is None:  #nn_out(B, 3, H, W), continuous xyz output, simple
        assert model_transform is None, 'Model transform not implemented for continuous xyz output'
        return nn_out.permute(0, 2, 3, 1) * noc_scale_xfd[:,None,None,:]   #(B, H, W, 3)

    if not inference:
        noc_xformed = floatbits.nn_logits2noc_with_gt(nn_out, raw_bits_gt, bit_cnt, noc_mask)    #(B,H,W,3)
    else:
        noc_xformed = floatbits.nn_logits2noc(nn_out, bit_cnt)

    xyz_xformed = noc_xformed * noc_scale_xfd[:,None,None,:]    #(B,H,W,3)*(B,1,1,3)

    xyz:Tensor = (xyz_xformed-model_transform[:,None,None,:3,3])@model_transform[:,None,:3,:3] if model_transform is not None else xyz_xformed
    return xyz


@torch.no_grad()
def xyz_to_nn_target(xyz:Tensor, noc_scale_xfd:Tensor = None, *,
    noc_mask:Tensor = None, 
    model_transform:Tensor=None, bit_cnt:Union[int, List[int]] = None):
    '''
    xyz: (B, H, W, 3)
    '''
    xformed = (xyz @ model_transform[:,None,:3,:3].mT + model_transform[:,None,None,:3,3]) if model_transform is not None else xyz
    if model_transform is not None and noc_mask is not None:
        xformed = xformed * noc_mask.unsqueeze(-1)
    noc = xformed/noc_scale_xfd[:,None,None,:]

    if bit_cnt is None:
        assert model_transform is None, 'coordinate transform not implemented for continuous xyz output'
        return noc.permute(0,3,1,2), None
        
    target, raw_bits_gt = floatbits.nn_noc2target(noc, bit_cnt)
    return target, raw_bits_gt


def selete_best_pose(gt_dict, out_dict, sym_aware_started):
    candis, homo_z, R_no_aug, t_no_aug, K_no_aug, msk_noc,  = \
        itemgetter('Rt_candi', 'homo_z_out', 'R_no_aug', 't_no_aug', 'K_no_aug', 'msk_noc', )(gt_dict)

    if len(candis)==1 and candis[0].shape[-3]==1:  # no symmetric candidates
        Rt_best = candis[0].squeeze(-3)
        pose_best = xforms.RT_to_quaternion_rep(Rt_best[...,:3,:3],Rt_best[...,:,3])
        xyz_gt = xyz_from_homo_z(homo_z, R_no_aug, t_no_aug, K_no_aug) * msk_noc.unsqueeze(-1)
        return Rt_best, pose_best, xyz_gt

    chunk_sizes = [len(candi_blk) for candi_blk in candis]

    if not sym_aware_started:
        bests = [candi[...,0,:,:] for candi in candis]
    else:
        if 'pts2d' not in out_dict:
            K_no_aug_chunks = K_no_aug.split(chunk_sizes)
            noc_scale = gt_dict['noc_scale']
            pts2d:Tensor = gt_dict['sym_ck_pts2d']  #(B, N, 2)
            batch_idx = torch.arange(len(pts2d))[...,None].expand(-1,pts2d.shape[-2])   #(B, N)

            if 'xyz_noc' in out_dict:
                nn_out = out_dict['xyz_noc']
            elif 'xyz_noc_bin' in out_dict:
                nn_out = out_dict['xyz_noc_bin']
            else:
                raise RuntimeError('False branch')

            nn_out_samples:Tensor = nn_out[batch_idx,...,pts2d[...,1],pts2d[...,0]]
            ck_net_out = nn_out_samples.mT.unsqueeze(-1)   # unsqueeze to make it look like batched network output
            ck_pt3ds = nn_out_to_xyz(ck_net_out, noc_scale, 
                bit_cnt=gt_dict.get('bit_cnt',None),
                model_transform=gt_dict.get('model_transform',None),
                inference=True).squeeze(-2).split(chunk_sizes)

            homo_z_trunks = homo_z[batch_idx, pts2d[...,1],pts2d[...,0]].split(chunk_sizes)
            bests = [symmetry.select_pose_3d(cK, p3d, homo_z, candi) \
                for cK, p3d, homo_z, candi in zip(K_no_aug_chunks, ck_pt3ds, homo_z_trunks, candis)]
        else:
            out_K_chunks = gt_dict['out_K'].split(chunk_sizes)
            pts2d_chunks = out_dict['pts2d'].split(chunk_sizes)
            pts3d_chunks = gt_dict['pts3d'].split(chunk_sizes)
            bests = [symmetry.select_pose_2d(oK, p3d, p2d, candi) \
                for oK, p3d, p2d, candi in zip(out_K_chunks, pts3d_chunks, pts2d_chunks,candis)]


    Rt_best = torch.cat(bests,dim=0)
    pose_best = xforms.RT_to_quaternion_rep(Rt_best[...,:3,:3],Rt_best[...,:,3])
    xyz_gt = xyz_from_homo_z(homo_z, Rt_best[...,:3,:3],Rt_best[...,:,3], K_no_aug) * msk_noc.unsqueeze(-1)
    return Rt_best, pose_best, xyz_gt


@torch.no_grad()
def annots_on_the_fly(gt_dict, out_dict, cfg_global, step):
    sym_aware_start_step = cfg_global.get('sym_aware_start', 0)
    sym_aware_started = step >= sym_aware_start_step

    Rt_best, pose_best, xyz_gt = selete_best_pose(gt_dict, out_dict, sym_aware_started)

    xyz_target, raw_bits = xyz_to_nn_target(xyz_gt, gt_dict['noc_scale'], 
        noc_mask=gt_dict['msk_noc'],
        model_transform=gt_dict.get('model_transform',None),
        bit_cnt=gt_dict.get('bit_cnt',None))

    annot_dict = dict(Rt_best = Rt_best, pose_best = pose_best, xyz_gt = xyz_gt)
    if raw_bits is None:
        annot_dict['xyz_noc_tgt'] = xyz_target
    else:
        annot_dict['xyz_noc_bin_tgt'] = xyz_target
        annot_dict['xyz_noc_bin_raw'] = raw_bits

    gt_dict.update(annot_dict)


def dense_pnp_matching_from_xyz(
    xyz_out:Tensor,
    weights_out:Tensor, valid_msk_full:Tensor,
    xyz_scale:Tensor, sample = 2, top_left = None):
    '''
    noc_out: (*, 3, H, W)
    weights_out: (*, 2, H, W)
    dilated_msk: (*, H, W)
    noc_factor:  (*, 3)
    '''
    top, left = np.random.randint(0,sample,size=2) if top_left is None else top_left
    uv_grid = xforms.gen_uv(xyz_out.shape[-2:],xyz_out.device)
    pts2d = uv_grid[...,top::sample,left::sample,:].flatten(start_dim=-3,end_dim=-2)     #(*, N, 2), N=(W*H)/(sample**2)
    inv_std2d = weights_out[...,top::sample,left::sample].flatten(start_dim=-2).mT   #(*, N, 2)
    pts3d = xyz_out[...,top::sample,left::sample].flatten(start_dim=-2).mT 
    if xyz_scale is not None:
        pts3d = pts3d * xyz_scale.unsqueeze(-2) #(*, N, 3)
    valid_msk = valid_msk_full[...,top::sample,left::sample].flatten(start_dim=-2) if valid_msk_full is not None else None #(*, N)

    return pts2d.expand_as(inv_std2d), inv_std2d, pts3d, valid_msk

def dense_pnp_matching_from_noc_bin(
    noc_bin_out_logits:Tensor, noc_bin_gt_raw:Tensor,
    weights_out:Tensor, valid_msk_full:Tensor, noc_mask:Tensor,
    noc_scale:Tensor, gt_dict:dict, sample = 2, top_left = None):

    top, left = np.random.randint(0,sample,size=2) if top_left is None else top_left
    uv_grid = xforms.gen_uv(weights_out.shape[-2:],weights_out.device)
    pts2d = uv_grid[...,top::sample,left::sample,:].flatten(start_dim=-3,end_dim=-2)     #(*, N, 2), N=(W*H)/(sample**2)
    inv_std2d = weights_out[...,top::sample,left::sample].flatten(start_dim=-2).mT   #(*, N, 2)
    valid_msk = valid_msk_full[...,top::sample,left::sample]

    pts3d_noc_bin_out_logits = noc_bin_out_logits[...,top::sample,left::sample]
    pts3d_noc_bin_raw = noc_bin_gt_raw[...,top::sample,left::sample]

    pts3d:Tensor = nn_out_to_xyz(pts3d_noc_bin_out_logits, noc_scale,
        raw_bits_gt = pts3d_noc_bin_raw,
        noc_mask = noc_mask[...,top::sample,left::sample],
        model_transform = gt_dict.get('model_transform',None),
        bit_cnt = gt_dict['bit_cnt'])
    pts3d = pts3d.flatten(start_dim=-3, end_dim=-2)
    valid_msk = valid_msk.flatten(start_dim=-2)
    return pts2d.expand_as(inv_std2d), inv_std2d, pts3d, valid_msk


def xyz_from_homo_z(homo_z:Tensor, pose_R:Tensor, pose_t:Tensor, cam_K:Tensor):
    inv_K = torch.linalg.inv_ex(cam_K)[0].unsqueeze(-3)
    xR = pose_R.unsqueeze(-3).mT
    neg_xt = pose_t[...,None,None,:] @ xR.mT

    pts3d = homo_z @ (inv_K.mT @ xR.mT) - neg_xt
    return pts3d


class Loss_xyz_bin(nn.Module):
    def __init__(self, total_bit_cnt:int, momentum = 0.05) -> None:
        super().__init__()
        self.register_buffer('histogram',torch.full((total_bit_cnt,), 0.5))
        self.momentum = momentum

    def forward(self, noc_xyz_bin_logits:Tensor, noc_xyz_bin_gt:Tensor, msk_vis_logits:Tensor):
        msk_hard = msk_vis_logits > 0
        code1_hard = noc_xyz_bin_logits > 0
        code2_hard = noc_xyz_bin_gt.to(torch.bool, non_blocking=True) 
        hamm = code1_hard.logical_xor_(code2_hard).logical_and_(msk_hard)
        hist = hamm.sum([0,2,3])/(msk_hard.sum()+1)
        self.histogram.mul_(1-self.momentum).add_(hist * self.momentum)

        hist_soft = torch.minimum(self.histogram, 0.51-self.histogram)
        bin_weights = (hist_soft*3).softmax(dim=-1)

        loss_raw = F.binary_cross_entropy_with_logits(noc_xyz_bin_logits * msk_hard, noc_xyz_bin_gt.float(), reduction='none')
        weighted = (loss_raw.mean([0,2,3]) * bin_weights).sum(-1)
        loss = weighted.mean()
        return loss


class Loss_seg_L1(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        weight: Tensor= None,
        reduction: str = "mean",):
        prob = input.sigmoid()
        err = (prob - target).abs()
        if weight is not None:
            err = err * weight
        if reduction == "mean":
            return err.mean()
        else:
            raise NotImplementedError


class Loss_fn(nn.Module):
    def __init__(self, cfg, cfg_global, total_bit_cnt = 0) -> None:
        super().__init__()
        self.cfg = cfg
        pose_cfg = cfg.pose_loss_cfg

        self.weight_grad_clipper = NormClipper() if pose_cfg.get('clip_weight_grad',True) else None
        self.scale_grad_clipper = NormClipper(rel_thresh = 2) if pose_cfg.get('clip_scale_grad',False) else None
        self.pts_grad_clipper = NormClipper(rel_thresh = 2) if pose_cfg.get('clip_pts_grad',False) else None

        self.cfg_global = cfg_global
        binary_bits = total_bit_cnt
        if binary_bits > 0:
            self.xyz_bin_loss_fn = Loss_xyz_bin(binary_bits)

        seg_loss_type = cfg.get('seg_loss_type', 'BCE')
        if seg_loss_type.lower()=='bce':
            self.seg_loss_fn = F.binary_cross_entropy_with_logits
        elif seg_loss_type.lower()=='l1':
            self.seg_loss_fn = Loss_seg_L1()

    
    def forward(self, gt_dict, out_dict, epoch, step, steps_per_epoch):
        cfg = self.cfg
        msk_noc, msk_vis = itemgetter('msk_noc', 'msk_vis')(gt_dict)
        loss_dict = {}

        # sparse case
        if 'pts2d' in out_dict:
            loss_kpts = self.sparse_kpt_loss(cfg, gt_dict, out_dict)
            loss_dict['loss_kpts'] =loss_kpts
            if cfg.get('w_loss_pose',0)>0:
                # fully apply lc loss after some steps
                full_pose_loss_step = max(cfg.get('pose_loss_start_step',0), cfg.get('pose_loss_start_epoch',0)*steps_per_epoch)
                loss_pose_nz_step = cfg.get('loss_pose_nz_step',0)
                loss_pose_factor = max(0,min((step - loss_pose_nz_step + 1) / (max(full_pose_loss_step - loss_pose_nz_step,0)+1e-5),1))
                loss_pose = self.sparse_pose_loss(cfg, gt_dict, out_dict)
                loss_dict['loss_pose']= loss_pose_factor * loss_pose + (1-loss_pose_factor)*loss_kpts

            w_loss_dict = {k : v*cfg.get('w_'+k,0) for k,v in loss_dict.items() if cfg.get('w_'+k,0)>0}
            return loss_dict, w_loss_dict

        # dense case
        has_noc, has_noc_bin = 'xyz_noc' in out_dict, 'xyz_noc_bin' in out_dict

        if has_noc:
            # gdr-net structure, the surrogate loss for direct correspondance learning
            noc_msked, noc_gt = out_dict['xyz_noc']*msk_noc.unsqueeze(-3), gt_dict['xyz_noc_tgt']
            loss_dict['loss_noc'] = torch.nn.functional.l1_loss(noc_msked, noc_gt, reduction = 'mean')

        if has_noc_bin:
            # zebra-pose structure, the surrogate loss for direct correspondance learning
            loss_dict['loss_noc_bin'] = self.xyz_bin_loss_fn(out_dict['xyz_noc_bin'], gt_dict['xyz_noc_bin_tgt'], out_dict['msk_vis_logits'])
        
        loss_seg = self.seg_loss_fn(out_dict['msk_vis_logits'], msk_vis.unsqueeze(-3), reduction='mean')
        loss_dict['loss_seg'] = loss_seg

        full_pose_loss_step = max(cfg.get('pose_loss_start_step',0), cfg.get('pose_loss_start_epoch',0)*steps_per_epoch)

        loss_pose = self.dense_pose_loss(cfg.pose_loss_cfg, gt_dict, out_dict)

        # fully apply lc loss after some steps
        loss_pose_nz_step = cfg.get('loss_pose_nz_step',0)
        loss_pose_factor = max(0,min((step - loss_pose_nz_step + 1) / (max(full_pose_loss_step - loss_pose_nz_step,0)+1e-5),1))

        if loss_pose_factor != 1:
            weight_logits = out_dict['xyz_weight_logits']
            msk_vis_tgt = msk_vis.unsqueeze(-3).expand_as(weight_logits)
            loss_weight_seg = self.seg_loss_fn(weight_logits, msk_vis_tgt, reduction='mean')
            loss_pose = loss_pose_factor * loss_pose + (1 - loss_pose_factor)*loss_weight_seg
            
        loss_dict['loss_pose'] = loss_pose

        loss_dict = {k : v.mean() if len(v.shape)>0 else v \
            for k,v in loss_dict.items() if isinstance(v, Tensor)}

        w_loss_dict = {k : v*cfg.get('w_'+k,0) for k,v in loss_dict.items() if cfg.get('w_'+k,0)>0}
        return loss_dict, w_loss_dict

    def sparse_kpt_loss(self, cfg, gt_dict, out_dict):
        # nll based on Laplace distribution
        pts2d, pts2d_std = itemgetter('pts2d','pts2d_std')(out_dict)
        pose_best, K, pts3d = itemgetter('pose_best', 'out_K', 'pts3d',)(gt_dict)
        pts2d_proj = xforms.project_apply(K, pts3d, *xforms.quaternion_rep_to_RT(pose_best))
        err = (pts2d - pts2d_proj).abs()
        std = pts2d_std
        nll_loss = torch.log(std)+err/std
        return nll_loss.mean()


    def sparse_pose_loss(self, cfg, gt_dict, out_dict):
        pts2d, pts2d_std = itemgetter('pts2d','pts2d_std')(out_dict)
        pts2d_inv_std = 1 / pts2d_std
        pose_best, K, pts3d, bbox_3d = itemgetter('pose_best', 'out_K', 'pts3d', 'bbox_3d')(gt_dict)
        loss_cov = Loss_cov_mixed(K, pose_best, pts3d, pts2d, pts2d_inv_std, None, bbox_3d = bbox_3d)
        return loss_cov.mean()

    def dense_pose_loss(self, cfg, gt_dict, out_dict):
        noc_scale = gt_dict['noc_scale']
        pose_best, K, bbox_3d = itemgetter('pose_best', 'out_K', 'bbox_3d')(gt_dict)
        has_noc, has_noc_bin = 'xyz_noc' in out_dict, 'xyz_noc_bin' in out_dict

        # install gradient clippers on weight related outputs
        xyz_weight_logits:Tensor = out_dict['xyz_weight_logits']
        if self.weight_grad_clipper is not None:
            def clip_weight_grad(grad:Tensor):
                return self.weight_grad_clipper.clip(grad)
            xyz_weight_logits.register_hook(clip_weight_grad)

        xyz_weights_scale:Tensor = out_dict['xyz_weights_scale']
        if self.scale_grad_clipper is not None:
            def clip_scale_grad(grad:Tensor):
                return self.scale_grad_clipper.clip(grad)
            xyz_weights_scale.register_hook(clip_scale_grad)

        # calculate weights
        xyz_weights_raw:Tensor = xyz_weight_logits.reshape(xyz_weight_logits.shape[:-3]+(1, -1)).softmax(dim=-1)
        xyz_weights = xyz_weights_raw.reshape_as(xyz_weight_logits) * xyz_weights_scale

        dense_sample = cfg.get('dense_sample', 2)
        
        assert has_noc != has_noc_bin  # either use gdr-net structure or zebrapose structure

        if has_noc: # gdr-net structure
            xyz_noc:Tensor = out_dict['xyz_noc']
            den_pts2d, den_inv_std2d, den_pts3d, _ = dense_pnp_matching_from_xyz(
                xyz_noc, xyz_weights, gt_dict['msk_vis'], noc_scale, sample = dense_sample)
            den_valid_msk = torch.ones_like(den_pts3d[...,0])
            
        if has_noc_bin: # zebrapose structure
            xyz_noc_bin_logits = out_dict['xyz_noc_bin']
            msk_noc, noc_bin_raw_gt = gt_dict['msk_noc'], gt_dict['xyz_noc_bin_raw']
            msk_vis = out_dict['msk_vis_logits'] > 0
            den_pts2d, den_inv_std2d, den_pts3d, _ = dense_pnp_matching_from_noc_bin(
                xyz_noc_bin_logits, noc_bin_raw_gt, xyz_weights,
                msk_vis.squeeze(-3), msk_noc, noc_scale, gt_dict, sample = dense_sample)
            den_valid_msk = torch.ones_like(den_pts3d[...,0])
        
        # install gradient clippers for 3d points
        if self.pts_grad_clipper is not None:
            def clip_pts_grad(grad:Tensor):
                return self.pts_grad_clipper.clip(grad)
            den_pts3d.register_hook(clip_pts_grad)

        loss_cov = Loss_cov_mixed(K, pose_best, den_pts3d, den_pts2d, den_inv_std2d, den_valid_msk, 
                                  bbox_3d=bbox_3d, max_err_len = cfg.get('max_err_len',32),)

        return loss_cov.mean()
