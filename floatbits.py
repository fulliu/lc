import torch
import math
from torch import Tensor
from typing import Union, List
from collections import abc

_black_background = True

def set_black_background(black = True):
    global _black_background
    _black_background = black

def nn_noc2target(noc:Tensor, bit_cnt:Union[int,List[int]]):
    '''
    noc: (B, H, W, 3)
    bits: N or [N1, N2, N3]
    out: (B, 3*N, H, W) or (B, N1+N2+N3, H, W)
    '''
    if isinstance(bit_cnt, abc.Sequence) and all(bit_cnt[0]==b for b in bit_cnt):
        bit_cnt = bit_cnt[0]
    if not isinstance(bit_cnt, abc.Sequence):
        mod_bits, bits = mod_noc2bits(noc,bit_cnt)    #(B,H,W,3,N)
        mod_bits = mod_bits.flatten(start_dim=-2).permute(0,3,1,2)  #(B,3*N,H,W)
        bits = bits.flatten(start_dim=-2).permute(0,3,1,2)  #(B,3*N,H,W)
        return mod_bits, bits
    else:
        noc = noc.unbind(dim=-1)# (B,H,W)
        mod_bits, bits=mod_noc2bits(noc, bit_cnt)    #(B,H,W,N) for N in bit_cnt
        bits:Tensor = torch.cat(bits,dim=-1)   #(B,H,W,sum(bit_cnt))
        mod_bits:Tensor = torch.cat(mod_bits,dim=-1)   #(B,H,W,sum(bit_cnt))
        return mod_bits.permute(0,3,1,2), bits.permute(0,3,1,2)

def nn_logits2noc(logits:Tensor, bit_cnt:Union[int,List[int]], nearest_lut:Tensor = None):
    '''
    logits: (B,N,H,W)
    return:
    noc: (B,H,W,3)
    '''
    logits = logits.permute(0,2,3,1)    #(B,H,W,N)
    if isinstance(bit_cnt, abc.Sequence) and all(bit_cnt[0]==b for b in bit_cnt):
        bit_cnt = bit_cnt[0]
    if not isinstance(bit_cnt, abc.Sequence):
        noc = mod_logits2noc(logits.reshape(logits.shape[:-1]+(3,-1)), nearest_lut)
    else:
        logits_lst = torch.split(logits,bit_cnt,dim=-1)
        noc = mod_logits2noc(logits_lst, nearest_lut)
    return noc

def nn_logits2noc_with_gt(logits:Tensor, gt_raw_bits:Tensor, bit_cnt:Union[int,List[int]], gt_msk:Tensor):
    '''
    logits: (B,N,H,W)
    gt_bits:(B,N,H,W)
    gt_mask:(B,H,W)
    return:
    noc: (B,H,W,3)
    '''
    logits = logits.permute(0,2,3,1)    #(B,H,W,N)
    gt_raw_bits = gt_raw_bits.permute(0,2,3,1)  #(B,H,W,N)
    if isinstance(bit_cnt, abc.Sequence) and all(bit_cnt[0]==b for b in bit_cnt):
        bit_cnt = bit_cnt[0]
    if not isinstance(bit_cnt, abc.Sequence):
        logits = logits.reshape(logits.shape[:-1]+(3,-1))
        gt_raw_bits = gt_raw_bits.reshape(gt_raw_bits.shape[:-1]+(3,-1))
        gt_msk = gt_msk.unsqueeze(-1)
    else:
        logits = logits.split(bit_cnt,dim=-1)
        gt_raw_bits = gt_raw_bits.split(bit_cnt,dim=-1)
    noc = mod_logits2noc_with_gt(logits, gt_raw_bits, gt_msk)#(B,H,W,3)
    return noc

def mod_noc2bits(numbers:Union[Tensor,List[Tensor]], N:Union[int,List[int]]):
    return mod_noc2bits_bb(numbers, N, black_background=_black_background)

@torch.no_grad()
def mod_noc2bits_bb(numbers:Union[Tensor,List[Tensor]], N:Union[int,List[int]], black_background = True):
    '''
    numbers: (*), normalized to range (-1,1)
    bits:(*,N)
    '''
    if isinstance(numbers, abc.Sequence):
        results = [mod_noc2bits_bb(number,n,black_background) for number, n in zip(numbers,N)]
        return list(zip(*results))

    max_num = 2 ** N - 1
    ints = torch.clamp_((numbers + 1) * (max_num * 0.5), 0, max_num).round_().to(torch.int32)
    mask = 2**torch.arange(N-1,-1,-1,dtype=torch.int32,device=numbers.device)
    bits = ints.unsqueeze(-1).bitwise_and(mask).bool()
    mod_bits = bits.clone()
    mod_bits[...,1:].logical_xor_(bits[...,0:-1])
    if black_background:
        mod_bits[...,0:2].logical_not_()

    return mod_bits, bits

def mod_logits2noc_with_gt(mod_logits:Union[Tensor,List[Tensor]], gt_bits:Union[Tensor,List[Tensor]], gt_msk:Tensor):
    return mod_logits2noc_with_gt_bb(mod_logits, gt_bits, gt_msk, black_background=_black_background)

def mod_logits2noc_with_gt_bb(mod_logits:Union[Tensor,List[Tensor]], gt_raw_bits:Union[Tensor,List[Tensor]], gt_msk:Tensor, black_background = True):
    '''
    logits: (*,N)
    gt_bits:(*,N)
    gt_msk:(*)
    out:(*)
    '''
    val = mod_logits2float_with_gt_bb(mod_logits, gt_raw_bits, gt_msk, black_background)
    if isinstance(val, abc.Sequence):
        val = torch.stack(val, dim=-1)
        max_val = val.new_tensor([2**l.shape[-1]-1 for l in mod_logits])
    else:
        max_val = 2**mod_logits.shape[-1] - 1
    noc = val/(max_val*0.5)-1
    return noc

def mod_logits2float_with_gt(mod_logits:Union[Tensor,List[Tensor]], gt_bits:Union[Tensor,List[Tensor]], gt_msk:Tensor):
    return mod_logits2float_with_gt_bb(mod_logits, gt_bits, gt_msk, black_background=_black_background)

def mod_logits2float_with_gt_bb(mod_logits:Union[Tensor,List[Tensor]], gt_raw_bits:Union[Tensor,List[Tensor]], gt_msk:Tensor, black_background = True):
    '''
    logits: (*,N)
    gt_bits:(*,N)
    gt_msk:(*)
    out:(*)
    '''
    black_factor = -1 if black_background else 1
    if isinstance(mod_logits,abc.Sequence):
        return [mod_logits2float_with_gt_bb_scripted(l, b, gt_msk, black_factor) for l,b in zip(mod_logits, gt_raw_bits)]
    return mod_logits2float_with_gt_bb_scripted(mod_logits, gt_raw_bits, gt_msk, black_factor)


@torch.jit.script
def mod_logits2float_with_gt_bb_scripted(mod_logits:Tensor, gt_raw_bits:Tensor, gt_msk:Tensor, black_factor:int):
    '''
    logits: (*,N)
    gt_bits:(*,N)
    gt_msk:(*)
    out:(*)
    '''

    gt_raw_bits = gt_raw_bits.to(torch.bool, non_blocking=True, copy=True)
    logits_msk = torch.ones_like(mod_logits)
    logits_msk[...,1:].masked_fill_(gt_raw_bits[...,0:-1], -1)
    logits_msk[...,0:2]*=black_factor
    logits = mod_logits * logits_msk    # after this step, logits correspond to binary representations of positive integers
    with torch.no_grad():
        N = logits.shape[-1]
        mask = 2**torch.arange(N-1,-1,-1,dtype=logits.dtype,device=logits.device)
        pred_bits = logits > 0
        out_msk_vals = (pred_bits * mask).sum(-1)

        err_mask = pred_bits.logical_xor_(gt_raw_bits)
        err_mask[...,-1]=True

        err_msb_idx = torch.argmax(err_mask.to(torch.uint8),dim=-1) # torch.argmax returns index of first maximal value if multiple maximum exists
        err_msb_idx_none = err_msb_idx.unsqueeze(-1)
        gt_bits_without_msberr = gt_raw_bits.scatter_(-1, err_msb_idx_none, 0) # 0 is false, 1 is true

    correct_part = (gt_bits_without_msberr * mask).sum(-1)
    in_msk_vals = correct_part + torch.gather(logits, -1, err_msb_idx_none).squeeze(-1).sigmoid()*mask[err_msb_idx]
    val = torch.where(gt_msk, in_msk_vals, out_msk_vals)
    return val


def mod_logits2noc(logits:Union[Tensor,List[Tensor]], nearest_lut:Tensor = None):
    return mod_logits2noc_bb(logits, nearest_lut, black_background=_black_background)

@torch.no_grad()
def mod_logits2noc_bb(logits:Union[Tensor,List[Tensor]], nearest_lut:Tensor = None, black_background = True):
    '''
    logits: (*, 3, N) or [(*, N1), (*, N2), (*, N3)]
    lut: (2^N, 2^N, 2^N, 3) or (2^N1, 2^N2, 2^N3)
    return:
    noc: (*, 3)
    '''
    if nearest_lut is None:
        val = mod_logits2float_bb(logits, black_background)
        if isinstance(val, list):
            val = torch.stack(val, dim=-1)
            max_val = val.new_tensor([2**l.shape[-1]-1 for l in logits])
        else:
            max_val = 2**logits.shape[-1] - 1
        noc = val/(max_val*0.5)-1
    else:
        idx = mod_logits2long_bb(logits, black_background)#(*, 3), dtype=int64
        ix,iy,iz = idx.unbind(dim=-1) if isinstance(idx, Tensor) else idx #(*,), (*,), (*,)
        noc = nearest_lut[ix, iy, iz]
    return noc

# _mod_lut_float = {}
_mod_lut_int32 = {}

def mod_logits2float(mod_logits:Union[Tensor,List[Tensor]]):
    return mod_logits2float_bb(mod_logits, black_background=_black_background)

@torch.no_grad()
def mod_logits2float_bb(mod_logits:Union[Tensor,List[Tensor]], black_background = True):
    '''
    logits: (*,N)
    gt_bits:(*,N)
    out:(*)
    '''
    if isinstance(mod_logits, abc.Sequence):
        return [mod_logits2float_bb(l, black_background) for l in mod_logits]

    N = mod_logits.shape[-1]
    mask = 2**torch.arange(N-1,-1,-1,device=mod_logits.device)

    bin_code_bits = mod_logits>0
    if black_background:
        bin_code_bits[...,0:2].logical_not_()

    bin_code = (mask * bin_code_bits).sum(-1).to(dtype=torch.int64,non_blocking=True)
    lut_key = (N, bin_code.device)
    lut = _mod_lut_int32.get(lut_key,None)
    if lut is None:
        dst = torch.arange(0,2**N,dtype=torch.int64,device=bin_code.device)
        src = dst.bitwise_xor(dst.bitwise_right_shift(1))
        lut = mod_logits.new_empty(dst.size(), dtype=torch.int32)
        lut[src]=dst.to(lut.dtype, non_blocking=True)
        _mod_lut_int32[lut_key]=lut
    val = lut[bin_code]
    lsb_factor = 1 - val.bitwise_and(2)
    val = val.bitwise_and_(-2) + (mod_logits[...,-1] * lsb_factor).sigmoid_()
    return val

def mod_logits2long(mod_logits:Union[Tensor,List[Tensor]]):
    return mod_logits2long_bb(mod_logits, black_background=_black_background)

_mod_lut_long = {}
@torch.no_grad()
def mod_logits2long_bb(mod_logits:Union[Tensor,List[Tensor]], black_background = True):
    '''
    logits: (*,N)
    gt_bits:(*,N)
    out:(*)
    '''
    if isinstance(mod_logits, abc.Sequence):
        return [mod_logits2long_bb(l, black_background) for l in mod_logits]
        
    N = mod_logits.shape[-1]
    mask = 2**torch.arange(N-1,-1,-1,device=mod_logits.device)
    bin_code_bits = mod_logits>0
    if black_background:
        bin_code_bits[...,0:2].logical_not_()
    bin_code = (mask * bin_code_bits).sum(-1).to(dtype=torch.int64,non_blocking=True)
    lut_key = (N, bin_code.device)
    lut = _mod_lut_long.get(lut_key,None)
    if lut is None:
        dst = torch.arange(0,2**N,dtype=torch.int64,device=bin_code.device)
        src = dst.bitwise_xor(dst.bitwise_right_shift(1))
        lut = torch.empty_like(dst)
        lut[src]=dst
        _mod_lut_long[lut_key]=lut
    val = lut[bin_code]
    return val


def calc_bit_count(sizes, max_bits = 7, min_bits = 2):
    max_size = max(sizes)
    delta_bits = [math.log2(size/max_size) for size in sizes]
    num_bits = [max(min_bits,round(max_bits + delta_bit)) for delta_bit in delta_bits]
    return num_bits


