import torch
import numpy as np
import ptnet, train, floatbits
import os, argparse, logging, copy

import torchvision.transforms as transforms
import lib.transforms as xforms

from utils import xfer_to
from mmcv import Config, DictAction
from tqdm import tqdm
from losses import dense_pnp_matching_from_xyz, nn_out_to_xyz
from collections import abc
from operator import itemgetter
from lib.pnp import cer_solver, cv2_solver
from lib.utils import evaluate
from lib.utils import checkpoint
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Union
from lib.utils.setup_logger import setup_logger
from tensorboardX import SummaryWriter
import cv2

logger = logging.getLogger(__name__)
    
def to_np(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, str):
        return input
    elif isinstance(input, abc.Sequence):
        return [to_np(i) for i in input]
    elif isinstance(input, abc.Mapping):
        return {k:to_np(v) for k,v in input.items()}
    return input


def quantile_msk(den_inv_std2d:Tensor, quantile:Union[float, Tensor]):
    weights = den_inv_std2d.sum(dim=-1)
    q = torch.quantile(weights, quantile, dim=1, keepdims=True)
    if isinstance(quantile, Tensor):
        q = torch.diagonal(q, dim1=0,dim2=1).mT
    msk = weights >= q
    return msk

@torch.no_grad()
def solve_pnp(cfg, out_dict, gt_dict):
    if 'pts2d' not in out_dict:
        return solve_pnp_dense(cfg, out_dict, gt_dict)

    K, pts3d = itemgetter('out_K', 'pts3d',)(gt_dict)
    pts2d, pts2d_std = itemgetter('pts2d','pts2d_std')(out_dict)
    inv_cov2d = 1/(pts2d_std**2)

    reprojectionError=2
    if cfg.get('rel_reproj_err', False):
        reprojectionError = 2 / gt_dict['out_pix_scale']
    invalids, cv_states, inliers = cv2_solver.solve(K, pts3d, pts2d, reprojectionError=reprojectionError)
    cv_states = [st.to(torch.float32) for st in cv_states]
    res_list = [('ransac', torch.stack(cv_states))]
    weighted = cer_solver.solve(K, pts3d.unbind(0), pts2d, inv_cov2d, cv_states, num_workers=4,filter_input_nan=True)[1]
    res_list.append(('weighted', weighted))
    return dict(res_list[::-1])


@torch.no_grad()
def solve_pnp_dense(cfg, out_dict, gt_dict):
    K = gt_dict['out_K']
    
    seg_msk = torch.sigmoid(out_dict['msk_vis_logits']) > cfg.get('seg_thresh',0.5)
    sample = cfg.get('dense_sample',2)

    if 'xyz_noc' in out_dict:
        nn_out = out_dict['xyz_noc']
    elif 'xyz_noc_bin' in out_dict:
        nn_out = out_dict['xyz_noc_bin']
    
    xyz_out = nn_out_to_xyz(
        nn_out, gt_dict['noc_scale'],
        model_transform = gt_dict.get('model_transform', None),
        bit_cnt = gt_dict.get('bit_cnt', None),
        inference=True)

    xyz_weight_logits:Tensor = out_dict['xyz_weight_logits']
    xyz_weights_scale = out_dict['xyz_weights_scale']
    weight_scale_dim = xyz_weights_scale.shape[-3]
    xyz_weights_raw:Tensor = xyz_weight_logits.reshape(xyz_weight_logits.shape[:-3]+(weight_scale_dim, -1)).softmax(dim=-1)
    xyz_weights = xyz_weights_raw.reshape_as(xyz_weight_logits) * xyz_weights_scale

    noc_scale_dummy = None
    den_pts2d, den_inv_std2d, den_pts3d, seg_valid_mask =\
        dense_pnp_matching_from_xyz(xyz_out.permute(0,3,1,2), xyz_weights, seg_msk.squeeze(-3), noc_scale_dummy, sample, top_left=(0,0))

    den_inv_cov2d = den_inv_std2d ** 2

    if cfg.dense_point_select == 'mask':
        den_valid_msk = seg_valid_mask
    if cfg.dense_point_select == 'quantile':
        den_valid_msk = quantile_msk(den_inv_std2d, cfg.quantile)
    if cfg.dense_point_select == 'quantile_in_mask':
        vis_ratio = seg_valid_mask.float().mean(dim=-1)
        quantile = 1 - (1-cfg.quantile) * vis_ratio
        den_valid_msk = quantile_msk(den_inv_std2d * seg_valid_mask[...,None], quantile) * seg_valid_mask

    valid_index_lst = [v.nonzero()[:,0] for v in den_valid_msk]

    def min_len_index(idx:Tensor, src_len, min_len):
        n, dev, dtype = min_len - len(idx), idx.device, idx.dtype
        return idx if n<=0 else torch.cat((idx, torch.from_numpy(np.random.choice(src_len, n, n>src_len)).to(dev, dtype)))

    def select_valid(src_tensors, idx, min_cnt = 4):
        return [t[min_len_index(i,len(t), min_cnt)] if len(t)>min_cnt else t for t, i in zip(src_tensors, idx)]

    reprojectionError=3
    if cfg.get('rel_reproj_err', False):
        reprojectionError = 2 / gt_dict['out_pix_scale']
        
    pts3d, pts2d = (select_valid(t, valid_index_lst) for t in (den_pts3d, den_pts2d))
    invalids, cv_states, inliers = cv2_solver.solve(K, pts3d, pts2d, reprojectionError=reprojectionError)
    cv_states = [st.to(torch.float32) for st in cv_states]
    
    solvers = cfg.solvers
    res_list = []
    if 'weighted' in solvers:
        inv_cov2d = select_valid(den_inv_cov2d, valid_index_lst)
        weighted = cer_solver.solve(K, pts3d, pts2d, inv_cov2d, cv_states, num_workers=4,filter_input_nan=True)[1]
        res_list.append(('weighted', weighted))

    if 'weighted_filtered' in solvers:
        filtered_valid_idx = select_valid(valid_index_lst, inliers)
        pts3d, pts2d, inv_cov2d = (select_valid(t, filtered_valid_idx) for t in (den_pts3d, den_pts2d, den_inv_cov2d))
        weighted_filtered = cer_solver.solve(K, pts3d, pts2d, inv_cov2d, cv_states, num_workers=4,filter_input_nan=True)[1]
        res_list.append(('weighted-filtered', weighted_filtered))

    return dict(res_list[::-1])


def get_evaluator(cfg_dataset, cfg_global):
    list_files = cfg_dataset.list_files
    dataset_root = cfg_dataset.dataset_root
    obj_ids = cfg_global.get('obj_ids', None)
    dataset_name = cfg_dataset.get('name', None)
    eval_model = cfg_dataset.get('use_eval_model',True)
    eval_model_dir = cfg_dataset.get('eval_model_dir', os.path.join(dataset_root,'models_eval' if eval_model else 'models'))
    evaluator = evaluate.Evaluator(list_files, dataset_root, dataset_name, eval_model_dir, obj_ids, visib_fract_th=cfg_dataset.visib_frac)
    return evaluator


@torch.no_grad()
def test(args, cfg, model:torch.nn.Module, dataloader:DataLoader, evaluator:evaluate.Evaluator, score_key = 'add(-s)_0.10'):
    transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    training = model.training
    model.eval()
    pbar = tqdm(dataloader)
    results = {}
    solver_cfg = cfg.pnp_solver
    bit_cnt = dataloader.dataset.bit_cnt
    for i, blob_cpu in enumerate(pbar):
        blob_dev = xfer_to(blob_cpu, device=args.device)
        if bit_cnt is not None:
            blob_dev['bit_cnt'] = bit_cnt
        img_in = transform(blob_dev['rgb_in'])
        out_dict = model(img_in)

        states = solve_pnp(solver_cfg, out_dict, blob_dev)

        im_ids, scene_ids, obj_ids = itemgetter('im_id','scene_id', 'obj_id')(blob_dev)
        blob = [None, im_ids.tolist(), scene_ids.tolist(), obj_ids.tolist()]
        for solver_name, res_states in states.items():
            blob[0]=res_states
            res_lst = results.setdefault(solver_name,[])
            for w_st, im_id, scene_id, obj_id in zip(*blob):
                R, t = xforms.quaternion_rep_to_RT(w_st)
                res_lst.append(dict(obj_id = obj_id, im_id = im_id, scene_id = scene_id, R_est=R, t_est = t))

    outputs = {}
    for solver_name, res_lst in results.items():
        per_obj_score, errors = evaluator.evaluate(to_np(res_lst))
        tables = evaluate.gen_score_table(per_obj_score, dataset_name = evaluator.dataset_name)
        avg_score = sum(ds[score_key] for ds in per_obj_score.values())/len(per_obj_score)
        outputs[solver_name] = dict(avg_score = avg_score, per_obj_score=per_obj_score, errors=errors, tables=tables)

    model.train(training)
    
    return outputs


def csv_from_results(results):
    score = 1
    time = -1
    csvs = {}
    for name, res in results.items():
        lines = []
        for single_res in res['errors']:
            obj_id, im_id, scene_id, R, t = itemgetter('obj_id', 'im_id', 'scene_id', 'R_est', 't_est')(single_res)
            if R is None:
                continue
            flatten_R = R.flatten().tolist()
            flatten_t = t.flatten().tolist()
            str_R = ' '.join([str(v) for v in flatten_R])
            str_t = ' '.join([str(v) for v in flatten_t])
            line = ','.join([str(v) for v in (scene_id, im_id, obj_id, score, str_R, str_t, time)])+'\n'
            lines.append(line)
        csvs[name] = ''.join(lines)
    return csvs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--obj',type=int, nargs='+', required=True)

    parser.add_argument('--opts', nargs='+', action=DictAction)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if not os.path.exists(args.weight):
        print(f'weight file: "{args.weight}" not found.')
        exit(-1)

    cfg_name = os.path.split(args.config)[1].rsplit('.',maxsplit=1)[0]
    cfg = Config.fromfile(args.config)
    cfg.obj_ids = args.obj

    if args.opts is not None:
        cfg.merge_from_dict(args.opts)

    log_name = cfg_name, cfg.train_dataset.name, 'test' , ','.join(map(str,cfg.obj_ids))+'.log'
    log_path = os.path.realpath(os.path.join(args.output,'_'.join(log_name)))
    setup_logger(log_path)

    evaluator = get_evaluator(cfg.test_dataset, cfg)

    dataset, dataloader = train.get_dataloader(cfg.test_dataset, cfg.dataloader, cfg, train=False, shuffle=False)
    total_bit_cnt = 0 if dataset.bit_cnt is None else sum(dataset.bit_cnt)

    model = ptnet.ptnet(cfg.model, cfg, total_bit_cnt=total_bit_cnt)
    checkpoint.load_model(args.weight, model,strict=False)

    model = model.to(args.device)
    floatbits.set_black_background(cfg.get('black_background',False))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)
    cv2.setRNGSeed(0)

    test_res = test(args, cfg, model, dataloader, evaluator)

    csv = csv_from_results(test_res)

    table = ''
    for solver_name, res_dict in test_res.items():
        table+='\n'+solver_name+'\n'+res_dict['tables'][0]+'\n'
    logger.info(table)
    
    for k, v in csv.items():
        csv_name = '_'.join([cfg_name + '-'+ k, cfg.train_dataset.name, 'test' , ','.join(map(str,cfg.obj_ids))+'.csv'])
        csv_path = os.path.join(args.output, csv_name)
        with open(csv_path,'wt') as f:
            f.write(v)
        
