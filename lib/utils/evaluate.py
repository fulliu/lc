import sys
import numpy as np
import os.path as osp
import logging
import json
import trimesh
import multiprocessing
import copy
import tqdm

import itertools
import lib.utils.error6d as error6d
from lib.bop import load_annots_from_image_list

from tabulate import tabulate

logger = logging.getLogger()


dataset_symmetric_obj_ids = {
    "lm": [3, 7, 10, 11],
    "lmo": [10, 11],
    "tless": list(range(1, 31)),
    "tudl": [],
    "tyol": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21],
    "ruapc": [8, 9, 12, 13],
    "icmi": [1, 2, 6],
    "icbin": [1],
    "itodd": [2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 17, 18, 19, 23, 24, 25, 27, 28],
    "hbs": [10, 12, 18, 29],
    "hb": [6, 10, 11, 12, 13, 14, 18, 24, 29],
    "ycbv": [1, 13, 14, 16, 18, 19, 20, 21],  # bop symmetric objs
    "ycbvposecnn": [13, 16, 19, 20, 21],  # posecnn symmetric objs
}
# object info
ycbv_obj_id2name = {
    1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
    2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
    3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
    4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
    5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
    6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
    7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
    8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
    9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
    10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
    11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
    12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
    13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
    14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
    15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
    16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
    17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
    18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
    19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
    20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
    21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
}

lm_obj_id2name = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}


dataset_obj_id2names = {
    'lm': lm_obj_id2name,
    'lmo': lm_obj_id2name,
    'ycbv':ycbv_obj_id2name,
    'ycbvposecnn':ycbv_obj_id2name
}


def compute_auc_posecnn(errors):
    # NOTE: Adapted from https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_keyframe.m
    errors = errors.copy() * 1e-3 # we use mm, convert to m
    d = np.sort(errors)
    d[d > 0.1] = np.inf
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]
    ids = np.isfinite(d)
    d = d[ids]
    accuracy = accuracy[ids]
    if len(ids) == 0 or ids.sum() == 0:
        return np.nan
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.arange(1, len(mpre))
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids-1]) * mpre[ids]).sum() * 10
    return ap

class Evaluator:
    def __init__(self,
        file_lists, 
        dataset_root,
        dataset_name,
        eval_model_dir = None,
        obj_ids=None,
        symmetric_obj_ids = None,
        use_eval_model = True,
        visib_fract_th = 0,
    ):
        if eval_model_dir is None:
            eval_model_dir = osp.join(dataset_root, 'models_eval' if use_eval_model else 'models')

        if isinstance(file_lists, str):
            file_lists = [file_lists]
            
        records = [load_annots_from_image_list(flist, dataset_root=dataset_root, visib_fract_th=visib_fract_th ) for flist in file_lists]
        records = list(itertools.chain.from_iterable(records))

        with open(osp.join(eval_model_dir,'models_info.json')) as f:
            models_info = json.load(f)
        models_info = {int(key):value for key, value in models_info.items()}

        if obj_ids is not None:
            obj_ids = set(obj_ids)
            id_validater = lambda id : id in obj_ids
        else:
            id_validater = lambda id : True

        models = {key : trimesh.load(osp.join(eval_model_dir, f'obj_{key:06d}.ply')).vertices for key in models_info.keys()}
        max_float = sys.float_info.max
        gts = []
        for record in records:
            instances =  record[1]
            im_info = record[0]
            scene_id = im_info['scene_id']
            im_id = im_info['im_id']
            for inst in instances:
                obj_id = inst['obj_id']
                if not id_validater(obj_id):
                    continue

                R_gt = inst['cam_R_m2c']
                t_gt = inst['cam_t_m2c']
                gts.append(dict(
                    scene_id = scene_id,
                    im_id = im_id,
                    obj_id = obj_id,
                    R_gt = R_gt,
                    t_gt = t_gt,
                    score = 0,
                    R_est = None,
                    t_est = None,
                    time_est = max_float,
                    add = max_float,
                    adi = max_float,
                    re = max_float,
                    te = max_float,
                ))

        self.dataset_name = dataset_name
        self.symmetric_obj_ids = symmetric_obj_ids
        self.models_info = models_info
        self.models = models
        self.gts = gts

    def match_results_to_gt(self, results):
        '''
        find the ground truth corresponding to results, based on key: 'scene_id', 'im_id', 'obj_id'
        '''
        gts = copy.deepcopy(self.gts)
        get_key = lambda result:(result['scene_id'],result['im_id'],result['obj_id'])
        results_map = {get_key(result):result for result in results}
        for gt in gts:
            key = get_key(gt)
            res = results_map.get(key, None)
            if res is not None:
                gt.update(res)
                results_map.pop(key)
                
        matched_gts = gts
        return matched_gts

    def compute_errors(self, matched_gts):
        '''
        object instanced distinguished by 'scene_id', 'im_id', 'obj_id'
        pose parameters with key: 'R_est', 't_est'
        gt parameters with key: 'R_gt', 't_gt'
        :param R_est: 3x3 ndarray with the estimated rotation matrix.
        :param t_est: 3x1 ndarray with the estimated translation vector.
        '''
        errors=[]
        process_count = 6
        chunksize = 10
        with multiprocessing.Pool(process_count) as p:
            res_iter = p.imap(Evaluator.compute_single_error, zip(matched_gts, itertools.repeat(self.models)), chunksize = chunksize)
            if len(matched_gts)>300:
                res_iter = tqdm.tqdm(res_iter, total = len(matched_gts))
            for res in res_iter:
                errors.append(res)
        return errors

    def evaluate(self, outputs, errors = None):
        '''
        object instanced distinguished by 'scene_id', 'im_id', 'obj_id'
        pose parameters with key: 'R_est', 't_est'
        gt parameters with key: 'R_gt', 't_gt'
        :param R_est: 3x3 ndarray with the estimated rotation matrix.
        :param t_est: 3x1 ndarray with the estimated translation vector.
        '''
        if errors is None:
            matched = self.match_results_to_gt(outputs)
            errors = self.compute_errors(matched)

        if self.symmetric_obj_ids is not None:
            sym_obj_ids = self.symmetric_obj_ids
        elif self.dataset_name is not None:
            sym_obj_ids = dataset_symmetric_obj_ids[self.dataset_name]
        else:
            sym_obj_ids = []

        per_obj_score_dict = self.compute_scores(errors, sym_obj_ids)
        return per_obj_score_dict, errors

    def compute_scores(self, all_errors, symmetric_obj_ids):
        symmetric_obj_ids=set(symmetric_obj_ids)
        per_obj_errors = {}
        for e in all_errors:
            per_obj_errors.setdefault(e['obj_id'],[]).append(e)
        per_obj_errors = dict(sorted(per_obj_errors.items()))
        ad_xd_list=[0.1]
        per_obj_score_dict={}
        for obj_id, obj_errors in per_obj_errors.items():
            models_info = self.models_info[obj_id]
            diameter = models_info['diameter']
            is_symmetric = obj_id in symmetric_obj_ids
            add_values = np.array([err['add'] for err in obj_errors])
            adi_values = np.array([err['adi'] for err in obj_errors])
            ad_values = adi_values if is_symmetric else add_values
            sample_count = len(ad_values)
            score_dict={}

            max_cm = 10  # AUC of threshold under 1 cm
            # use average of score at threshold: [0,1,...,10] cm as the auc (as gdrn did)
            # score_dict[f'AUCadd_{max_cm}'] = np.mean([(add_values<(idx+1)*10).sum()/sample_count for idx in range(max_cm)])
            score_dict[f'AUCadi_{max_cm}_p11'] = np.mean([(adi_values<(idx+1)*10).sum()/sample_count for idx in range(max_cm)])
            score_dict[f'AUCad_{max_cm}_p11'] = np.mean([(ad_values<(idx+1)*10).sum()/sample_count for idx in range(max_cm)])
            score_dict[f'AUCadi_{max_cm}_all'] = compute_auc_posecnn(adi_values)
            score_dict[f'AUCad_{max_cm}_all'] = compute_auc_posecnn(ad_values)

            # add-(s) 
            for ad_th in ad_xd_list:
                corrects = (ad_values<ad_th*diameter).sum()
                score = corrects/sample_count
                score_dict[f'add(-s)_{ad_th:.2f}']=score

            for ad_th in ad_xd_list:
                corrects = (adi_values<ad_th*diameter).sum()
                score = corrects/sample_count
                score_dict[f'add-s_{ad_th:.2f}']=score

            for ad_th in ad_xd_list:
                corrects = (add_values<ad_th*diameter).sum()
                score = corrects/sample_count
                score_dict[f'add_{ad_th:.2f}']=score

            per_obj_score_dict[obj_id]=score_dict
        return per_obj_score_dict

    @staticmethod
    def compute_single_error(input_dict_AND_models):
        input_dict, models = input_dict_AND_models
        d = input_dict
        obj_id = d['obj_id']
        R_est = d['R_est']
        if R_est is None:
            return input_dict
        t_est, R_gt, t_gt = d['t_est'], d['R_gt'], d['t_gt']
        pts = models[obj_id]
        input_dict.update(compute_pose_errors(R_est, t_est, R_gt, t_gt, pts))
        return input_dict


def gen_score_table(per_obj_score, obj_id_2_name = None, dataset_name = None, num_digits = 2):
    '''
    per_obj_score:  {obj_id:{"metric_name" : score}}
    
    return:
    per_row: table with one object per row
    per_col: table with one object per col
    '''
    if obj_id_2_name is None:
        obj_id_2_name = dataset_obj_id2names.get(dataset_name,None)
    header = None
    rows=[]
    scores_by_type={}
    for obj_id, score_dict in per_obj_score.items():
        if header is None:
            header=['object']
            header.extend(score_dict.keys())
            rows.append(header)
            for stype in score_dict.keys():
                scores_by_type[stype] = []

        if obj_id_2_name is not None:
            obj_name = obj_id_2_name[obj_id]
        else:
            obj_name = str(obj_id)
        row=[obj_name]
        row.extend(map(lambda v:f'{100*v:.{num_digits}f}', score_dict.values()))
        for stype, svalue in score_dict.items():
            scores_by_type[stype].append(svalue)

        rows.append(row)
    obj_count = len(per_obj_score)
    row=[f'Avg({obj_count})']
    row.extend(map(lambda v: f'{100*np.mean(v):.{num_digits}f}', scores_by_type.values()))
    rows.append(row)
    xposed_rows = np.array(rows, dtype=object).T.tolist()
    per_row:str = tabulate(rows, tablefmt="plain")
    per_col:str = tabulate(xposed_rows, tablefmt="plain"), 
    return per_row, per_col

def compute_pose_errors(R_est, t_est, R_gt, t_gt, pts):
    return dict(
        adi = error6d.adi(R_est, t_est, R_gt, t_gt, pts),
        add = error6d.add(R_est, t_est, R_gt, t_gt, pts),
        re = error6d.re(R_est, R_gt),
        te = error6d.te(t_est, t_gt),
    )

