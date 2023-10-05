import numpy as np
import os.path as osp
import json
from operator import itemgetter

def bbox3d_from_scale(noc_scale):
    return np.array([
        [ 1, 1, 1],
        [ 1, 1,-1],
        [ 1,-1, 1],
        [ 1,-1,-1],

        [-1, 1, 1],
        [-1, 1,-1],
        [-1,-1, 1],
        [-1,-1,-1],
        ], dtype=np.float32
    )*noc_scale


def load_composed_model_info(dataset_root, model_dir = 'models', transform_model = False, xform_path = None):
    info_path = osp.join(dataset_root, model_dir, 'models_info.json')
    xform_path = osp.join(dataset_root, 'models_xform.json') if xform_path is None else xform_path
    with open(info_path) as f:
        infos = {int(k):v for k,v in json.load(f).items()}
    if transform_model:
        with open(xform_path) as f:
            xforms = {int(k):v for k,v in json.load(f).items()}
    else:
        xforms = {}

    id_xform = np.eye(4, dtype=np.float32)
    for k, v in infos.items():
        x_info = xforms.get(k,None)
        xform = id_xform if x_info is None else np.array(x_info['xform'],dtype=np.float32).reshape(4,4)
        ori_noc_scale = abs(np.array(itemgetter('min_x','min_y','min_z')(v),dtype=np.float32))
        xformed_noc_scale = ori_noc_scale if x_info is None else np.array(x_info['xformed_noc_scale'], dtype=np.float32)
        v['xform'] = xform
        v['noc_scale_ori'] = ori_noc_scale
        v['noc_scale_xfd'] = xformed_noc_scale
        v['bbox_3d_ori'] = (bbox3d_from_scale(xformed_noc_scale) - xform[:3,3])@xform[:3,:3]
    return infos
            