import os.path as osp
import json
import collections
import numpy as np
import os
import itertools

import torch
from .transforms import rotation_conversions as rcvt

def merged_file_list(file_lists):
    def load_list(file_list):
        if isinstance(file_list, str):
            with open(file_list) as f:
                return f.readlines()
        return file_list
    file_lists = [load_list(file_list) for file_list in file_lists]
    return list(itertools.chain.from_iterable(file_lists))

def is_flattened(all_annots):
    return not isinstance(all_annots[0][1], list)


def gen_base_cache_path(file_list_names,
    visib_fract_th = None,
    px_count_visib_th = None,
    scene_ids = None, obj_ids = None,
    cache_dir = '.cache'
    ):
    if not isinstance(file_list_names, list):
        file_list_names = [file_list_names]
    list_names =','.join(sorted([osp.splitext(osp.split(file)[1])[0] for file in file_list_names]))
    n_scene_ids = 'all' if scene_ids is None else str(list(scene_ids))[1:-1]
    n_obj_ids = 'all' if obj_ids is None else str(list(obj_ids))[1:-1]
    n_pix_cnt = '0' if px_count_visib_th is None else str(px_count_visib_th)
    n_vf = '0' if visib_fract_th is None else str(visib_fract_th)
    cache_name = ''.join(f'{list_names}_vf{n_vf}_vp{n_pix_cnt}_s{n_scene_ids}_o{n_obj_ids}'.split())
    return cache_name if cache_dir is None else os.path.join(cache_dir, cache_name)

def load_annots_from_image_list(
    file_list_names,
    dataset_root, 
    flatten = False,
    visib_fract_th = None,
    px_count_visib_th = None,
    scene_ids = None,
    obj_ids = None,
    cache_dir = '.cache',
    gt_keys=['px_count_visib'],
    **kwargs,
    ):
    if isinstance(file_list_names, str):
        file_list_names=[file_list_names]

    all_annots = None
    # try load from cache dir
    if cache_dir:
        base_path = gen_base_cache_path(file_list_names, visib_fract_th, px_count_visib_th, scene_ids, obj_ids, cache_dir=cache_dir)
        cache_path = base_path + '.npy'
        if osp.exists(cache_path):
            d = np.load(cache_path, allow_pickle=True).item()
            all_annots = d['all_annots']

    if all_annots is None:
        # load raw annotations from jsons
        visib_frac_filter = (lambda d : d['visib_fract'] >= visib_fract_th) if visib_fract_th is not None else None
        obj_id_filter = (lambda d: d['obj_id'] in obj_ids) if obj_ids is not None else None
        px_count_visib_filter = (lambda d:d['px_count_visib'] >= px_count_visib_th) if px_count_visib_th is not None else None
        filters = [f for f in [visib_frac_filter, obj_id_filter, px_count_visib_filter] if f is not None]
        instance_filter = (lambda d:all(f(d) for f in filters)) if len(filters)!=0 else (lambda d:True)

        file_list = merged_file_list(file_list_names)

        all_annots = _load_annots_from_image_list(
            file_list, dataset_root,
            gt_keys = gt_keys,
            scene_ids = scene_ids,
            instance_filter = instance_filter,
            **kwargs)

        append_quaternion_state(all_annots)

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_path, dict(all_annots=all_annots))
    if not flatten:
        return all_annots
    else:
        return flatten_annots(all_annots)



def _load_annots_from_image_list(image_list, dataset_root, *, 
    scene_ids = None, 
    instance_filter = lambda d:True,
    flatten = False,
    float_type=np.float32,
    gt_keys = ['cam_R_m2c', 'cam_t_m2c', 'obj_id','inst_idx'],
    ):
    if isinstance(image_list, str):
        with open(image_list) as f:
            frame_list = f.readlines()
    else:
        frame_list = image_list
    frame_list = sorted(frame_list)
    gt_keys = {'cam_R_m2c', 'cam_t_m2c', 'obj_id','inst_idx'}.union(set() if gt_keys is None else gt_keys)
    if scene_ids is None:
        def check_scene_id(scene_id):
            return True
    else:
        scene_ids = set(scene_ids)
        def check_scene_id(scene_id):
            return scene_id in scene_ids
        
    im_width = 640
    im_height = 480

    records = []
    scene_gt_dicts = {}
    scene_gt_info_dicts = {}
    scene_cam_dicts = {}

    # read everything in json into dict
    for frame in frame_list:
        frame = frame.rstrip()
        parts = frame.rsplit('/',3)
        split=parts[-4]
        scene_id = int(parts[-3])
        if not check_scene_id(scene_id):
            continue
        
        scene_key = (scene_id, split)
        im_id = int(parts[-1].split('.')[0])
        if scene_key not in scene_gt_dicts:
            gt_path = osp.join(dataset_root, f'{split}/{scene_id:06d}/scene_gt.json')
            gt_info_path = osp.join(dataset_root, f'{split}/{scene_id:06d}/scene_gt_info.json')
            camera_path = osp.join(dataset_root, f'{split}/{scene_id:06d}/scene_camera.json')
            with open(gt_path) as f:
                scene_gt_dicts[scene_key] = json.load(f)
            with open(gt_info_path) as f:
                scene_gt_info_dicts[scene_key]=json.load(f)
            with open(camera_path) as f:
                scene_cam_dicts[scene_key]=json.load(f)

        im_key = str(im_id)
        annot_cam = scene_cam_dicts[scene_key][im_key]
        annot_gt = scene_gt_dicts[scene_key][im_key]
        annot_gt_info = scene_gt_info_dicts[scene_key][im_key]

        ## functions convert every matrix or vector to numpy type
        def try_to_nparray(mat_elems):
            if not isinstance(mat_elems, collections.abc.Sequence):
                return mat_elems
            length = len(mat_elems)
            nparray = np.array(mat_elems)
            if length == 3:
                nparray = nparray.reshape((3,1)).astype(float_type)
            elif length == 9:
                nparray = nparray.reshape((3,3)).astype(float_type)
            elif length == 4:
                pass
            else:
                assert False, 'unknown length'
            return nparray
            
        def try_to_nparray_dict(the_dict):
            for key, value in the_dict.items():
                the_dict[key]=try_to_nparray(value)
        
        def merge_annot(idx, d0, d1):
            d = d0.copy()
            d.update(d1)
            d['inst_idx'] = idx
            return d


        ## do filtering
        instances = [ merge_annot(idx, gt, gt_info) for idx, (gt, gt_info) in enumerate(zip(annot_gt, annot_gt_info))]
        instances = [inst for inst in instances if instance_filter(inst)]
        if len(instances) == 0:
            continue

        ## do merging
        try_to_nparray_dict(annot_cam)
        for idx, d in enumerate(instances):
            try_to_nparray_dict(d)
            d['inst_idx_filtered'] = idx

        ### per image annot
        record = {
            # "dataset_name": self.name,
            "rgb": frame,
            "split": split,
            "scene_id":scene_id,
            "im_id":im_id,
            "im_wh": (im_width, im_height),
            # "gt":instances,
        }
        instances = [{k:inst[k] for k in gt_keys} for inst in instances]
        record.update(annot_cam)

        if not flatten:
            # record[1] = instances
            records.append((record, instances))
        else:
            for inst in instances:
                records.append((record, inst))
            
    return records

def flatten_annots(annots):
    new_annots = []
    for annot in annots:
        re = annot[0]
        for gt in annot[1]:
            new_annots.append((re,gt))
    return new_annots

def append_quaternion_state(annots, ):
    if is_flattened(annots):
        gts = [a[1] for a in annots]
    else:
        gts = list(itertools.chain.from_iterable([a[1] for a in annots]))
    Rs = np.stack([gt['cam_R_m2c'] for gt in gts])
    ts = np.stack([gt['cam_t_m2c'] for gt in gts])
    quats = rcvt.matrix_to_quaternion(torch.from_numpy(Rs))
    quat_states = torch.cat((quats, torch.from_numpy(ts).squeeze(-1)), dim=-1).numpy()
    for idx, gt in enumerate(gts):
        gt['state'] = quat_states[idx].copy()

def write_mask_paths(all_annots, key = 'mask_visib', flattened = None):
    if flattened is None:
        flattened = is_flattened(all_annots)
    if flattened:
        for annot in all_annots:
            gt = annot[1]
            gt[key]=osp.join(annot[0]['split'], '%06d/'%annot[0]['scene_id'] + key + '/%06d_%06d.png'%(annot[0]['im_id'],gt['inst_idx']))
    else:
        for annot in all_annots:
            for gt in annot[1]:
                gt[key]=osp.join(annot[0]['split'], '%06d/'%annot[0]['scene_id'] + key + '/%06d_%06d.png'%(annot[0]['im_id'],gt['inst_idx']))
