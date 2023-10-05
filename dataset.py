import gzip, pickle, json, random, math, itertools, logging, os, collections
import cv2, torch

import numpy as np
import imgaug.augmenters as iaa
import os.path as osp
import pycocotools.mask as cocomask
import imageio.v2 as iio
import lib.bop as bop

from tqdm import tqdm
from operator import itemgetter
from torch.utils.data.dataloader import default_collate, default_convert
from model_transform import load_composed_model_info
import symmetry, floatbits

logger = logging.getLogger(__name__)

cv2.setNumThreads(0)

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


class nplist:
    def __init__(self, lst) -> None:
        assert isinstance(lst, collections.abc.Sequence)
        lst = [np.frombuffer(pickle.dumps(l, protocol=-1),dtype=np.uint8) for l in lst]
        mem_range = np.asarray([0,]+[len(x) for x in lst], dtype=np.int64)
        self._len = len(lst)
        self._mem_range = np.cumsum(mem_range)
        self._data = np.concatenate(lst)


    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx >= self._len:
            raise IndexError
        mem_begin, mem_end = self._mem_range[idx:idx+2]
        bytes = memoryview(self._data[mem_begin:mem_end])
        return pickle.loads(bytes)
    
    def buffer_bytes(self):
        return len(self._data)

def _get_affine_transform(center, scale, rot_rad, output_size, shift=np.array([0, 0], dtype=np.float32)):
    """
    align center, keep aspect ratio, fit width
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    def get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def get_dir(src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_result = [src_point[0] * cs - src_point[1] * sn, src_point[0] * sn + src_point[1] * cs]
        return src_result

    src_w = scale[0]
    dst_w, dst_h = output_size

    # rot_rad = np.pi * rot_deg / 180
    src_dir = np.array(get_dir([0, src_w * -0.5], rot_rad))
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    affine = cv2.getAffineTransform(src, dst).astype(np.float32)
    affien_inv = cv2.getAffineTransform(dst, src).astype(np.float32)

    return affine, affien_inv


def filter_annots_by_det(det_path, annots):
    '''
    filter out instances that not detected
    '''
    det_flattened = {}
    with open(det_path) as f:
        for k,v in json.load(f).items():
            per_obj={}
            for d in v:
                per_obj.setdefault(d['obj_id'],[]).append(d)
            for pv in per_obj.values():
                d = sorted(pv,key=lambda d:d['score'])[-1]
                det_flattened[k+'_'+str(d['obj_id'])] = np.array(d['bbox_est'])
    filtered_annots = []
    for annot in annots:
        im_info, inst_info = annot
        (scene_id, im_id), obj_id = itemgetter('scene_id','im_id')(im_info), inst_info['obj_id']
        key=f'{scene_id}/{im_id}_{obj_id}'
        det = det_flattened.get(key, None)
        if det is not None:
            inst_info['bbox_det'] = det
            filtered_annots.append(annot)

    return filtered_annots


def _switch_bg(img_patch, msk, bg_path):
    H,W = img_patch.shape[:2]
    bg_img = np.asarray(iio.imread(bg_path, as_gray=False, pilmode = 'RGB'))
    bg_hw = bg_img.shape[:2]
    msk = msk.astype(np.float32)[...,None]
    bg_roi_w = max(int(random.random()*bg_hw[1]), W)
    bg_roi_h = max(int(random.random()*bg_hw[0]), H)
    bg_roi_l = max(int(random.random()*(bg_hw[1]-bg_roi_w)), 0)
    bg_roi_t = max(int(random.random()*(bg_hw[0]-bg_roi_h)), 0)
    bg_img = cv2.resize(bg_img[bg_roi_t:bg_roi_t+bg_roi_h, bg_roi_l:bg_roi_l+bg_roi_l+bg_roi_w], (W,H))
    img_patch = img_patch*msk+bg_img*(1-msk)
    return img_patch


def _gen_color_aug(cfg):
    augmentations = []
    if cfg.get('use_peper_salt',False):
        augmentations.append(iaa.Sometimes(0.3, iaa.SaltAndPepper(0.05)))
    if cfg.get('use_motion_blur',False):
        augmentations.append(iaa.Sometimes(0.2, iaa.MotionBlur(k=5)))
        
    augmentations=augmentations+[
        iaa.Sometimes(0.5, iaa.CoarseDropout(p=0.1, size_percent=0.05)),
        iaa.Sometimes(0.5, iaa.GaussianBlur((0,1.2))),
        iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3))]

    if cfg.get('use_invert',False):
        augmentations.append(iaa.Sometimes(0.4, iaa.Invert(0.2, per_channel=True))),
        
    augmentations=augmentations+[
        iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
        iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
        iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 2.2), per_channel=0.3)),
    ]
    return iaa.Sequential(augmentations)


def _compress_masks(msk_paths, dataset_root):
    msks = {}
    msk_paths = tqdm(msk_paths)
    for msk_path in msk_paths:
        path = osp.join(dataset_root, msk_path)
        msk = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        rle = cocomask.encode(np.asfortranarray(msk))
        msks[msk_path]=rle
    return msks


class BOP_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, cfg_global, train = True) -> None:
        super().__init__()
        annots_lst = []
        obj_ids = [None] if cfg_global.obj_ids is None else [[oid] for oid in cfg_global.obj_ids]
        cache_dir = cfg_global.get('cache_dir','')
        for list_file in cfg.list_files:
            visib_frac = max(cfg.get('visib_frac',0),0)
            for oid in obj_ids:
                cache_name = bop.gen_base_cache_path(list_file, visib_frac, obj_ids=oid, cache_dir = cache_dir)
                logger.info('loading annotations from image list ...')
                annots =bop.load_annots_from_image_list(list_file, cfg.dataset_root, flatten=True,
                    visib_fract_th = visib_frac, obj_ids=oid, cache_dir = cache_dir, gt_keys=['bbox_visib'])
                if train or True:
                    bop.write_mask_paths(annots, key='mask_visib', flattened=True)
                    msk_cache = cache_name + '_msk.npy'
                    if cache_dir and osp.exists(msk_cache):
                        msks = np.load(msk_cache,allow_pickle=True).item()
                    else:
                        logger.info('compressing masks ...')
                        msks = _compress_masks([a[1]['mask_visib'] for a in annots], cfg.dataset_root)
                        if cache_dir:
                            np.save(msk_cache, msks)
                    for an in annots:
                        im_info, inst_info = an
                        inst_info['mask_visib']=msks[inst_info['mask_visib']]
                        scene_id, im_id, inst_id = im_info['scene_id'],im_info['im_id'],inst_info['inst_idx']
                        z_path = osp.join(cfg.dataset_root, im_info['split'], f'z_crop/{scene_id:06d}/{im_id:06d}_{inst_id:06d}.pkl.gz')
                        inst_info['z_path'] = z_path

                annots_lst.append(annots)

        annots = [list(a) for a in itertools.chain.from_iterable(annots_lst)]
        im_infos = {(im_info['scene_id'],im_info['im_id'],im_info['split']):im_info for im_info, _ in annots}
        for im_info in im_infos.values():
            im_info['rgb'] = osp.join(cfg.dataset_root, im_info['rgb'])
        for ann in annots:
            im_info = ann[0]
            ann[0] = im_infos[(im_info['scene_id'],im_info['im_id'],im_info['split'])]

        if not train and cfg.get('detection',None):
            annots = filter_annots_by_det(cfg.detection, annots)

        np_annots = nplist(annots)
        WHs = np.array([a[0]['im_wh'] for a in annots])
        maxW, maxH = WHs.max(axis=-2)

        if cfg_global.get('sparse_cnt',0) > 0:
            with open(cfg_global.fps, 'rb') as f:
                fps = pickle.load(f)
        else:
            fps = None

        transform_model = cfg_global.get('transform_model', False)

        model_info = load_composed_model_info(cfg.dataset_root, 
                                              transform_model = transform_model,
                                              xform_path=cfg_global.get('transform_path',None))
        max_bit_cnt = cfg_global.get('max_bit_cnt',0)
        if max_bit_cnt>0:
            obj_id = annots[0][1]['obj_id']
            single_obj = all(obj_id == ann[1]['obj_id'] for ann in annots)
            assert single_obj, 'binary mode only allowed in single object mode'
            bit_cnt = floatbits.calc_bit_count(model_info[obj_id]['noc_scale_xfd'].tolist(),max_bits=max_bit_cnt)
            self.bit_cnt = bit_cnt
        else:
            self.bit_cnt = None

        self.sym_obj_ids = []
        if cfg_global.get('sym_aware',cfg.get('sym_aware',False)):
            sym_obj_ids = dataset_symmetric_obj_ids.get(cfg.name,None)
            if sym_obj_ids is None:
                raise RuntimeError("dataset name not found:",cfg.name)
            self.sym_obj_ids = sym_obj_ids

        self.im_msk = np.ones((maxH, maxW),dtype=np.float32)
        self.annots = annots
        self.np_annots = np_annots
        self.fps = fps
        self.sparse_cnt = cfg_global.get('sparse_cnt',0)
        self.model_info = model_info
        self.bg_list= None
        if train and cfg.get('bg_dir',None):
            self.bg_list=sorted([osp.join(cfg.bg_dir, fname) for fname in os.listdir(cfg.bg_dir)])

        self.debug = cfg_global.get('debug', False)
        self.valid_pix_cnt_th = cfg.get('valid_pix_cnt_th', 100)
        self.transform_model = transform_model
        self.cfg = cfg
        self.cfg_global = cfg_global
        self.net_input_wh = cfg_global.get('net_input_wh', cfg.get('net_input_wh', None))
        self.net_output_wh = cfg_global.get('net_output_wh', cfg.get('net_output_wh', None))
        self.training = train
        self.augmenter = _gen_color_aug(cfg)

        mask_interp = cfg.get('mask_interp', 'linear')
        if mask_interp.lower()=='linear':
            mask_interp = cv2.INTER_LINEAR
        elif mask_interp.lower()=='nearest':
            mask_interp = cv2.INTER_NEAREST
        self.mask_interp = mask_interp

    def _get_homo_with_depth(self, annot, size_hw, fill_hole = True):
        im_info, inst_info = annot
        path = inst_info['z_path']
        with gzip.open(path,'rb') as f:
            z_info = pickle.load(f)

        homo_z = np.zeros(size_hw+(3,), dtype=np.float32)
        msk_full = np.zeros(size_hw, dtype=np.float32)
        (x1, y1, x2, y2), z_crop, z_max, z_min = itemgetter('xyxy','z_crop','z_max','z_min')(z_info)
        if z_max==0:
            raise RuntimeError("No target in ROI")

        msk = z_crop!=0
        v, u = np.nonzero(msk)

        z_crop = cv2.medianBlur(z_crop, ksize=3) if fill_hole else z_crop
        z = (z_crop[msk]-1).astype(np.float32)*((z_max - z_min)/65534)+z_min

        u1,v1 = u+x1, v+y1
        ones = np.ones_like(v)
        homo = np.stack((u1+0.5,v1+0.5,ones),axis=-1) * z[...,None]

        msk_full[y1:y2+1, x1:x2+1]=msk
        homo_z[y1:,x1:][v, u] = homo
        return homo_z, msk_full
    
    def _aug_bbox(self, bbox_xyxy, im_H, im_W):
        cfg = self.cfg
        x1, y1, x2, y2 = bbox_xyxy
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bh = y2 - y1
        bw = x2 - x1

        scale_ratio = 1 + cfg.dzi_scale_ratio * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
        shift_ratio = cfg.dzi_shift_ratio * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
        bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
        scale = max(y2 - y1, x2 - x1) * scale_ratio * cfg.dzi_pad_scale

        scale = min(scale, max(im_H, im_W)) * 1.0
        return bbox_center, scale

    def __len__(self,):
        return len(self.np_annots)

    def __getitem__(self, index):
        if self.training:
            while True:
                data = self._get_single_item(index)
                if data is not None:
                    return data
                index = np.random.choice(len(self))
        else:
            return self._get_single_item(index)

    def file_list(self,):
        annots = self.annots
        flist = []
        for im_info, inst_info in annots:
            rgb_path = im_info['rgb']
            z_path = inst_info['z_path']
            flist.append([rgb_path, z_path])
        return flist

    def collate_fn(self,):
        def sym_collate(batch):
            if 'Rt_candi' not in batch[0]:
                return default_collate(batch)
            dbatch = {}
            for b in batch:
                dbatch.setdefault(len(b['Rt_candi']),[]).append(b)
            dbatch = sorted(dbatch.items())

            batched_candis = [torch.stack(default_convert([c.pop('Rt_candi') for c in candis])) for _,candis in dbatch]
            batched = default_collate(list(itertools.chain.from_iterable((b[1] for b in dbatch))))
            batched['Rt_candi'] = batched_candis
            return batched
            
        return sym_collate

    def _get_single_item(self, index):
        cfg, cfg_global, annot = self.cfg, self.cfg_global, self.np_annots[index]
        im_info, inst_info = annot
        rgb_path = im_info['rgb']
        rgb = np.asarray(iio.imread(rgb_path, as_gray=False, pilmode = 'RGB'))
        H, W, C = rgb.shape
        no_aug = self.debug or not self.training
        cam_K = im_info['cam_K']
        obj_id, R, t = itemgetter('obj_id','cam_R_m2c', 'cam_t_m2c')(inst_info)
        t=t[...,0] 
        m_info = self.model_info[obj_id]

        msk_visib = cocomask.decode(inst_info['mask_visib']).astype(np.float32)
        if self.training:
            homo_z, mask_full = self._get_homo_with_depth(annot, (H, W), False)
        else:
            homo_z, mask_full, msk_visib = np.ones((H,W,3),dtype=np.float32), np.ones((H,W),dtype=np.float32), np.zeros_like(msk_visib)

        bbox_xywh = inst_info['bbox_visib']
        if not self.training:
            if 'bbox_det' in inst_info:
                bbox_xywh = inst_info['bbox_det']
            else:
                logger.warning('using ground truth bounding box when testing')

        bbox_xyxy = np.concatenate((bbox_xywh[:2], bbox_xywh[:2]+bbox_xywh[2:]),axis=-1)

        if not no_aug:
            bbox_center, scale = self._aug_bbox(bbox_xyxy, H, W)
        else:
            bbox_center, scale = (bbox_xyxy[:2]+bbox_xyxy[2:])*0.5, float(max(bbox_xywh[2],bbox_xywh[3], 1)) * cfg.dzi_pad_scale
        clip_box_xywh = np.zeros(4,dtype=np.float32)
        clip_box_xywh[:2], clip_box_xywh[2:4] = bbox_center - scale * 0.5, scale

        net_output_wh, net_input_wh = self.net_output_wh, self.net_input_wh
        rotate = np.random.random()*4*math.pi if not no_aug and np.random.random()<cfg.rotate_prob else 0

        if obj_id in self.sym_obj_ids:
            Rt_candi = symmetry.symmetry_pose_candidates(R,t,m_info)
        else:
            Rt_candi = np.concatenate((R,t[...,None]),axis=-1)[None]

        out_affine, inv_out_affine = _get_affine_transform(bbox_center, scale, rotate, net_output_wh)
        in_affine, inv_in_affine = _get_affine_transform(bbox_center, scale, rotate, net_input_wh)
        rgb_in = cv2.warpAffine(rgb, in_affine, net_input_wh, flags = cv2.INTER_LINEAR)

        if not no_aug and (np.random.random()<cfg.switch_bg_prob or im_info['split'] in {'imgn'}):
            msk_in = cv2.warpAffine(msk_visib, in_affine, net_input_wh, flags=cv2.INTER_LINEAR)
            rgb_in = _switch_bg(rgb_in, msk_in, np.random.choice(self.bg_list))
        if not no_aug and np.random.random()<cfg.pixel_aug_prob:
            if rgb_in.dtype == np.float32:
                rgb_in = np.round(rgb_in).astype(np.uint8)
            rgb_in = self.augmenter(image=rgb_in)

        affine33 = np.eye(3,dtype=np.float32)
        affine33[:2] = out_affine
        out_K = affine33 @ cam_K

        msk_vis = cv2.warpAffine(msk_visib, out_affine, dsize=net_output_wh, flags=self.mask_interp)
        msk_noc = cv2.warpAffine(msk_visib, out_affine, dsize=net_output_wh, flags=cv2.INTER_NEAREST)>0.5
        valid_positions = msk_noc.nonzero()
        valid_cnt = len(valid_positions[0])
        if self.training and valid_cnt < self.valid_pix_cnt_th:
            return None

        check_pt_cnt = 256
        if valid_cnt >= self.valid_pix_cnt_th:
            selected_point, sym_ck_idx = 0, []
            while selected_point < check_pt_cnt:
                new_pt_cnt = min(valid_cnt, check_pt_cnt - selected_point)
                selected_point += new_pt_cnt
                sym_ck_idx.append(np.random.choice(valid_cnt, new_pt_cnt, replace = False))
            sym_ck_idx = np.concatenate(sym_ck_idx)
            sym_ck_pts2d = np.stack((valid_positions[1][sym_ck_idx], valid_positions[0][sym_ck_idx])).T
        else:
            sym_ck_pts2d = np.full((check_pt_cnt,2),-1)

        homo_z_out = cv2.warpAffine(homo_z, out_affine, dsize=net_output_wh, flags=cv2.INTER_NEAREST)
        noc_scale_xfd = m_info['noc_scale_ori']
        model_transform = m_info['xform']
        bbox_3d = m_info['bbox_3d_ori']
        noc_scale_xfd, noc_scale_ori, model_transform, bbox_3d, diameter = \
            itemgetter('noc_scale_xfd', 'noc_scale_ori', 'xform', 'bbox_3d_ori', 'diameter')(m_info)
        if self.training:
            blob = {
                'rgb_in':torch.from_numpy(rgb_in).permute(2,0,1).to(torch.float32).div(255),
                'noc_scale': noc_scale_xfd,
                'noc_scale_ori':noc_scale_ori,
                'out_pix_scale': scale/net_output_wh[0],

                'msk_vis':msk_vis,
                'msk_noc':msk_noc,

                'homo_z_out':homo_z_out,
                'K_no_aug':cam_K,
                'R_no_aug':R,
                't_no_aug':t,
                'sym_ck_pts2d':sym_ck_pts2d,
                'Rt_candi':Rt_candi.astype(np.float32),
                'bbox_3d':bbox_3d, 
                'diameter':diameter,
                'out_K': out_K,

                'obj_id': obj_id,
                'im_id':im_info['im_id'],
                'scene_id':im_info['scene_id'],
            }
        else:
            blob = {
                'rgb_in':torch.from_numpy(rgb_in).permute(2,0,1).to(torch.float32).div(255),
                'noc_scale': noc_scale_xfd,
                'noc_scale_ori':noc_scale_ori,
                'out_pix_scale': scale/net_output_wh[0],

                'out_K': out_K,
                'obj_id': obj_id,
                'im_id':im_info['im_id'],
                'scene_id':im_info['scene_id'],
            }
        if self.sparse_cnt > 0:
            blob['pts3d'] = self.fps[obj_id][:self.sparse_cnt]
        if self.transform_model:
            blob['model_transform'] = model_transform

        return blob






