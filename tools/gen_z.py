from __future__ import division, print_function

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
import shutil
import argparse

import mmcv
import numpy as np
from tqdm import tqdm

cur_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, cur_dir)

from lib.meshrenderer.meshrenderer_phong import Renderer
import gzip

idx2class_lm = {
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
idx2class_lmo = {
    1: "ape",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
}
idx2class_ycbv = {
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


IM_H = 480
IM_W = 640
near = 0.01
far = 6.5


def mask2bbox_xyxy(mask):
    """NOTE: the bottom right point is included"""
    ys, xs = np.nonzero(mask)[:2]
    bb_tl = [xs.min(), ys.min()]
    bb_br = [xs.max(), ys.max()]
    return [bb_tl[0], bb_tl[1], bb_br[0], bb_br[1]]

class XyzGen(object):
    def __init__(self, data_dir, xyz_root, model_paths, cls_indexes, dataset):
        sel_scene_ids = sorted([int(sc) for sc in os.listdir(data_dir) if os.path.exists(os.path.join(data_dir,sc,'scene_gt.json'))])
        if dataset=='lmo' and data_dir.split('/')[-1]=='train_real':
            sel_scene_ids = [sid for sid in sel_scene_ids if sid in [1,5,6,8,9,10,11,12]]
        print("scene ids: ", sel_scene_ids)
        self.xyz_root = xyz_root
        self.sel_scene_ids = sel_scene_ids
        self.data_root = data_dir
        self.model_paths = model_paths
        self.cls_indexes = cls_indexes
        self.renderer = None

    def get_renderer(self):
        if self.renderer is None:
            # self.renderer = Renderer(
            #     self.model_paths, vertex_tmp_store_folder=osp.join(PROJ_ROOT, ".cache"), vertex_scale=0.001
            # )
            self.renderer = Renderer(
                self.model_paths, vertex_tmp_store_folder=osp.join(cur_dir, ".cache"), vertex_scale=0.001
            )
        return self.renderer

    def main(self, scene_range, args):
        sel_scene_ids = self.sel_scene_ids
        data_root = self.data_root
        xyz_root = self.xyz_root
        cls_indexes = self.cls_indexes
        for scene_id in tqdm(sel_scene_ids):
            if scene_id<scene_range[0] or scene_id > scene_range[1]:
                continue

            scene_path = osp.join(xyz_root, f"{scene_id:06d}")
            if osp.exists(scene_path) and osp.isdir(scene_path):
                if not args.remove_existing:
                    print(scene_path+' already exists, specify --remove_existing if you want to delete them')
                    exit(0)
                print("removding old scene: ", scene_path)
                shutil.rmtree(scene_path)
                print("Old scene path: ", scene_path, " deleted!")
                
            print("scene: {}".format(scene_id))
            scene_root = osp.join(data_root, f"{scene_id:06d}")
            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                K = np.array(cam_dict[str_im_id]['cam_K'], dtype=np.float32).reshape(3,3)

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in idx2class:
                        continue

                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                    # pose = np.hstack([R, t.reshape(3, 1)])

                    save_path = osp.join(
                        xyz_root,
                        f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}.pkl",
                    )
                    # if osp.exists(save_path) and osp.getsize(save_path) > 0:
                    #     continue

                    render_obj_id = cls_indexes.index(obj_id)  # 0-based
                    bgr_gl, depth_gl = self.get_renderer().render(render_obj_id, IM_W, IM_H, K, R, t, near, far)
                    mask = (depth_gl > 0).astype("uint8")

                    if mask.sum() == 0:  # NOTE: this should be ignored at training phase
                        print(
                            f"not visible, scene {scene_id}, im {int_im_id} obj {idx2class[obj_id]} {obj_id}"
                        )
                        print(f"{save_path}")
                        z_info = {
                            "z_crop": np.zeros((IM_H, IM_W), dtype=np.uint16),
                            "xyxy": [0, 0, IM_W - 1, IM_H - 1],
                            "z_max":np.zeros(1, dtype=np.float32),
                            "z_min":np.zeros(1, dtype=np.float32),
                        }
                        
                    else:
                        x1, y1, x2, y2 = mask2bbox_xyxy(mask)
                        msk = mask!=0
                        # xyz_np = misc.calc_xyz_bp_fast(depth_gl, R, t, K)
                        z_np_view = depth_gl[msk]
                        z_min = z_np_view.min()
                        z_max = z_np_view.max()
                        depth_gl[msk]=(z_np_view[...]-z_min)/(z_max - z_min + 1e-30) * 65534 + 1
                        z_np_crop = depth_gl[y1 : y2 + 1, x1 : x2 + 1]
                        z_info = {
                            "z_crop": z_np_crop.round().astype(np.uint16),  # save disk space w/o performance drop
                            "xyxy": [x1, y1, x2, y2],
                            "z_max":z_max * 1000,
                            "z_min":z_min * 1000,
                        }

                    if True:
                        mmcv.mkdir_or_exist(osp.dirname(save_path))
                        with gzip.open(save_path+'.gz', 'wb') as f:
                            mmcv.dump(z_info, f, file_format='pkl')
        if self.renderer is not None:
            self.renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gen z_crop")
    parser.add_argument("--scene", type=int)
    parser.add_argument("--dataset", type=str, choices=['ycbv','lmo'], required=True)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=sys.maxsize)
    parser.add_argument("--remove_existing", action="store_true")
    parser.add_argument("--data_dir",type=str, required=True)
    parser.add_argument("--xyz_root",type=str)
    parser.add_argument("--model_dir",type=str)
    args = parser.parse_args()

    # data_dir = "datasets/BOP_DATASETS/ycbv/train_pbr"
    # xyz_root = "datasets/BOP_DATASETS/ycbv/train_pbr/z_crop"
    # model_dir = "datasets/BOP_DATASETS/ycbv/models"

    if args.xyz_root is None:
        args.xyz_root = os.path.join(args.data_dir,'z_crop')
    if args.model_dir is None:
        args.model_dir = os.path.join(args.data_dir,'../models')

    if args.scene is not None:
        args.begin = args.scene
        args.end = args.scene + 1

    id2c = {'ycbv':idx2class_ycbv,'lmo':idx2class_lmo}
    idx2class = id2c[args.dataset]
    class2idx = {_name: _id for _id, _name in idx2class.items()}

    cls_indexes = sorted(idx2class.keys())
    model_paths = [osp.join(args.model_dir, f"obj_{obj_id:06d}.ply") for obj_id in cls_indexes]

    xyz_gen = XyzGen(args.data_dir, args.xyz_root, model_paths, cls_indexes, dataset=args.dataset)
    xyz_gen.main([args.begin, args.end], args)
    print("scene", args.scene)
