import torch
import ptnet
import collections
import collections.abc
from torch.utils.data import DataLoader
from copy import deepcopy

from dataset import BOP_Dataset
from lib.optim.lr_scheduler import flat_and_anneal_lr_scheduler
from lib.optim.ranger import Ranger


string_classes = (str, bytes)

def xfer_to(pack, device, non_blocking = True):
    if isinstance(pack, torch.Tensor):
        return pack.to(device=device, non_blocking = non_blocking)

    if isinstance(pack, collections.abc.Mapping):
        return {key:xfer_to(value, device, non_blocking = non_blocking) for key, value in pack.items()}

    pack_type = type(pack)
    if isinstance(pack, tuple) and hasattr(pack, '_fields'):  # namedtuple
        return pack_type(*(xfer_to(sub_pack, device, non_blocking = non_blocking) for sub_pack in pack))

    if isinstance(pack, collections.abc.Sequence) and not isinstance(pack, string_classes):
        return [xfer_to(sub_pack, device, non_blocking = non_blocking) for sub_pack in pack]

    return pack

def get_dataloader(cfg_dataset, cfg_dataloader, cfg_global, train = True, shuffle = None, pin_memory = True, persistent_workers = False):

    dataset = BOP_Dataset(cfg_dataset, cfg_global, train = train)
    batch_size, num_workers = cfg_dataloader.batch_size, cfg_dataloader.num_workers

    shuffle = shuffle if shuffle is not None else train
    collate_fn = dataset.collate_fn()

    dataloader = DataLoader(dataset, batch_size, shuffle = shuffle, 
        num_workers = num_workers, collate_fn=collate_fn, pin_memory = pin_memory,
        persistent_workers = persistent_workers and num_workers > 0)

    return dataset, dataloader

def get_model_optim(cfg, **kwargs):
    model = ptnet.ptnet(cfg.model, cfg, **kwargs)
    optim_cfg = cfg.optimizer
    type_name = optim_cfg.type.lower()
    if type_name=='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr = optim_cfg.lr, weight_decay=optim_cfg.wd)
    elif type_name == 'ranger':
        optimizer = Ranger(model.parameters(),lr = optim_cfg.lr, weight_decay=optim_cfg.wd)
    return model, optimizer

def build_lr_scheduler(
    cfg, optimizer: torch.optim.Optimizer, total_iters: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a LR scheduler from config."""
    name = cfg.name
    if name.lower() == "flat_and_anneal":
        return flat_and_anneal_lr_scheduler(
            optimizer,
            total_iters=total_iters,  # NOTE: TOTAL_EPOCHS * len(train_loader)
            warmup_factor=cfg.warmup_factor,
            warmup_iters=cfg.warmup_iters,
            warmup_method=cfg.warmup_method,  # default "linear"
            anneal_method=cfg.anneal_method,
            anneal_point=cfg.anneal_point,  # default 0.72
            steps=cfg.get("rel_steps", [2 / 3.0, 8 / 9.0]),  # default [2/3., 8/9.],  use relative decay steps
            target_lr_factor=cfg.get("target_lr_factor", 0),
            poly_power=cfg.get("poly_power", 1.0),
            step_gamma=cfg.step_gamma,  # default 0.1
        )
    elif name.lower() == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer,lambda v:1.0)
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

class MultiLoader:
    def __init__(self, cfg) -> None:
        batch_size = cfg.dataloader.batch_size
        loader_cfg = deepcopy(cfg.dataloader)
        if 'train_dataset_1' in cfg:
            d1_cfg = deepcopy(cfg.train_dataset_1)
            for k in cfg.train_dataset:
                if k not in d1_cfg:
                    d1_cfg[k] = cfg.train_dataset[k]
            d1_batch_size = int(batch_size * cfg.train_dataset_1.ratio)
            loader_cfg.batch_size = d1_batch_size

            self.train_set_1, self.train_loader_1 = get_dataloader(d1_cfg, loader_cfg, cfg, persistent_workers=True)
            self.iter_1 = iter(self.train_loader_1)

            loader_cfg.batch_size = batch_size - d1_batch_size
        else:
            self.train_set_1, self.train_loader_1, self.iter_1 = None, None, None
        
        self.train_set_0, self.train_loader_0 = get_dataloader(cfg.train_dataset, loader_cfg, cfg, persistent_workers=True) 
        self.iter_0 = iter(self.train_loader_0)
    
    def _composite_batched(self, blob_0, blob_1, device, non_blocking = True):
        pack = blob_0
        if isinstance(pack, torch.Tensor):
            l0,l1 = len(blob_0),len(blob_1)
            out = pack.new_empty((l0+l1,)+blob_0.shape[1:],device=device)
            out[:l0].copy_(blob_0,non_blocking)
            out[l0:].copy_(blob_1,non_blocking)
            return out

        if isinstance(pack, collections.abc.Mapping):
            composed = {key:self._composite_batched(value, blob_1[key], device, non_blocking)\
                for key, value in pack.items() if key != 'Rt_candi'}
            candi0, candi1 = blob_0['Rt_candi'], blob_1['Rt_candi']
            if len(candi0)==1 and len(candi1)==1 and candi0[0].shape[1]==candi1[0].shape[1]:
                composed['Rt_candi'] = [self._composite_batched(candi0[0],candi1[0],device, non_blocking)]
            else:
                composed['Rt_candi'] = xfer_to(blob_0,device,non_blocking) + xfer_to(blob_1,device,non_blocking)
            return composed
            
        if isinstance(pack, collections.abc.Sequence) and isinstance(pack[0], (str, bytes)):
            return blob_0 + blob_1

        raise NotImplementedError

    def _from_d0(self):
        try:
            blob_0 = next(self.iter_0)
        except StopIteration:
            self.iter_0 = iter(self.train_loader_0)
            blob_0 = next(self.iter_0)
        return blob_0

    def _from_d1(self):
        try:
            blob_1 = next(self.iter_1)
        except StopIteration:
            self.iter_1 = iter(self.train_loader_1)
            blob_1 = next(self.iter_1)
        return blob_1
        
    def get_batch(self, to_device = None):
        blob = self._from_d0()
        if self.iter_1 is not None:
            blob_1 = self._from_d1()
            return self._composite_batched(blob, blob_1,to_device)

        return xfer_to(blob, to_device, non_blocking=True)
        