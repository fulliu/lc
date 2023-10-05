import torch
from torch import Tensor
import torchvision.transforms as transforms
import losses, test
import argparse, os, logging
import floatbits
from utils import xfer_to
from lib.utils.checkpoint import Checkpoint
from lib.utils.random_state import seed_all
from lib.utils.setup_logger import setup_logger

from utils import get_dataloader, get_model_optim, build_lr_scheduler, MultiLoader
from mmcv import Config, DictAction
from tqdm import tqdm

import tensorboardX

logger = logging.getLogger(__name__)

torch.backends.cuda.preferred_linalg_library("cusolver")

# for gdr-net structure
def train_by_epoch(args, cfg, outdir): 
    train_set, train_loader = get_dataloader(cfg.train_dataset, cfg.dataloader, cfg)
    test_set, test_loader = get_dataloader(cfg.test_dataset, cfg.dataloader, cfg, train=False,)
    evaluator = test.get_evaluator(cfg.test_dataset, cfg)

    bit_cnt = train_set.bit_cnt
    total_bit_cnt = 0 if bit_cnt is None else sum(bit_cnt)
    model, optimizer = get_model_optim(cfg, total_bit_cnt=total_bit_cnt)
    model.loss_fn = losses.Loss_fn(cfg.loss, cfg, total_bit_cnt)

    num_epochs = cfg.train.num_epochs
    num_steps = num_epochs * len(train_loader)
    scheduler = build_lr_scheduler(cfg.scheduler, optimizer, num_steps)
    model = model.to(args.device)

    writer = tensorboardX.SummaryWriter(outdir)
    ckpter = Checkpoint.by_epoch(os.path.join(outdir,'ckpts'))
    resume = args.resume or args.ckpt
    start_epoch = ckpter.resume(model,optimizer,scheduler,ckpt_path = args.ckpt) if resume else 0
    steps_per_epoch =  len(train_loader)
    steps = start_epoch * steps_per_epoch
    
    if start_epoch >= num_epochs:
        exit(0)

    test_start = cfg.train.get('test_start', 0)
    transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(train_loader)
        for blob_cpu in pbar:
            blob_dev = xfer_to(blob_cpu, device=args.device)
            if bit_cnt is not None:
                blob_dev['bit_cnt']=bit_cnt
            img_in = transform(blob_dev['rgb_in'])
            out_dict = model(img_in)
            losses.annots_on_the_fly(blob_dev, out_dict, cfg, steps)
            loss_dict, w_loss_dict = model.loss_fn(blob_dev, out_dict, epoch, steps, steps_per_epoch)

            loss:Tensor = sum(w_loss_dict.values())
            pbar.set_description(f'{epoch+1}/{num_epochs}:{steps}/{num_steps}',refresh=False)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            for k, v in w_loss_dict.items():
                writer.add_scalar('loss/'+k,v, steps)
            writer.add_scalar('loss',loss.cpu(), steps)
            steps += 1

        score = -1
        if (epoch + 1) % cfg.train.test_every == 0 and epoch != num_epochs and epoch >= test_start:
            test_res = test.test(args, cfg, model, test_loader, evaluator)
            score = next(iter(test_res.values()))['avg_score']
        ckpter.step(model, optimizer, scheduler, score = score)

    ckpter.finish(model, os.path.join(outdir, 'model_final.pth'), os.path.join(outdir, 'model_best'))


def train_by_step(args, cfg, outdir):
    train_loader = MultiLoader(cfg)
    test_set, test_loader = get_dataloader(cfg.test_dataset, cfg.dataloader, cfg, train=False,)

    evaluator = test.get_evaluator(cfg.test_dataset, cfg)

    bit_cnt = train_loader.train_set_0.bit_cnt
    total_bit_cnt = 0 if bit_cnt is None else sum(bit_cnt)
    model, optimizer = get_model_optim(cfg, total_bit_cnt=total_bit_cnt)

    model.loss_fn = losses.Loss_fn(cfg.loss, cfg, total_bit_cnt)

    num_steps = cfg.train.num_steps
    scheduler = build_lr_scheduler(cfg.scheduler, optimizer, num_steps)
    model = model.to(args.device)

    writer = tensorboardX.SummaryWriter(outdir)
    ckpter = Checkpoint.by_step(os.path.join(outdir,'ckpts'), resume_period = cfg.train.get('ckpt_every', 5000))
    resume = args.resume or args.ckpt
    start_step = ckpter.resume(model,optimizer,scheduler,ckpt_path = args.ckpt) if resume else 0

    if start_step >= num_steps:
        exit(0)

    test_start = cfg.train.get('test_start', 0)
    transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    pbar = tqdm(total=num_steps,initial=start_step)
    for step in range(start_step, num_steps):
        blob_dev = train_loader.get_batch(args.device)
        if bit_cnt is not None:
            blob_dev['bit_cnt']=bit_cnt
        img_in = transform(blob_dev['rgb_in'])
        out_dict = model(img_in)
        losses.annots_on_the_fly(blob_dev, out_dict, cfg, step)
        loss_dict, w_loss_dict = model.loss_fn(blob_dev, out_dict, 0, step, 0)

        loss:Tensor = sum(w_loss_dict.values())
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        for k,v in w_loss_dict.items():
            writer.add_scalar('loss/'+k,v, step)
        writer.add_scalar('loss',loss.cpu(), step)

        score = -1
        if (step + 1) % cfg.train.test_every == 0 and step != num_steps and step >= test_start:
            test_res = test.test(args, cfg, model, test_loader, evaluator)
            score = next(iter(test_res.values()))['avg_score']

        ckpter.step(model, optimizer, scheduler, score = score)
        pbar.update(1)

    ckpter.finish(model, os.path.join(outdir, 'model_final.pth'), os.path.join(outdir, 'model_best'))



def update_cfg_from_args(args, cfg):
    if args.name:
        cfg['exp_name'] = args.name

    if args.obj is not None:
        cfg.obj_ids = args.obj

    if args.opts is not None:
        cfg.merge_from_dict(args.opts)
    return cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='configs/config.yaml')
    parser.add_argument('--output',type=str, default='output')
    parser.add_argument('--name',type=str)
    parser.add_argument('--obj',type=int, nargs='+')

    parser.add_argument('--ckpt',type=str)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--opts',nargs='+', action=DictAction)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg = update_cfg_from_args(args, cfg)
    outdir = os.path.join(args.output, cfg.exp_name)+'-'+str(cfg.obj_ids).replace(' ','').replace(',','_')[1:-1]
    setup_logger(outdir)
    os.makedirs(outdir, exist_ok=True)
    
    assert bool(cfg.train.get('num_epochs',None))!=bool(cfg.train.get('num_steps',None))
    seed_all(42)
    floatbits.set_black_background(cfg.get('black_background',False))

    if cfg.train.get('num_steps',None):
        train_by_step(args, cfg, outdir)
    else:
        train_by_epoch(args, cfg, outdir)




