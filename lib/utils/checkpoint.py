try:
    from . import random_state
except:
    import random_state
import os
import torch
import logging
import os.path as osp


logger = logging.getLogger()

class Checkpoint:
    @staticmethod
    def by_step(save_dir, persist_period = -1 ,*, start = 0, resume_period = 10000, latest_n = 2, best_n = 2):
        return Checkpoint(save_dir, persist_period=persist_period, start=start,
            resume_period=resume_period, latest_n = latest_n, best_n = best_n)
    
    @staticmethod
    def by_epoch(save_dir, persist_period = -1 ,*, start = 0, resume_period = 1, latest_n = 2, best_n = 2):
        return Checkpoint(save_dir, persist_period=persist_period, start=start,
            resume_period=resume_period, latest_n = latest_n, best_n = best_n)
    
    def __init__(self, save_dir, persist_period = 5 ,*, start = 0, resume_period = 1, latest_n = 2, best_n = 2) -> None:
        os.makedirs(save_dir, exist_ok=True)

        self.persist_period = int(persist_period)
        self.resume_period = max(resume_period, 1)
        self.steps = start
        self.save_dir = save_dir
        self.best_score = -1
        self.best_step = -1
        self.latest_n = latest_n
        self.best_n = best_n


    @staticmethod
    def _save_state(path, model, optimizer, scheduler, random, next_step, best_score = -1, best_step = -1):
        if isinstance(model,torch.nn.DataParallel):
            model=model.module.state_dict()
        else:
            model=model.state_dict()
        if optimizer is not None:
            optimizer=optimizer.state_dict()
        if scheduler is not None:
            scheduler=scheduler.state_dict()
        if random:
            random = random_state.get_random_state()
        state = dict(model=model, optimizer=optimizer, scheduler = scheduler, random = random,\
            best_score = best_score, best_step = best_step, next_step = next_step)
        if path:
            torch.save(state, path)
        return state
    
    @staticmethod
    def load(path, model, optimizer, scheduler, random = True, state = None, strict = True):
        if state is None:
            state = torch.load(path)
        if 'model' not in state:
            if isinstance(model,torch.nn.DataParallel):
                model.module.load_state_dict(state, strict = strict)
            else:
                model.load_state_dict(state, strict = strict)
            logger.info("ONLY got model parameters")
            #best_score, best_step, next_step
            return -1,-1,0 

        model_st=state['model']
        
        if isinstance(model,torch.nn.DataParallel):
            model.module.load_state_dict(model_st, strict = strict)
        else:
            model.load_state_dict(model_st, strict = strict)

        optimizer_st = state.setdefault('optimizer', None)
        scheduler_st = state.setdefault('scheduler', None)
        best_score = state.setdefault('best_score', -1)
        best_step = state.setdefault('best_step', -1)
        next_step = state.setdefault('next_step', 0)
        random_st = state.setdefault('random', None)
        if optimizer is not None and optimizer_st is not None:
            optimizer.load_state_dict(optimizer_st)
        if scheduler is not None and scheduler_st is not None:
            scheduler.load_state_dict(scheduler_st)
        if random and random_st is not None:
            random_state.restore_random_state(random_st)
        return best_score, best_step, next_step
    
    def _need_persist(self, check_step):
        return self.persist_period > 0 and (check_step % self.persist_period) == 0

    def finish(self, model = None, final_path = '', best_path = '', keep_ckpt = False):
        if model is not None and final_path:
            self.save_model(final_path, model)
        if best_path:
            best_ckpts = get_best(osp.join(self.save_dir, 'best'))
            for ckpt in best_ckpts:
                step = ckpt['step']
                score = ckpt['score']
                ckpt_path = ckpt['path']
                finalize_model(best_path+f'_{step}_{score:.4f}.pth', ckpt_path)

        if keep_ckpt:
            return
        best_ckpts = get_best(osp.join(self.save_dir, 'best'))
        for ck in best_ckpts:
            os.remove(ck['path'])
        ckpts = get_latest(self.save_dir)
        for ck in ckpts:
            os.remove(ck['path'])


    def _gen_path(self, name):
        return osp.join(self.save_dir, name)

    def step(self, model, optimizer = None, scheduler = None, random = True, score = -1):
        score = round(score, 4)

        self.steps += 1
        persist = self._need_persist(self.steps)
        resume_pt = (self.steps % self.resume_period) == 0

        best = False
        if score > 0 and score > self.best_score:
            self.best_score = score
            self.best_step = self.steps
            best = True

        need_save = persist or resume_pt or best
        if not need_save:
            return

        tmp_name = 'ckpt_tmp_save.tmp'
        os.makedirs(self.save_dir, exist_ok=True)
        tmp_save_path = self._gen_path(tmp_name)
        if os.path.exists(tmp_save_path):
            os.remove(tmp_save_path)

        self._save_state(tmp_save_path, model, optimizer, scheduler, random, self.steps, self.best_score, self.best_step)
        if persist:
            file_name = f'{self.steps}'
            if score > 0:
                file_name+=f'_{score:.4f}'
            os.makedirs(self._gen_path('persist'),exist_ok=True)
            new_path = self._gen_path(f'persist/{file_name}.pth')
            os.link(tmp_save_path, new_path)
            logger.info(f'persisted checkpoint at step {self.steps} saved to {new_path}')
        if resume_pt:
            new_path = link_latest(self.save_dir, tmp_save_path, step=self.steps, latest_n=self.latest_n)
            logger.info(f'checkpoint at step {self.steps} saved to {new_path}')
        if best:
            best_dir=os.path.join(self.save_dir, 'best')
            os.makedirs(best_dir,exist_ok=True)
            new_path = link_best(best_dir, tmp_save_path, step=self.steps, score = score, best_n=self.best_n)
            logger.info(f'best checkpoint at step {self.steps} saved to {new_path}')
        os.remove(tmp_save_path)

    def resume(self, model, optimizer = None, scheduler = None, random = True, ckpt_path = None, strict = True):
        '''
        return: the next step to train (starting from 0)
        '''
        if ckpt_path is not None:
            self.best_score, self.best_step, self.steps = self.load(ckpt_path, model, optimizer, scheduler, random, strict=strict)
            return self.steps

        old_state = self._save_state(None, model, optimizer, scheduler, random, 0)
        ckpts = get_latest(self.save_dir)
        
        nxt_stp = None
        for ckpt in ckpts:
            try:
                ckpt_path = ckpt['path']
                self.best_score, self.best_step, self.steps = self.load(ckpt_path, model, optimizer, scheduler, random, strict=strict)
                nxt_stp = self.steps
                break
            except Exception as e:
                logger.warning(f"Exception raised when resuming from checpoint: {ckpt_path}")
                logger.warning(str(e))

        if nxt_stp is None:
            self.load(None, model, optimizer, scheduler, random, old_state,)
            logger.info('Training from epoch 0.')
        return nxt_stp if nxt_stp is not None else 0

    @classmethod
    def save_model(cls, path, model:torch.nn.Module):
        save_model(path, model)

    @classmethod
    def load_model(cls, path, model:torch.nn.Module, map_location=None, strict=False):
        load_model(path, model, map_location, strict)

    @classmethod
    def finalize_model(cls, output_path, input_path, map_location=None):
        finalize_model(output_path, input_path, map_location)


def _best_path(dir, step, score, prefix=''):
    return os.path.join(dir, prefix+f'{step}_{score:.4f}.pth')

def get_best(dir, prefix=''):
    names = os.listdir(dir) if os.path.exists(dir) else []
    names = [name for name in names if name.endswith('.pth') and name.startswith(prefix)]
    pre_len = len(prefix)
    ckpts = [(int(info[0]),float(info[1])) for info in [name[pre_len:-4].split('_')[:2] for name in names]]
    ckpts = sorted(ckpts, key=lambda c:(c[1],c[0]), reverse=True)
    return [{'score':ckpt[1],'step':ckpt[0],'path':_best_path(dir, ckpt[0], ckpt[1], prefix)} for ckpt in ckpts]

def get_latest(dir, prefix=''):
    names = os.listdir(dir) if os.path.exists(dir) else []
    pre_len = len(prefix)
    steps = [int(name[pre_len:-4]) for name in names if name.endswith('.pth') and name.startswith(prefix)]
    steps = sorted(steps, reverse=True)

    return [{'step':step, 'path':os.path.join(dir, prefix + f'{step}.pth') }for step in steps]
    

def link_latest(dir, ckpt_src, step, latest_n = 2, prefix=''):
    names = os.listdir(dir) if os.path.exists(dir) else []
    pre_len = len(prefix)
    steps = [int(name[pre_len:-4]) for name in names if name.endswith('.pth') and name.startswith(prefix)]
    new_path = os.path.join(dir, prefix + f'{step}.pth')
    os.link(ckpt_src, new_path)
    steps.append(step)
    steps = sorted(steps, reverse=True)
    for s in steps[latest_n:]:
        path = os.path.join(dir, f'{s}.pth')
        os.remove(path)
    return new_path


def link_best(dir, ckpt_src, step, score, best_n = 3, prefix=''):
    score = round(score,4)
    names = os.listdir(dir) if os.path.exists(dir) else []
    names = [name for name in names if name.endswith('.pth') and name.startswith(prefix)]
    pre_len = len(prefix)
    ckpts = [(int(info[0]),float(info[1])) for info in [name[pre_len:-4].split('_')[:2] for name in names]]
    new_ckpt = (int(step), float(score))
    ckpts.append(new_ckpt)
    ckpts = sorted(ckpts, key=lambda c:(c[1],c[0]), reverse=True)
    if new_ckpt not in ckpts[:best_n]:
        return ''
    new_path = _best_path(dir, new_ckpt[0], new_ckpt[1], prefix)
    os.link(ckpt_src, new_path)
    for ckpt in ckpts[best_n:]:
        name = _best_path(dir, ckpt[0], ckpt[1], prefix)
        os.remove(name)
    return new_path


def finalize_model(output_path, input_path, map_location=None):
    st_model = torch.load(input_path, map_location=map_location)
    if 'model' in st_model:
        st_model = st_model['model'] 
    torch.save(st_model, output_path)
    

def save_model(path, model:torch.nn.Module):
    if isinstance(model,torch.nn.DataParallel):
        model_state=model.module.state_dict()
    else:
        model_state=model.state_dict()
    torch.save(model_state, path)


def load_model(path, model:torch.nn.Module, map_location=None, strict=False):
    st_model=torch.load(path, map_location=map_location)
    if 'model' in st_model:
        st_model = st_model['model']
        
    if isinstance(model,torch.nn.DataParallel):
        model.module.load_state_dict(st_model,strict=strict)
    else:
        model.load_state_dict(st_model,strict=strict)
    logger.info(f'loaded model from {path}')
