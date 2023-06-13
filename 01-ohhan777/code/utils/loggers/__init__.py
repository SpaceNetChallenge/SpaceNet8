import os
import pkg_resources as pkg
import torch
from threading import Thread
#from utils.plots import plot_images
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.general import colorstr

LOGGERS = ('csv', 'tb', 'wandb')
RANK = int(os.getenv('RANK', -1))

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in [0, -1]:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None


class Loggers():
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = ['train/loss', # train loss
                     # 'metrics/precision', 'metrics/recall', # metrics
                     'val/building_iou', 'val/road_iou', 'val/flood_mean_iou', 'val/roadspeed_m_iou', 'val/building_pix_acc', 'val/road_pix_acc', 'val/loss',  # val loss
                     'x/lr0', 'x/lr1', 'x/lr2']  # params
                     #'x/lr']  # params
        self.best_keys = ['best/epoch', 'best/building_iou', 'best/road_iou', 'best/flood_mean_iou', 'best/roadspeed_m_iou', 'best/building_pix_acc', 'best/road_pix_acc', 'best/loss']
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Message
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize kari-seg 🚀 runs (RECOMMENDED)"
        
        # W&B
        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = torch.load(self.weights).get('wandb_id') if self.opt.resume and not wandb_artifact_resume else None
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, run_id)
        else:
            self.wandb = None

    def on_train_start(self):
        # Callback runs on train start
        pass


    def on_train_epoch_start(self, epoch):
        # Callback runs on train epoch start
        pass

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

    def on_train_batch_end(self, ni, model, imgs, targets):
        pass
        # Callback runs on train batch end                
        #if ni < 3:
        #    filename = str(self.save_dir / f'train_batch{ni}.png')
        #    Thread(target=plot_images, args=(imgs, targets, targets, filename), daemon=True).start() 
    
    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

        if self.wandb:
            if best_fitness == fi:
                best_results = [epoch] + vals[1:8]
                for i, name in enumerate(self.best_keys):
                    self.wandb.wandb_run.summary[name] = best_results[i]   # log best results in the summary
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        if self.wandb:
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best, epoch, results):
        if self.wandb:
            self.wandb.finish_run()

    def on_val_end(self):
        # Callback runs on val end
        if self.wandb:
            files = sorted(self.save_dir.glob('val*.png'))
            self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})





