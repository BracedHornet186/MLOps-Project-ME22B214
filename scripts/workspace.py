import json
import logging
import math
import os
import pickle
import shutil
import sys
import time
import contextlib
from pathlib import Path
from typing import Any, AnyStr, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.utils
import tqdm
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter

try:
    import rich
except ImportError:
    rich = None


from data import SHOW_PREF_TIME


global_workspace_dict = {}


def ws(name: str = 'current') -> Optional['Workspace']:
    return global_workspace_dict.get(name)


def log(message: Any, name: str = 'current', **kwargs) -> None:
    workspace = ws(name=name)
    if workspace is None:
        if rich is None:
            print(message)
        else:
            rich.print(message)
    else:
        workspace.log(message, **kwargs)


@contextlib.contextmanager
def perf_time(title: str):
    start_at = time.perf_counter()
    try:
        yield
    finally:
        end_at = time.perf_counter()
    elapsed = end_at - start_at
    if SHOW_PREF_TIME:
        log(f'({title}) Elapsed time: {elapsed}')


def get_logger(name: str, log_file: Optional[str] = None):
    logger = logging.getLogger(name)
    logger.setLevel('DEBUG')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    if rich is None:
        handler = logging.StreamHandler()
    else:
        from rich.logging import RichHandler
        handler = RichHandler()

    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class Colab:

    DRIVE_MOUNT_DIR = '/content/drive' 
    DRIVE_PROJECT_DIR_RELPATH = 'MyDrive/dev/kaggle/TODO'
    REPOS_DIR = 'TODO'

    @classmethod
    def setup(cls):
        import sys
        sys.path.insert(0, cls.REPOS_DIR)

    @classmethod
    def mount_drive(cls):
        from google.colab import drive
        drive.mount(cls.DRIVE_MOUNT_DIR, force_remount=True)

    @classmethod
    def project_dir(cls) -> Path:
        return Path(cls.DRIVE_MOUNT_DIR) / cls.DRIVE_PROJECT_DIR_RELPATH

    @classmethod
    def workspace_dir(cls) -> Path:
        return cls.project_dir() / 'ws'

    @classmethod
    def in_colab(cls):
        import os
        return 'COLAB_GPU' in os.environ

    @classmethod
    def upload_file(cls):
        from google.colab import files
        uploaded = files.upload()
        for fn in uploaded.keys():
            print('User uploaded file "{name}" with length {length} bytes'.format(
                  name=fn, length=len(uploaded[fn])))

    @classmethod
    def sleep(cls):
        import time
        x = torch.rand((10, 10)).cuda()
        while True:
            time.sleep(1)


class Workspace:
    def __init__(self, run_id: str,
                 root_dir: Optional[str] = None,
                 flush: bool = False,
                 use_colab_local_storage: bool = False,
                 set_global_workspace: bool = True,
                 use_tensorboard: bool = True):
        self.run_id = run_id
        self.root_dir = Path(root_dir or 'workspace')
        if not use_colab_local_storage and Colab.in_colab():
            self.root_dir = Colab.workspace_dir()
            print(f'[Colab] Use workspace root dir: {self.root_dir}')
        self.logger = None
        self.best_score: Optional[float] = None
        self.best_epoch: int = 1
        self.tb_writer = None
        self.flush = flush
        self.use_tensorboard = use_tensorboard
        if set_global_workspace:
            global_workspace_dict['current'] = self

    def setup(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tb_root_log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_log_dir.mkdir(parents=True, exist_ok=True)
        self.var_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f'{__name__}_{self.run_id}', log_file=str(self.log_file))
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.tb_log_dir))

        self.logger.propagate = False
        self.logger.info(f'run_dir={self.run_dir}')

        return self

    @property
    def run_dir(self) -> Path:
        return self.root_dir / self.run_id

    @property
    def model_dir(self) -> Path:
        return self.run_dir / 'model'

    @property
    def conf_file(self) -> Path:
        return self.run_dir / f'{self.run_id}.yaml'

    @property
    def tb_root_log_dir(self) -> Path:
        return self.root_dir / 'tb'

    @property
    def tb_log_dir(self) -> Path:
        return self.tb_root_log_dir / self.run_id

    @property
    def var_dir(self) -> Path:
        return self.run_dir / 'var'

    @property
    def vis_dir(self) -> Path:
        return self.run_dir / 'vis'

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / 'eval'

    @property
    def log_file(self) -> Path:
        return self.run_dir / f'{self.run_id}.log'

    def log(self, message: Any,
            epoch: Optional[int] = None,
            iteration: Optional[int] = None):
        recs = [f'{k}: {v}'
                for k, v in {'epoch': epoch,
                             'iteration': iteration}.items()
                if v is not None]
        if recs:
            message = f'[{", ".join(recs)}] {message}'
        try:
            self.logger.info(message)
            if self.flush:
                if self.logger is not None:
                    for handler in self.logger.handlers:
                        handler.flush()
        except OSError:
            print(f'[Logging error]')
            print(message)

    def save_conf(self, config_file: Union[Path, str]):
        shutil.copy(config_file, self.conf_file)

    def save_batch_image(self,
                         x: torch.Tensor,
                         filename: str = 'batch.jpg',
                         num_max_images: int = 16) -> None:
        save_to = str(self.vis_dir / filename)
        x = x.detach()
        x = x[:num_max_images]
        torchvision.utils.save_image(
            x,
            save_to,
            padding=2,
            normalize=True,
            nrow=4,
        )

    def save_bestmodel(self,
                       model: torch.nn.Module,
                       epoch: int,
                       score: float,
                       maximize: bool = True) -> bool:
        if self.best_score is None:
            if maximize:
                self.best_score = 0
            else:
                self.best_score = math.inf

        def check_save_condition(_score: float) -> bool:
            assert self.best_score is not None
            if maximize:
                return _score >= self.best_score
            return _score <= self.best_score

        if check_save_condition(score):
            best_model_path = self.model_dir / f'{self.run_id}_best.model'
            torch.save(model.state_dict(), best_model_path)
            with open(self.model_dir / 'bestmodel_info.txt', 'w') as fp:
                fp.write(
                    json.dumps({
                        'epoch': epoch,
                        'score': score
                    })
                )
            self.log('Saved best model', epoch=epoch)
            self.log(f'Best score {self.best_score} -> {score}', epoch=epoch)
            self.best_score = score
            self.best_epoch = epoch
            return True

        self.log('Skipped: Saving a bestmodel', epoch=epoch)
        return False

    def save_model(self, model: torch.nn.Module, epoch: int):
        model_path = self.model_dir / f'{self.run_id}_epoch{epoch}.model'
        torch.save(model.state_dict(), model_path)
        self.log(f'Saved model: {model_path}', epoch=epoch)

    def save_lastmodel(self, model: torch.nn.Module):
        model_path = self.model_dir / f'{self.run_id}_last.model'
        torch.save(model.state_dict(), model_path)
        self.log(f'Saved last model: {model_path}')

    def save_checkpoint(self, epoch: int, name=None, **kwargs):
        name = name or f'epoch_{epoch}'
        checkpoint_path = self.model_dir / f'{self.run_id}_{name}.checkpoint'
        checkpoint = kwargs
        torch.save(checkpoint, checkpoint_path)
        self.log(f'Saved checkpoint: {checkpoint_path}', epoch=epoch)

    def save_as(self, model: torch.nn.Module, filename: str):
        model_path = self.model_dir / filename
        torch.save(model.state_dict(), model_path)
        self.log(f'Saved model: {model_path}')

    def plot_value(self, tag, value, global_step):
        if not self.tb_writer:
            return
        self.tb_writer.add_scalar(tag, value,
                                  global_step=global_step)

    def plot_metrics(
        self,
        metrics: Dict[str, float],
        global_step: int,
        prefix_tag: str = 'val'
    ) -> None:
        for key, value in metrics.items():
            tag = f'{prefix_tag}/{key}'
            self.plot_value(tag, value, global_step)

    def plot_image(
        self,
        tag: str,
        image: Union[np.ndarray, torch.Tensor],
        global_step: int
    ) -> None:
        if not self.tb_writer:
            return
        self.tb_writer.add_image(tag, image,
                                 global_step=global_step)

    def plot_figure(
        self,
        tag: str,
        fig: Any,
        global_step: int
    ) -> None:
        if not self.tb_writer:
            return
        self.tb_writer.add_figure(tag, fig,
                                  global_step=global_step)
    
    def plot_figures(
        self,
        figs: Dict[str, Any],
        global_step: int,
        prefix_tag: str = 'val',
        close_figures: bool = True
    ) -> None:
        for key, fig in figs.items():
            tag = f'{prefix_tag}/{key}'
            self.plot_figure(tag, fig, global_step)
            if close_figures:
                plt.close(fig)

    def plot_text(
        self, tag: str, text: str, global_step: int
    ) -> None:
        if not self.tb_writer:
            return
        self.tb_writer.add_text(tag, text, global_step=global_step)

    def plot_conf(
        self, tag: str = 'config'
    ) -> None:
        if not self.conf_file.exists():
            self.log('No config file found')
            return
        with open(self.conf_file, 'r') as fp:
            text = fp.read()
        self.plot_text(tag, f'<pre>{text}</pre>', global_step=0)

    def plot_loadavg(self, global_step: int) -> None:
        load_avg, _, _ = os.getloadavg()
        self.plot_value('cpu/loadavg/1m',
                        load_avg, global_step=global_step)

    def plot_lr(
        self,
        optimizer: torch.optim.Optimizer,
        global_step: int,
    ) -> None:
        for i, g in enumerate(optimizer.param_groups):
            tag = f'train/lr/group{i}'
            value = float(g['lr'])
            self.plot_value(
                tag, value, global_step
            )
