import os
import sys, os.path as osp, json, shutil
import click, yaml
from munch import Munch
from termcolor import colored

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics

sys.path.append(osp.dirname(osp.dirname(__file__)))
from tools.pl_models import LitBase

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from munch import Munch
from picasso.models import shape_seg, shape_cls
from my_model.datasets import build_dataset


def rot6d_to_rot3x3(rot6d):
    rot3x3 = torch.zeros((len(rot6d), 3, 3), device=rot6d.device)
    rot3x3[:, :, 0] = rot6d[:, :3]
    rot3x3[:, :, 0] /= torch.norm(rot3x3[:, :, 0], dim=1, keepdim=True) + 1e-8
    rot3x3[:, :, 1] = rot6d[:, 3:6]
    rot3x3[:, :, 2] = torch.cross(rot6d[:, :3], rot6d[:, 3:6])
    rot3x3[:, :, 2] /= torch.norm(rot3x3[:, :, 2], dim=1, keepdim=True) + 1e-8
    rot3x3[:, :, 1] = torch.cross(rot3x3[:, :, 2], rot3x3[:, :, 0])
    return rot3x3


def angle_diff(pred, y):
    rot_pred = rot6d_to_rot3x3(pred)
    rot_y = rot6d_to_rot3x3(y)
    rot = torch.bmm(rot_pred, rot_y.transpose(1, 2))
    angle = torch.clip((rot.diagonal(dim1=1, dim2=2).sum(-1) - 1.) / 2., -1, 1).acos() * 180 / torch.pi  # (n, )
    return angle


class AngleGeodesic(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("angle_geodesic", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        angles = angle_diff(preds, target)
        self.angle_geodesic += angles.sum()
        self.total += len(preds)

    def compute(self):
        return self.angle_geodesic / self.total


class ToothAxisReg(LitBase):

    def __init__(self, cfg: Munch):
        super().__init__(cfg)
        self.net = shape_cls.PicassoNetII(**cfg.model.cfg)

        # metrics
        self.train_angle_geodesic = AngleGeodesic()
        self.val_angle_geodesic = AngleGeodesic()

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        vertex_in = batch['vertex_in']
        face_in = batch['face_in']
        label_in = batch['label_in']
        nv_in = batch['nv_in']
        mf_in = batch['mf_in']

        # shuffle normals: flip normals
        out = self.net(vertex_in, face_in, nv_in, mf_in, shuffle_normals=False)
        loss = F.mse_loss(out, label_in)

        angles = angle_diff(out, label_in)
        self.log('loss', loss, batch_size=self.cfg.dataloader.train.batch_size)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        self.log("angle_diff_max", angles.max(), prog_bar=True, batch_size=self.cfg.dataloader.train.batch_size)
        self.train_angle_geodesic(out, label_in)
        self.log("train_angle_diff", self.train_angle_geodesic, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vertex_in = batch['vertex_in']
        face_in = batch['face_in']
        label_in = batch['label_in']
        nv_in = batch['nv_in']
        mf_in = batch['mf_in']

        out = self.net(vertex_in, face_in, nv_in, mf_in, shuffle_normals=False)
        loss = F.mse_loss(out, label_in)
        angles = angle_diff(out, label_in)
        self.log('val_loss', loss, True, batch_size=self.cfg.dataloader.val.batch_size)
        self.log("val_angle_diff_max", angles.max(), prog_bar=True, batch_size=self.cfg.dataloader.val.batch_size)
        self.val_angle_geodesic(out, label_in)
        self.log("val_angle_diff", self.val_angle_geodesic, prog_bar=True)


@click.command()
@click.option('--config', help="train config file path", default="../configs/tooth_axis.yaml")
@click.option('--gpus', help="the gpus to use", default=1)
@click.option('--version', help="the version of the experiment", default=None)
@click.option('--auto_resume', help="auto resume from the latest checkpoint", is_flag=True, default=False)
@click.option('--resume_from', help="the checkpoint file to resume from", default=None)
@click.option('--no_validate', help="do not validate the model", is_flag=True)
@click.option('--seed', help="random seed", default=42)
@click.option('--deterministic', help="whether to set deterministic options for CUDNN backend", is_flag=True)
@click.option('--save_last_k', help="save the top k checkpoints", default=None)
def run(**kwargs):
    print(colored(json.dumps(kwargs, indent=2), 'blue'))

    # assign to cfg
    cfg = Munch.fromDict(yaml.safe_load(open(kwargs['config'])))
    cfg["cmd_params"] = kwargs
    print(colored(json.dumps(cfg, indent=2), 'green'))

    pl.seed_everything(kwargs['seed'])

    # logger
    version = 0 if kwargs['version'] is None else kwargs['version']
    logger = TensorBoardLogger("work_dir", name=osp.split(kwargs['config'])[-1].split('.')[0], version=version)
    os.makedirs(logger.log_dir, exist_ok=True)
    shutil.copy(kwargs['config'], logger.log_dir)  # copy config file

    # trainer
    debug = False
    debug_args = {'limit_train_batches': 10, "limit_val_batches": 10} if debug else {}

    model = ToothAxisReg(cfg)
    callback = ModelCheckpoint(
        save_top_k=kwargs['save_last_k'] if kwargs['save_last_k'] is not None else cfg.save_last_k,
        monitor='epoch', mode='max', save_last=True,
        every_n_epochs=cfg.save_freq, save_on_train_epoch_end=True)
    trainer = pl.Trainer(logger, accelerator='gpu', devices=kwargs['gpus'], max_epochs=cfg.epochs, callbacks=[callback],
                         deterministic=kwargs['deterministic'], **debug_args)

    # resume
    ckpt_path = osp.join(logger.log_dir, 'checkpoints/last.ckpt') if kwargs['auto_resume'] else None
    if kwargs['resume_from'] is not None:
        ckpt_path = kwargs['resume_from']

    # fit
    trainer.fit(model, ckpt_path)

    results = trainer.test()
    print(results)


if __name__ == '__main__':
    run()
