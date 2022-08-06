import os.path as osp
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from munch import Munch
from picasso.models import shape_seg, shape_cls
from my_model.datasets import build_dataset

import pytorch_lightning as pl
import torchmetrics


def build_model(cfg: Munch):
    if cfg.model.type == 'picasso_cls':
        return LitPicassoCls(cfg)
    elif cfg.model.type == 'picasso_shape_seg':
        return LitPicassoShapeSeg(cfg)
    elif cfg.model.type == 'picasso_scene_seg':
        return LitPicassoSceneSeg(cfg)
    else:
        raise NotImplementedError


class LitBase(pl.LightningModule):

    def __init__(self, cfg: Munch):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.cfg.optimizer.type == 'AdamW':
            optimizer = torch.optim.AdamW(self.net.parameters(), **self.cfg.optimizer.cfg)
        else:
            raise NotImplementedError

        if self.cfg.scheduler.type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.cfg.scheduler.cfg)
            return [optimizer], [scheduler]
        elif self.cfg.scheduler.type == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, total_steps=self.cfg.epochs * len(self.train_dataloader()), **self.cfg.scheduler.cfg)
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
        else:
            raise NotImplementedError

    def train_dataloader(self):
        dataset = build_dataset(self.cfg.data.train)
        return DataLoader(dataset,
                          collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
                          **self.cfg.dataloader.train)

    def val_dataloader(self):
        dataset = build_dataset(self.cfg.data.val)
        return DataLoader(dataset,
                          collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
                          **self.cfg.dataloader.val)

    def test_dataloader(self):
        dataset = build_dataset(self.cfg.data.test)
        return DataLoader(dataset,
                          collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
                          **self.cfg.dataloader.test)


class LitPicassoCls(LitBase):

    def __init__(self, cfg: Munch):
        super().__init__(cfg)
        self.net = shape_cls.PicassoNetII(**cfg.model.cfg)

        # metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        vertex_in = batch['vertex_in']
        face_in = batch['face_in']
        label_in = batch['label_in']
        nv_in = batch['nv_in']
        mf_in = batch['mf_in']

        # shuffle normals: flip normals
        out = self.net(vertex_in, face_in, nv_in, mf_in, shuffle_normals=True)
        loss = F.cross_entropy(out, label_in)

        self.log('loss', loss, batch_size=self.cfg.dataloader.train.batch_size)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        self.train_accuracy(out, label_in)
        self.log("train_acc", self.train_accuracy, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vertex_in = batch['vertex_in']
        face_in = batch['face_in']
        label_in = batch['label_in']
        nv_in = batch['nv_in']
        mf_in = batch['mf_in']

        out = self.net(vertex_in, face_in, nv_in, mf_in, shuffle_normals=False)
        loss = F.cross_entropy(out, label_in)
        self.log('val_loss', loss, True, batch_size=self.cfg.dataloader.val.batch_size)
        self.val_accuracy(out, label_in)
        self.log("val_acc", self.val_accuracy, on_epoch=True)

    def test_step(self, batch, batch_idx):
        vertex_in = batch['vertex_in']
        face_in = batch['face_in']
        label_in = batch['label_in']
        nv_in = batch['nv_in']
        mf_in = batch['mf_in']

        out = self.net(vertex_in, face_in, nv_in, mf_in, shuffle_normals=False)
        loss = F.cross_entropy(out, label_in)
        self.log('test_loss', loss, True, batch_size=self.cfg.dataloader.test.batch_size)
        self.test_accuracy(out, label_in)
        self.log("test_acc", self.test_accuracy, on_epoch=True)


class LitPicassoShapeSeg(LitBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.net = shape_cls.PicassoNetII(**cfg.model.cfg)

    def forward(self, x):
        return self.net(x)


class LitPicassoSceneSeg(LitBase):
    def __init__(self, cfg: Munch):
        super().__init__(cfg)
        self.net = shape_cls.PicassoNetII(**cfg.model.cfg)

    def forward(self, x):
        return self.net(x)
