import os
import sys, os.path as osp, json, shutil
import click, yaml
from munch import Munch
from termcolor import colored

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append(osp.dirname(osp.dirname(__file__)))
from tools.pl_models import build_model


@click.command()
@click.option('--config', help="train config file path", default="configs/shapenetcore_v2.yaml")
@click.option('--work_dir', help="the dir to save logs and models", default="work_dir")
@click.option('--version', help="the version of the experiment", default=None)
@click.option('--resume_from', help="the checkpoint file to resume from", default=None)
@click.option('--auto_resume', help="auto resume from the latest checkpoint", is_flag=True, default=False)
@click.option('--gpus', help="the gpus to use", default=1)
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
    logger = TensorBoardLogger(kwargs['work_dir'], name=osp.split(kwargs['config'])[-1].split('.')[0], version=version)
    os.makedirs(logger.log_dir, exist_ok=True)
    shutil.copy(kwargs['config'], logger.log_dir)  # copy config file

    # trainer
    debug = False
    debug_args = {'limit_train_batches': 10, "limit_val_batches": 10} if debug else {}

    model = build_model(cfg)
    callback = ModelCheckpoint(
        save_top_k=kwargs['save_last_k'] if kwargs['save_last_k'] is not None else cfg.save_last_k,
        monitor='epoch', mode='max', save_last=True,
        every_n_epochs=cfg.save_freq, save_on_train_epoch_end=True)
    trainer = pl.Trainer(logger, accelerator='gpu', devices=kwargs['gpus'], max_epochs=cfg.epochs, callbacks=[callback],
                         **debug_args)

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
