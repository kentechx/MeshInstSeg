import sys, os.path as osp, json
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
@click.option('--resume_from', help="the checkpoint file to resume from", default=None)
@click.option('--auto_resume', help="auto resume from the latest checkpoint", is_flag=True)
@click.option('--gpus', help="the gpus to use", default=1)
@click.option('--no_validate', help="do not validate the model", is_flag=True)
@click.option('--seed', help="random seed", default=42)
@click.option('--deterministic', help="whether to set deterministic options for CUDNN backend", is_flag=True)
def run(**kwargs):
    print(colored(json.dumps(kwargs, indent=2), 'blue'))

    # assign to cfg
    cfg = Munch.fromDict(yaml.safe_load(open(kwargs['config'])))
    print(colored(json.dumps(cfg, indent=2), 'green'))

    pl.seed_everything(kwargs['seed'])

    model = build_model(cfg)
    # if args.load_from_checkpoint:
    #     model = LitTeethGNN.load_from_checkpoint(args.load_from_checkpoint)
    #
    # TODO: resume, workdir
    logger = TensorBoardLogger(kwargs["work_dir"], kwargs['work_dir'])
    callback = ModelCheckpoint(monitor='val_loss', save_top_k=20, save_last=True, mode='min')
    #
    debug = False
    debug_args = {'limit_train_batches': 10} if debug else {}
    trainer = pl.Trainer(logger, accelerator='gpu', devices=kwargs['gpus'], max_epochs=cfg.epochs, callbacks=[callback],
                         resume_from_checkpoint=kwargs['resume_from'], **debug_args)
    trainer.fit(model)

    results = trainer.test()
    print(results)


if __name__ == '__main__':
    run()
