from munch import Munch
from .shapenet import ShapenetCoreV2


def build_dataset(cfg: Munch):
    if cfg.type == 'shapenetcore_v2':
        return ShapenetCoreV2(**cfg.cfg)
    else:
        raise NotImplementedError