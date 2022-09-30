from munch import Munch
from .shapenet import ShapenetCoreV2
from .tooth_axis import ToothAxis


def build_dataset(cfg: Munch):
    if cfg.type == 'shapenetcore_v2':
        return ShapenetCoreV2(**cfg.cfg)
    if cfg.type == 'tooth_axis':
        return ToothAxis(**cfg.cfg)
    else:
        raise NotImplementedError