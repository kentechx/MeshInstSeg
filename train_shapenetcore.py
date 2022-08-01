import warnings
warnings.filterwarnings("ignore")
import argparse, time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from datetime import datetime
import picasso.mesh.utils as meshUtil
from picasso.augmentor import Augment
from picasso.mesh.dataset import MeshDataset
from picasso.models.shape_cls import PicassoNetII
from picasso.mesh.dataset import default_collate_fn
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to tfrecord data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../log_shapenetcore', help='Log dir [default: ../log_shapenetcore]')
parser.add_argument('--ckpt_epoch', type=int, default=None, help='epoch model to load [default: None]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--max_num_vertices', type=int, default=1500000, help='maximum vertices allowed in a batch')
parser.add_argument('--num_clusters', type=int, default=27, help='number of cluster components')
opt = parser.parse_args()

LOG_DIR = opt.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp train_ShapeNetCore.py %s'%(LOG_DIR))   # bkp of train procedure
os.system('cp picasso/models/shape_cls.py %s'%(LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')

NUM_CLASSES = 55


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


class TransformTrain(object):
    def __init__(self, prob=0.5, num_classes=None, drop_rate=None, voxel_size=None):
        self.prob = prob
        self.num_classes = num_classes
        self.drop_rate = drop_rate     # the rate of dropping vertices
        self.voxel_size = voxel_size

    def augment_fn(self, vertex, face, texture=None, vertex_label=None, face_label=None):
        assert(vertex.shape[1]==3)
        vertex = Augment.flip_point_cloud(vertex, prob=self.prob)
        vertex = Augment.random_scale_point_cloud(vertex, scale_low=0.5, scale_high=1.5, prob=self.prob)
        vertex = Augment.rotate_point_cloud(vertex, upaxis=3, prob=self.prob)
        vertex = Augment.rotate_perturbation_point_cloud(vertex, prob=self.prob)
        vertex = Augment.jitter_point_cloud(vertex, sigma=0.01, prob=self.prob)
        if texture is not None:
            texture = Augment.random_drop_color(texture, prob=self.prob)
            texture = Augment.shift_color(texture, prob=self.prob)
            texture = Augment.jitter_color(texture, prob=self.prob)
            texture = Augment.auto_contrast_color(texture, prob=self.prob)
            vertex  = torch.cat([vertex, texture], dim=1)    # concat texture back
        if (self.drop_rate>.0) and (self.drop_rate<1.):
            vertex, face, label, \
            face_mask = Augment.random_drop_vertex(vertex, face, vertex_label, face_label,
                                                   drop_rate=self.drop_rate, prob=self.prob)
        else:
            label = vertex_label if face_label is None else face_label

        return vertex, face, label

    def __call__(self, args):
        vertex, face, label = args
        vertex, face, label = self.augment_fn(vertex[:,:3], face, vertex[:,3:], vertex_label=label)
        if self.voxel_size>0:
            vertex, face, label = meshUtil.voxelize_mesh(vertex, face, self.voxel_size,
                                                         seg_labels=label)

        face_index = face.to(torch.long)
        face_texture = torch.cat([vertex[face_index[:,0],3:],
                                  vertex[face_index[:,1],3:],
                                  vertex[face_index[:,2],3:]], dim=1)
        bary_coeff = torch.eye(3).repeat([face.shape[0], 1])
        num_texture = 3*torch.ones(face.shape[0], dtype=torch.int)
        vertex = vertex[:,:3]

        num_labelled = torch.sum(((label>=0) & (label<self.num_classes)).to(torch.int))
        ratio = num_labelled/vertex.shape[0]
        if ratio<0.2:
            return None
        else:
            return vertex, face, face_texture, bary_coeff, num_texture, label
        # ===============================================================================================


class TransformTest(object):
    def __init__(self, prob=0.5, num_classes=None, drop_rate=None, voxel_size=None):
        self.prob = prob
        self.num_classes = num_classes
        self.drop_rate = drop_rate     # the rate of dropping vertices
        self.voxel_size = voxel_size

    def __call__(self, args):
        vertex, face, label = args
        if self.voxel_size>0:
            vertex, face, label = meshUtil.voxelize_mesh(vertex, face, self.voxel_size,
                                                         seg_labels=label)

        face_index = face.to(torch.long)
        face_texture = torch.cat([vertex[face_index[:,0],3:],
                                  vertex[face_index[:,1],3:],
                                  vertex[face_index[:,2],3:]], dim=1)
        bary_coeff = torch.eye(3).repeat([face.shape[0], 1])
        num_texture = 3*torch.ones(face.shape[0], dtype=torch.int)
        vertex = vertex[:,:3]

        num_labelled = torch.sum(((label>=0) & (label<self.num_classes)).to(torch.int))
        ratio = num_labelled/vertex.shape[0]
        if ratio<0.2:
            return None
        else:
            return vertex, face, face_texture, bary_coeff, num_texture, label
        # ===============================================================================================


if __name__=='__main__':
    # tfrecord file_lists
    Lists = {}
    Lists['train'] = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'train_files.txt'))]
    Lists['val'] = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'val_files.txt'))]
    Lists['test'] = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'test_files.txt'))]
    Lists['train'] = Lists['train'] + Lists['val']
    print("numbers of trainval and test samples are %d and %d, respectively." % (len(Lists['train']), len(Lists['test'])))

    # build training set dataloader
    trainSet = MeshDataset(Lists['train'], opt.data_dir, rendered_data=False, transform=TransformTrain)
    trainLoader = DataLoader(trainSet, batch_size=opt.batch_size, shuffle=True, collate_fn=default_collate_fn)
    # build validation set dataloader
    valSet = MeshDataset(Lists['val'], opt.data_dir, rendered_data=False, transform=TransformTest)
    valLoader = DataLoader(valSet, batch_size=opt.batch_size, shuffle=True, collate_fn=default_collate_fn)

    # create model & Make a loss object
    model = PicassoNetII(num_class=NUM_CLASSES, mix_components=opt.num_clusters, use_height=True)

    # define the schedualer, optimizer, to train and test the model
    '''Not implemented.'''

