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
from picasso.models.scene_seg_texture import PicassoNetII
from picasso.mesh.dataset import default_collate_fn
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to tfrecord data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../log_scannet_plain', help='Log dir [default: log_scannet_plain]')
parser.add_argument('--ckpt_epoch', type=int, default=None, help='epoch model to load [default: None]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--max_num_vertices', type=int, default=1500000, help='maximum vertices allowed in a batch')
parser.add_argument('--voxel_size', type=int, default=2, help='Voxel Size to downsample input mesh')
parser.add_argument('--num_clusters', type=int, default=27, help='number of cluster components')
opt = parser.parse_args()


LOG_DIR = opt.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (os.path.basename(__file__), LOG_DIR))     # bkp of train procedure
os.system('cp picasso/models/scene_seg_texture.py %s'%LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')

# labels, classnames, and colormaps
NUM_CLASSES = 20
labels = [*range(NUM_CLASSES)]
classnames = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
              'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator',
              'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
colormap = [(174, 199, 232), (152, 223, 138), (31, 119, 180),  (255, 187, 120),
            (188, 189, 34),  (140, 86, 75),   (255, 152, 150), (214, 39, 40),
            (197, 176, 213), (148, 103, 189), (196, 156, 148), (23, 190, 207),
            (247, 182, 210), (219, 219, 141), (255, 127, 14), (158, 218, 229),
            (44, 160, 44),   (112, 128, 144), (227, 119, 194), (82, 84, 163)]


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
        vertex = Augment.rotate_point_cloud(vertex, upaxis=3, prob=self.prob)
        vertex = Augment.rotate_perturbation_point_cloud(vertex, prob=self.prob)
        vertex = Augment.flip_point_cloud(vertex, prob=self.prob)
        vertex = Augment.random_scale_point_cloud(vertex, prob=self.prob)
        vertex = Augment.shift_point_cloud(vertex, prob=self.prob)
        if texture is not None:
            texture = Augment.clip_color(texture, prob=self.prob)
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


def evaluate_iou(gt_labels, pred_labels):
    class_names = classnames[:20]
    total_seen_class = {cat: 0 for cat in class_names}
    total_correct_class = {cat: 0 for cat in class_names}
    total_union_class = {cat: 0 for cat in class_names}

    for l, cat in enumerate(class_names):
        total_seen_class[cat] += np.sum(gt_labels==l)
        total_union_class[cat] += (np.sum((pred_labels==l) | (gt_labels==l)))
        total_correct_class[cat] += (np.sum((pred_labels==l) & (gt_labels==l)))

    class_iou = {cat: 0.0 for cat in class_names}
    class_acc = {cat: 0.0 for cat in class_names}
    for cat in class_names:
        class_iou[cat] = total_correct_class[cat] / (float(total_union_class[cat]) + np.finfo(float).eps)
        class_acc[cat] = total_correct_class[cat] / (float(total_seen_class[cat]) + np.finfo(float).eps)

    total_correct = sum(list(total_correct_class.values()))
    total_seen = sum(list(total_seen_class.values()))
    log_string('eval overall class accuracy:\t %d/%d=%3.2f'%(total_correct, total_seen,
                                                             100*total_correct/float(total_seen)))
    log_string('eval average class accuracy:\t %3.2f'%(100*np.mean(list(class_acc.values()))))
    for cat in class_names:
        log_string('eval mIoU of %14s:\t %3.2f'%(cat, 100*class_iou[cat]))
    log_string('eval mIoU of all %d classes:\t %3.2f'%(NUM_CLASSES, 100*np.mean(list(class_iou.values()))))



if __name__=='__main__':
    # json file_lists
    Lists = {}
    Lists['train'] = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'train_files.txt'))]
    Lists['val']   = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'val_files.txt'))]
    Lists['test']  = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'test_files.txt'))]
    num_repeat = np.ceil((600*opt.batch_size) / len(Lists['train'])).astype('int32')
    Lists['train'] = Lists['train']*num_repeat

    # build training set dataloader
    trainSet = MeshDataset(Lists['train'], opt.data_dir, rendered_data=False, transform=TransformTrain)
    trainLoader = DataLoader(trainSet, batch_size=opt.batch_size, shuffle=True,
                             collate_fn=default_collate_fn(opt.max_num_vertices))
    # build validation set dataloader
    valSet = MeshDataset(Lists['val'], opt.data_dir, rendered_data=False, transform=TransformTest)
    valLoader = DataLoader(valSet, batch_size=opt.batch_size, shuffle=True,
                           collate_fn=default_collate_fn(opt.max_num_vertices))

    # create model & Make a loss object
    model = PicassoNetII(num_class=NUM_CLASSES, mix_components=opt.num_clusters, use_height=True)

    # define the schedualer, optimizer, to train and test the model
    '''Not implemented.'''
