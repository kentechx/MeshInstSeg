import os
import torch
from torch.utils.data import Dataset
import numpy as np
from picasso.augmentor import Augment
import picasso.mesh.utils as meshUtil


class TransformTrain:
    def __init__(self, prob=0.5, num_classes=None, drop_rate=0, voxel_size=None):
        self.prob = prob
        self.num_classes = num_classes
        self.drop_rate = drop_rate  # the rate of dropping vertices
        self.voxel_size = 0 if voxel_size is None else voxel_size

    def augment_fn(self, vertex, face, texture=None, vertex_label=None, face_label=None):
        assert (vertex.shape[1] == 3)
        vertex = Augment.flip_point_cloud(vertex, prob=self.prob)
        vertex = Augment.random_scale_point_cloud(vertex, scale_low=0.5, scale_high=1.5, prob=self.prob)
        vertex = Augment.rotate_point_cloud(vertex, upaxis=3, prob=self.prob)
        vertex = Augment.rotate_perturbation_point_cloud(vertex, prob=self.prob)
        vertex = Augment.jitter_point_cloud(vertex, sigma=0.01, prob=self.prob)
        if texture is not None and texture.numel() > 0:
            texture = Augment.random_drop_color(texture, prob=self.prob)
            texture = Augment.shift_color(texture, prob=self.prob)
            texture = Augment.jitter_color(texture, prob=self.prob)
            texture = Augment.auto_contrast_color(texture, prob=self.prob)
            vertex = torch.cat([vertex, texture], dim=1)  # concat texture back
        if (self.drop_rate > .0) and (self.drop_rate < 1.):
            vertex, face, label, \
            face_mask = Augment.random_drop_vertex(vertex, face, vertex_label, face_label,
                                                   drop_rate=self.drop_rate, prob=self.prob)
        else:
            label = vertex_label if face_label is None else face_label

        return vertex, face, label

    def __call__(self, vertex, face, label):
        vertex, face, label = self.augment_fn(vertex[:, :3], face, vertex[:, 3:], vertex_label=label)
        if self.voxel_size > 0:
            vertex, face, label = meshUtil.voxelize_mesh(vertex, face, self.voxel_size,
                                                         seg_labels=label)

        face_index = face.to(torch.long)
        face_texture = torch.cat([vertex[face_index[:, 0], 3:],
                                  vertex[face_index[:, 1], 3:],
                                  vertex[face_index[:, 2], 3:]], dim=1)
        bary_coeff = torch.eye(3).repeat([face.shape[0], 1])
        num_texture = 3 * torch.ones(face.shape[0], dtype=torch.int)
        vertex = vertex[:, :3]

        # return vertex, face, face_texture, bary_coeff, num_texture, label
        return vertex, face, label
        # ===============================================================================================


class TransformTest:
    def __init__(self, prob=0.5, num_classes=None, drop_rate=None, voxel_size=None):
        self.prob = prob
        self.num_classes = num_classes
        self.drop_rate = drop_rate  # the rate of dropping vertices
        self.voxel_size = 0 if voxel_size is None else voxel_size

    def __call__(self, vertex, face, label):
        if self.voxel_size > 0:
            vertex, face, label = meshUtil.voxelize_mesh(vertex, face, self.voxel_size,
                                                         seg_labels=label)

        face_index = face.to(torch.long)
        face_texture = torch.cat([vertex[face_index[:, 0], 3:],
                                  vertex[face_index[:, 1], 3:],
                                  vertex[face_index[:, 2], 3:]], dim=1)
        bary_coeff = torch.eye(3).repeat([face.shape[0], 1])
        num_texture = 3 * torch.ones(face.shape[0], dtype=torch.int)
        vertex = vertex[:, :3]

        # return vertex, face, face_texture, bary_coeff, num_texture, label
        return vertex, face, label
        # ===============================================================================================


class ShapenetCoreV2(Dataset):
    def __init__(self, data_root, file_txt, max_num_vertices=1500000, training=False):
        self.data_root = data_root
        self.max_num_vertices = max_num_vertices
        self.file_list = []
        for _txt in file_txt:
            self.file_list = [line.rstrip() for line in open(os.path.join(data_root, _txt))]
            self.file_list += [line.rstrip() for line in open(os.path.join(data_root, _txt))]

        if training:
            self.transform = TransformTrain(num_classes=55)
        else:
            self.transform = TransformTest(num_classes=55)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mesh_path = os.path.join(self.data_root, self.file_list[idx])
        data = torch.load(mesh_path)

        vertex = torch.tensor(data['vertex'])
        face = torch.tensor(data['face'])
        label = torch.tensor(data['label'])

        if self.transform:
            vertex, face, label = self.transform(vertex, face, label)

        # plain args:  vertex, face, nv, mf, label
        # render args: vertex, face, nv, mf, face_texture, bary_coeff, num_texture, label
        return {'vertex': vertex, 'face': face, 'label': label}

    def collate_fn(self, batch_data):
        batch_size = len(batch_data)
        batch_num_vertices = 0
        trunc_batch_size = 1
        for b in range(batch_size):
            batch_num_vertices += batch_data[b]['vertex'].shape[0]
            if batch_num_vertices < self.max_num_vertices:
                trunc_batch_size = b + 1
            else:
                break
        if trunc_batch_size < batch_size:
            print("The batch data is truncated.")

        batch_data = batch_data[:trunc_batch_size]

        vertex_in = torch.cat([data['vertex'] for data in batch_data], dim=0)
        face_in = torch.cat([data['face'] for data in batch_data], dim=0)
        label_in = torch.tensor([data['label'] for data in batch_data], dtype=torch.long)
        nv_in = torch.tensor([data['vertex'].shape[0] for data in batch_data], dtype=torch.int32)
        mf_in = torch.tensor([data['face'].shape[0] for data in batch_data], dtype=torch.int32)

        return {"vertex_in": vertex_in, "face_in": face_in, "label_in": label_in, "nv_in": nv_in, "mf_in": mf_in}
