import glob
import os, os.path as osp
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from picasso.augmentor import Augment
import picasso.mesh.utils as meshUtil


class ToothAxis(Dataset):
    def __init__(self, data_root, scenes, max_num_vertices=1500000, training=False):
        self.data_root = data_root
        self.max_num_vertices = max_num_vertices
        self.file_list = []
        self.training = training
        for scene in scenes:
            self.file_list += glob.glob(osp.join(data_root, scene, '*/3[1-7]*.stl'))

        if not training:
            self.mat = Rotation.random(len(self.file_list), random_state=42).as_matrix().astype('f4')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fp = self.file_list[idx]
        tid = int(osp.basename(fp)[:2])
        m = trimesh.load(fp)
        m.vertices -= m.vertices.mean(0)

        if self.training:
            mat = Rotation.random().as_matrix().astype('f4')
        else:
            mat = self.mat[idx]

        vertex = torch.tensor(np.array(m.vertices, dtype='f4') @ mat.T)
        face = torch.tensor(np.array(m.faces, dtype='i4'))
        label = torch.tensor(mat[:, :2].T.flatten())

        # plain args:  vertex, face, nv, mf, label
        # render args: vertex, face, nv, mf, face_texture, bary_coeff, num_texture, label
        return {'vertex': vertex, 'face': face, 'label': label, 'tid': tid, 'fp': fp}

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
        label_in = torch.stack([data['label'] for data in batch_data], dim=0)
        nv_in = torch.tensor([data['vertex'].shape[0] for data in batch_data], dtype=torch.int32)
        mf_in = torch.tensor([data['face'].shape[0] for data in batch_data], dtype=torch.int32)

        face_in = torch.cat([data['face'] + torch.sum(nv_in[:i]) for i, data in enumerate(batch_data)], dim=0)
        tids = [data['tid'] for data in batch_data]
        fps = [data['fp'] for data in batch_data]

        return {"vertex_in": vertex_in, "face_in": face_in, "label_in": label_in, "nv_in": nv_in, "mf_in": mf_in,
                "tids": tids, "fps": fps}
