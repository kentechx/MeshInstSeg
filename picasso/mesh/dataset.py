import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np
import json
from picasso.augmentor import Augment
import picasso.mesh.utils as meshUtil


class MeshDataset(Dataset):
    def __init__(self, file_list, mesh_dir, rendered_data=False, transform=None):
        self.file_list = file_list
        self.mesh_dir = mesh_dir
        self.transform = transform
        if rendered_data:
            self.keys = ['verex', 'face', 'face_texture', 'bary_coeff', 'num_texture', 'label']
            self.num_keys = len(self.keys)
        else:
            self.keys = ['verex', 'face', 'label']
            self.num_keys = len(self.keys)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mesh_path = os.path.join(self.mesh_dir, self.file_list[idx])
        reader = open(mesh_path, "r")
        data = json.load(reader)
        reader.close()

        args = []
        assert(len(data.keys())==self.num_keys,"keys of input mesh should equal self.num_keys.")
        for key in self.keys:
            args.append(torch.tensor(data[key]))
        args.insert(2, torch.tensor([data['vertex'].shape[0]]))
        args.insert(3, torch.tensor([data['face'].shape[0]]))
        if self.transform:
            args = self.transform(*args)

        # plain args:  vertex, face, nv, mf, label
        # render args: vertex, face, nv, mf, face_texture, bary_coeff, num_texture, label
        return args


class default_collate_fn:
    def __init__(self, max_num_vertices):
        self.max_num_vertices = max_num_vertices

    def __call__(self, batch_data):
        batch_size = len(batch_data)
        batch_num_vertices = 0
        trunc_batch_size = 1
        for b in range(batch_size):
            batch_num_vertices += batch_data[b][2][0]
            if batch_num_vertices<self.max_num_vertices:
                trunc_batch_size = b+1
            else:
                break
        if trunc_batch_size<batch_size:
            print("The batch data is truncated.")

        batch_data = batch_data[trunc_batch_size]
        batch_data = list(map(list, zip(*batch_data))) # transpose batch dimension and args dimension
        args_num = len(batch_data)
        batch_concat = []
        for k in range(args_num):
            batch_concat.append(torch.cat(batch_data[k], dim=0))
        return batch_concat








