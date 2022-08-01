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


class PCloudDataset(Dataset):
    def __init__(self, file_list, pcloud_dir, transform=None):
        self.file_list = file_list
        self.pcloud_dir = pcloud_dir
        self.transform = transform
        self.keys = ['pcloud', 'label']
        self.num_keys = len(self.keys)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mesh_path = os.path.join(self.mesh_dir, self.file_list[idx])
        reader = open(mesh_path, "r")
        data = json.load(reader)
        reader.close()

        args = []
        assert(len(data.keys())==self.num_keys,"keys of input point cloud should equal self.num_keys.")
        for key in self.keys:
            args.append(torch.tensor(data[key]))
        args.insert(1, torch.tensor([data['pcloud'].shape[0]]))
        if self.transform:
            args = self.transform(*args)
        return args


def default_collate_fn(batch_data):
    batch_data = list(map(list, zip(*batch_data))) # transpose batch dimension and args dimension
    args_num = len(batch_data)
    batch_concat = []
    for k in range(args_num):
        batch_concat.append(torch.cat(batch_data[k], dim=0))
    return batch_concat








