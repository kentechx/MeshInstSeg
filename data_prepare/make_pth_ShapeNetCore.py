import numpy as np
import json
import os, sys, csv, os.path as osp
import glob
import scipy.io as sio
import argparse
import open3d as o3d
import torch
from termcolor import colored
import shutil
from tqdm import tqdm


def make_pth_cls(model_path, cls_label, store_folder):
    # load the mesh
    file_path = os.path.join(model_path, 'models/model_normalized.ply')
    if os.path.exists(file_path):
        mesh = o3d.io.read_triangle_mesh(file_path)
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
    else:
        return None

    xyz = np.asarray(mesh.vertices, dtype=np.float32)
    rot = np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    xyz = np.matmul(xyz, rot)

    face = np.asarray(mesh.triangles, dtype=np.int32)

    assert ((np.amin(face) == 0) & (np.amax(face) == (xyz.shape[0] - 1)))

    filename = os.path.join(store_folder, '%s.pth' % osp.split(model_path)[-1])
    data = {'vertex': xyz, 'face': face, 'label': cls_label}
    torch.save(data, filename)

    return xyz.shape[0], face.shape[0]


if __name__ == '__main__':
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', required=True, help='path to mesh data')
    parser.add_argument('--write_dir', required=True, help='path to json data')
    opt = parser.parse_args()

    store_folder = opt.write_dir
    os.makedirs(store_folder + '/train', exist_ok=True)
    os.makedirs(store_folder + '/val', exist_ok=True)
    os.makedirs(store_folder + '/test', exist_ok=True)

    # load the train/val/test splits
    iter = 0
    dataDict = {}
    with open(os.path.join(opt.read_dir, 'all.csv'), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if iter == 0:
                dict_keys = row
                for key in dict_keys:
                    dataDict[key] = []
            else:
                for key, content in zip(dict_keys, row):
                    dataDict[key].append(content)
            # print(row)
            iter += 1
    print(dict_keys)

    classes = sorted(list(set(dataDict[dict_keys[1]])))
    print('#classes: %d' % (len(classes)))
    print('#number of models: %d' % len(dataDict[dict_keys[1]]))
    print(classes)

    total_samples = len(dataDict[dict_keys[0]])

    num_vertex_facet = {'train': [], 'val': [], 'test': []}
    for k in tqdm(range(total_samples)):
        modelPath = os.path.join(opt.read_dir, dataDict[dict_keys[1]][k], dataDict[dict_keys[3]][k])

        className = dataDict[dict_keys[1]][k]
        cls_label = classes.index(className)
        # print(cls_label, type(cls_label), type(np.int64(cls_label)))
        # print(modelPath, className, cls_label, dataDict[dict_keys[-1]][k])
        if dataDict[dict_keys[-1]][k] == 'train':
            nV_nF = make_pth_cls(modelPath, cls_label, store_folder + '/train')
            if nV_nF is not None:
                num_vertex_facet['train'].append(nV_nF)
        elif dataDict[dict_keys[-1]][k] == 'val':
            nV_nF = make_pth_cls(modelPath, cls_label, store_folder + '/val')
            if nV_nF is not None:
                num_vertex_facet['val'].append(nV_nF)
        elif dataDict[dict_keys[-1]][k] == 'test':
            nV_nF = make_pth_cls(modelPath, cls_label, store_folder + '/test')
            if nV_nF is not None:
                num_vertex_facet['test'].append(nV_nF)

    for phase in ['train', 'val', 'test']:
        files = glob.glob(osp.join(store_folder, '%s/*.pth' % phase))
        files = [f[len(store_folder) + 1:] for f in files]

        print('\n'.join(files), file=open(osp.join(store_folder, '%s_files.txt' % phase), 'w'))
