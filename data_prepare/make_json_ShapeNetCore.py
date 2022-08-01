import numpy as np
import json
import os, sys, csv
import glob
import scipy.io as sio
import argparse
import open3d as o3d
from termcolor import colored
import shutil
from tqdm import tqdm
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


parser = argparse.ArgumentParser()
parser.add_argument('--read_dir', required=True, help='path to mesh data')
parser.add_argument('--write_dir', required=True, help='path to json data')
opt = parser.parse_args()


def make_json_cls(model_path, cls_label, store_folder, visualize=False):
    # load the mesh
    file_path = os.path.join(model_path,'models/model_normalized.ply')
    if os.path.exists(file_path):
        mesh = o3d.io.read_triangle_mesh(file_path)
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
    else:
        return None
    if visualize:
        lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        o3d.visualization.draw_geometries([mesh, lineset], mesh_show_back_face=True)

    xyz = np.asarray(mesh.vertices, dtype=np.float32)
    rot = np.asarray([[0,1,0],[0,0,1],[1,0,0]], dtype=np.float32)
    xyz = np.matmul(xyz,rot)

    face = np.asarray(mesh.triangles, dtype=np.int32)
    # mesh.vertices = o3d.utility.Vector3dVector(xyz)
    # o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)

    print(colored('xyz: min {}, max {}'.format(np.min(xyz, axis=0), np.max(xyz, axis=0)), 'green'))
    print('(%d,%d) vertices, (%d,%d) faces'%(xyz.shape[0], xyz.shape[1],
                                             face.shape[0], face.shape[1]))
    assert((np.amin(face)==0) & (np.amax(face)==(xyz.shape[0]-1)))

    filename = os.path.join(store_folder, '%s.json'%model_path.split('/')[-1])
    data = {'vertex':xyz, 'face':face, 'label':cls_label}
    writer = open(filename, 'w')
    json.dump(data, writer)
    writer.close()
    print("===================================The End====================================")

    return xyz.shape[0], face.shape[0]


if __name__=='__main__':
    store_folder = opt.write_dir
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)
        os.mkdir(store_folder+'/train')
        os.mkdir(store_folder+'/val')
        os.mkdir(store_folder+'/test')

    # load the train/val/test splits
    iter =0
    dataDict = {}
    with open(os.path.join(opt.read_dir, 'all.csv'), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if iter==0:
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
    print('#classes: %d'%(len(classes)))
    print('#number of models: %d'%len(dataDict[dict_keys[1]]))
    print(classes)

    total_samples = len(dataDict[dict_keys[0]])
    LOG_train = open(os.path.join(store_folder, 'log_shape_train.txt'), 'a')
    LOG_val   = open(os.path.join(store_folder, 'log_shape_val.txt'), 'a')
    LOG_test  = open(os.path.join(store_folder, 'log_shape_test.txt'), 'a')

    num_vertex_facet = {'train':[], 'val':[], 'test':[]}
    for k in tqdm(range(total_samples)):
        modelPath = os.path.join(opt.read_dir, dataDict[dict_keys[1]][k], dataDict[dict_keys[3]][k])

        className = dataDict[dict_keys[1]][k]
        cls_label = classes.index(className)
        print(cls_label, type(cls_label), type(np.int64(cls_label)))
        print(modelPath, className, cls_label, dataDict[dict_keys[-1]][k])
        if dataDict[dict_keys[-1]][k]=='train':
            nV_nF = make_json_cls(modelPath, cls_label, store_folder+'/train', LOG_train, visualize=False)
            if nV_nF is not None:
                num_vertex_facet['train'].append(nV_nF)
        elif dataDict[dict_keys[-1]][k]=='val':
            nV_nF = make_json_cls(modelPath, cls_label, store_folder+'/val', LOG_val, visualize=False)
            if nV_nF is not None:
                num_vertex_facet['val'].append(nV_nF)
        elif dataDict[dict_keys[-1]][k]=='test':
            nV_nF = make_json_cls(modelPath, cls_label, store_folder+'/test', LOG_test, visualize=False)
            if nV_nF is not None:
                num_vertex_facet['test'].append(nV_nF)
    #
    # num_vertex_facet['train'] = np.asarray(num_vertex_facet['train'], dtype=np.int32)
    # num_vertex_facet['val']   = np.asarray(num_vertex_facet['val'],   dtype=np.int32)
    # num_vertex_facet['test']  = np.asarray(num_vertex_facet['test'],  dtype=np.int32)
    # print(np.amin(num_vertex_facet['train'],axis=0), np.amax(num_vertex_facet['train'],axis=0), num_vertex_facet['train'].shape)
    # print(np.amin(num_vertex_facet['val'], axis=0), np.amax(num_vertex_facet['val'], axis=0),   num_vertex_facet['val'].shape)
    # print(np.amin(num_vertex_facet['test'], axis=0), np.amax(num_vertex_facet['test'], axis=0), num_vertex_facet['test'].shape)
    # np.savetxt("ShapeNetCore_train_meshSize.txt", num_vertex_facet['train'], fmt="%d", delimiter=',')
    # np.savetxt("ShapeNetCore_val_meshSize.txt",   num_vertex_facet['val'],   fmt="%d", delimiter=',')
    # np.savetxt("ShapeNetCore_test_meshSize.txt",  num_vertex_facet['test'],  fmt="%d", delimiter=',')

    for phase in ['train','val','test']:
        files = glob.glob(os.path.join(store_folder, '%s/*.json'%phase))
        file_list = open(os.path.join(store_folder, '%s_files.txt'%phase), 'w')

        for filepath in files:
            file_list.write("%s\n" %filepath)
        file_list.close()