import numpy as np
import json
import os, sys
import glob
import scipy.io as sio
import argparse
import open3d as o3d
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('--read_dir', required=True, help='path to mesh data')
parser.add_argument('--write_dir', required=True, help='path to json data')
opt = parser.parse_args()

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
           'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
num_classes = len(classes)


def make_json_seg(scenePath, store_folder="", visualize=False):
    AreaPath = os.path.dirname(scenePath)
    AreaID = scenePath.split('/')[-2]
    SceneID = scenePath.split('/')[-1].replace('.ply','')
    print('AreaPath:', AreaPath, AreaID, SceneID)

    labelPath = os.path.join(AreaPath,SceneID.replace('_mesh','')+'.semseg')
    print('labelPath', labelPath)

    if not store_folder == "" and not os.path.exists(store_folder):
        os.mkdir(store_folder)

    print("========================make json data of s3dis %s--%s========================" % (AreaID, SceneID))
    # load mesh and the vertex labels;
    mesh = o3d.io.read_triangle_mesh(scenePath)
    mesh.compute_vertex_normals()
    seg_labels = np.loadtxt(labelPath, dtype=np.int32, delimiter=',')
    if visualize:
        o3d.visualization.draw_geometries([mesh])

    xyz = np.asarray(mesh.vertices, dtype=np.float32)
    rgb = np.asarray(mesh.vertex_colors, dtype=np.float32)
    face = np.asarray(mesh.triangles, dtype=np.int32)
    print(seg_labels.shape, xyz.shape, rgb.shape)

    print(colored(np.unique(seg_labels), 'blue'))
    assert(np.all(seg_labels>=-1) & np.all(seg_labels<num_classes))
    assert((xyz.shape[0]==rgb.shape[0]) & (xyz.shape[1]==3) & (rgb.shape[1]==3))
    assert(xyz.shape[0]==seg_labels.shape[0])
    assert((np.amin(face)==0) & (np.amax(face)==(xyz.shape[0]-1)))

    # ==========================location normalization===========================
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_center[0][-1] = xyz_min[0][-1]
    xyz -= xyz_center  # align to room bottom center
    # ===========================================================================
    print(colored('xyz: min {}, max {}'.format(np.min(xyz, axis=0), np.max(xyz, axis=0)), 'green'))
    print(colored('rgb: min {}, max {}'.format(np.min(rgb, axis=0), np.max(rgb, axis=0)), 'red'))
    print('(%s, %16s): %d vertices, %d faces' % (AreaID, SceneID, xyz.shape[0], face.shape[0]))

    filename = os.path.join(store_folder, '%s_%s.json' % (AreaID, SceneID))
    data = {'vertex':np.concatenate([xyz,rgb],axis=-1), 'face':face, 'label':seg_labels}
    writer = open(filename,'w')
    json.dump(data, writer)
    writer.close()
    print("===================================The End====================================")

    return


if __name__=='__main__':
    folder_name = opt.read_dir.split('/')[-1]
    store_folder = os.path.join(opt.write_dir,folder_name)
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)

    folders = glob.glob(opt.read_dir+'/*')
    names = [os.path.basename(item) for item in folders if os.path.isdir(item)]
    print(names)

    for AreaName in names:
        scenes = glob.glob(os.path.join(opt.read_dir, AreaName, '*.ply'))
        for scenePath in scenes:
            make_json_seg(scenePath, store_folder=store_folder, visualize=False)

    for i, AreaName in enumerate(names):
        files = glob.glob(os.path.join(store_folder, '*.json'))
        testfile = open(os.path.join(store_folder, 'test_files_fold%d.txt'%(i+1)), 'w')
        trainfile = open(os.path.join(store_folder, 'train_files_fold%d.txt'%(i+1)), 'w')

        for filepath in files:
            if AreaName in filepath:
                testfile.write("%s\n" %filepath)
            else:
                trainfile.write("%s\n"%filepath)
        testfile.close()
        trainfile.close()
