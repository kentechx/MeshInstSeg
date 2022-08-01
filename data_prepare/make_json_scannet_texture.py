import numpy as np
import os, sys
import json
import argparse
import open3d as o3d
from termcolor import colored
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--read_dir', required=True, help='path to mesh data')
parser.add_argument('--write_dir', required=True, help='path to json data')
opt = parser.parse_args()

# labels of ignored an interested classes
ignored_cls = 0
interest_cls = [*range(1,13), 14, 16, 24, 28, 33, 34, 36, 39]
num_classes = len(interest_cls)
new_labels = [*range(num_classes)]
print('ignored_cls:',ignored_cls)
print('interest_cls:',interest_cls)
print('new_labels:',new_labels)


def make_json_seg(scenePath, phase, store_folder="", visualize=False):
    scene_name = os.path.basename(scenePath)

    print("========================make json data of scannet %s======================="%scene_name)
    print("start to make %s.json:"%scene_name)
    # ===================================Load Meshes===================================
    meshPath = os.path.join(scenePath, '%s_vh_clean_2.ply'%scene_name)
    mesh = o3d.io.read_triangle_mesh(meshPath)
    xyz = np.asarray(mesh.vertices,dtype=np.float32)
    face = np.asarray(mesh.triangles,dtype=np.int32)
    if visualize:
        o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)
    # =================================================================================

    # ========================get semantic segmentation labels=========================
    labelPath = meshPath.replace('vh_clean_2.ply','semantic_labels.txt')
    if os.path.exists(labelPath):
        seg_labels =  np.loadtxt(labelPath, dtype=np.int32)
    else:
        seg_labels = -np.ones(shape=[xyz.shape[0]], dtype=np.int32)
    assert(xyz.shape[0]==seg_labels.shape[0])
    # =================================================================================

    # =============================location normalization==============================
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    xyz_center = (xyz_min+xyz_max)/2
    xyz_center[0][-1] = xyz_min[0][-1]
    xyz -= xyz_center  # align to room bottom center
    # =================================================================================

    # =============================load the fine textures==============================
    cloudPath = meshPath.replace('vh_clean_2.ply', 'rendered_cloud.txt')
    textureCloud = np.loadtxt(cloudPath, delimiter=',', dtype=np.float32)
    textureCloud[:,:3] -= xyz_center
    textureCloud[:,3:] /= 255
    if visualize:
        mesh.vertices = o3d.utility.Vector3dVector(xyz)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(textureCloud[:,:3])
        cloud.colors = o3d.utility.Vector3dVector(textureCloud[:,3:])
        o3d.visualization.draw_geometries([cloud],mesh_show_back_face=True)
        o3d.visualization.draw_geometries([cloud,mesh], mesh_show_back_face=True)
    # =================================================================================
    textureCloud = textureCloud[:,3:] # (x,y,z) can be recovered use baryCoeff

    # =======================load the barycentric coefficients=========================
    baryPath = meshPath.replace('vh_clean_2.ply', 'rendered_baryCoeffs.txt')
    baryCoeff = np.loadtxt(baryPath, delimiter=',', dtype=np.float32)
    if visualize:
        baryCloud = o3d.geometry.PointCloud()
        baryCloud.points = o3d.utility.Vector3dVector(baryCoeff)
        o3d.visualization.draw_geometries([baryCloud],mesh_show_back_face=True)
    # =================================================================================

    # =============================location normalization==============================
    numinteriorPath = meshPath.replace('vh_clean_2.ply', 'rendered_numInterior.txt')
    numTexture = np.loadtxt(numinteriorPath, dtype=np.int32)
    assert (face.shape[0]==numTexture.shape[0])
    # =================================================================================
    assert(textureCloud.shape[0]==baryCoeff.shape[0])
    assert(textureCloud.shape[0]==np.sum(numTexture))

    print(colored('xyz: min {}, max {}'.format(np.min(xyz,axis=0),np.max(xyz,axis=0)),'green'))
    print('%s: %d vertices, %d faces, %d in_pts_textures %d baryCoeff, %d numTexture'
          %(scene_name,xyz.shape[0],face.shape[0],textureCloud.shape[0],baryCoeff.shape[0],
            numTexture.shape[0]))

    filename = os.path.join(store_folder, phase, '%s.json' % (scene_name))
    data = {'vertex':xyz, 'face':face, 'face_texture':textureCloud,
            'bary_coeff':baryCoeff, 'num_texture':numTexture, 'label':seg_labels}
    writer = open(filename, 'w')
    json.dump(data, writer)
    writer.close()
    print("======================================The End=======================================\n")

    return


if __name__=='__main__':
    store_folder = opt.write_dir
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)

    Lists = {}
    Lists['train'] = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'scannetv2_train.txt'))]
    Lists['val']   = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'scannetv2_val.txt'))]
    Lists['test']  = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'scannetv2_test.txt'))]

    for phase in ['test']:
        fileList = Lists[phase]
        if phase=='test':
            folder = 'scans_test'
        else:
            folder = 'scans_trainval'

        if not os.path.exists(os.path.join(store_folder,phase)):
            os.makedirs(os.path.join(store_folder,phase))

        for scene_name in fileList:
            scenePath = os.path.join(opt.data_dir,folder,scene_name)
            make_json_seg(scenePath, phase, store_folder=store_folder, visualize=False)

    jsonList = {}
    jsonList['train'] = glob.glob(os.path.join(store_folder, 'train', '*.json'))
    jsonList['val'] = glob.glob(os.path.join(store_folder, 'val', '*.json'))
    jsonList['test'] = glob.glob(os.path.join(store_folder, 'test', '*.json'))
    trainfile = open(os.path.join(store_folder, 'train_files.txt'), 'w')
    valfile = open(os.path.join(store_folder, 'val_files.txt'), 'w')
    testfile = open(os.path.join(store_folder, 'test_files.txt'), 'w')
    for filepath in jsonList['train']:
        trainfile.write("%s\n" % filepath)
    for filepath in jsonList['val']:
        valfile.write("%s\n" % filepath)
    for filepath in jsonList['test']:
        testfile.write("%s\n" % filepath)
    trainfile.close()
    valfile.close()
    testfile.close()


