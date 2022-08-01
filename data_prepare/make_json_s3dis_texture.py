import numpy as np
import os, sys
import json
import argparse
import open3d as o3d
from termcolor import colored
import glob
import time

parser = argparse.ArgumentParser()
parser.add_argument('--read_dir', required=True, help='path to mesh data')
parser.add_argument('--write_dir', required=True, help='path to json data')
opt = parser.parse_args()

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
           'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
num_classes = len(classes)


def make_json_seg(meshPath, store_folder="", visualize=False):
    AreaPath = os.path.dirname(meshPath)
    AreaID = meshPath.split('/')[-2]
    SceneID = meshPath.split('/')[-1].replace('_mesh.ply','')
    print(AreaID, SceneID, AreaPath)

    labelPath = meshPath.replace('_mesh.ply','.semseg')
    print('labelPath', labelPath)

    if not store_folder=="" and not os.path.exists(store_folder):
        os.mkdir(store_folder)

    print("========================make json data of s3dis %s--%s========================" % (AreaID, SceneID))
    # load mesh and the vertex labels;
    mesh = o3d.io.read_triangle_mesh(scenePath)
    mesh.compute_vertex_normals()
    seg_labels = np.loadtxt(labelPath, dtype=np.int32, delimiter=',')
    if visualize:
        o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)

    xyz = np.asarray(mesh.vertices, dtype=np.float32)
    face = np.asarray(mesh.triangles, dtype=np.int32)
    print(seg_labels.shape, xyz.shape)

    print(colored(np.unique(seg_labels), 'blue'))
    assert(np.all(seg_labels>=-1) & np.all(seg_labels<num_classes))
    assert(xyz.shape[0]==seg_labels.shape[0])
    assert((np.amin(face)==0) & (np.amax(face)==(xyz.shape[0]-1)))

    # =============================location normalization==============================
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_center[0][-1] = xyz_min[0][-1]
    xyz -= xyz_center  # align to room bottom center
    # =================================================================================

    # =============================load the fine textures==============================
    texturePath = meshPath.replace('mesh.ply','rendered_cloud.txt')
    textureCloud = np.loadtxt(texturePath,delimiter=',',dtype=np.float32)
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

    ## =======================load the barycentric coefficients=========================
    baryPath = meshPath.replace('mesh.ply','rendered_baryCoeffs.txt')
    baryCoeff = np.loadtxt(baryPath,delimiter=',',dtype=np.float32)
    if visualize:
        baryCloud = o3d.geometry.PointCloud()
        baryCloud.points = o3d.utility.Vector3dVector(baryCoeff)
        o3d.visualization.draw_geometries([baryCloud],mesh_show_back_face=True)
    # =================================================================================

    # =============================location normalization==============================
    numTexturePath = meshPath.replace('mesh.ply','rendered_numInterior.txt')
    numTexture = np.loadtxt(numTexturePath, dtype=np.int32)
    assert(face.shape[0]==numTexture.shape[0])
    uni_value, uni_cnts = np.unique(numTexture, return_counts=True)
    print(uni_value, uni_cnts)
    # =================================================================================
    assert(textureCloud.shape[0]==baryCoeff.shape[0])
    assert(textureCloud.shape[0]==np.sum(numTexture))

    print(colored('xyz: min {}, max {}'.format(np.min(xyz,axis=0),np.max(xyz,axis=0)),'green'))
    print('(%s, %16s): %d vertices, %d faces, %d in_pts_textures %d baryCoeffs, %d numTexture'
          %(AreaID,SceneID,xyz.shape[0],face.shape[0],textureCloud.shape[0],baryCoeff.shape[0],
            numTexture.shape[0]))

    filename = os.path.join(store_folder, '%s_%s.json' % (AreaID, SceneID))
    data = {'vertex':xyz, 'face':face, 'face_texture':textureCloud,
            'bary_coeff':baryCoeff, 'num_texture':numTexture, 'label':seg_labels}
    writer = open(filename, 'w')
    json.dump(data, writer)
    writer.close()
    print("======================================The End=======================================\n")

    return


if __name__=='__main__':
    folder_name = opt.read_dir.split('/')[-1]
    store_folder = os.path.join(opt.write_dir, folder_name+'Rendered')
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)

    folders = glob.glob(opt.read_dir + '/*')
    names = [os.path.basename(item) for item in folders if os.path.isdir(item)]
    print(names)

    for AreaName in names:
        scenes = glob.glob(os.path.join(opt.read_dir, AreaName, '*_mesh.ply'))
        for scenePath in scenes:
            make_json_seg(scenePath, store_folder=store_folder, visualize=False)

    for i, AreaName in enumerate(names):
        files = glob.glob(os.path.join(store_folder, '*.json'))
        testfile = open(os.path.join(store_folder, 'test_files_fold%d.txt' % (i + 1)), 'w')
        trainfile = open(os.path.join(store_folder, 'train_files_fold%d.txt' % (i + 1)), 'w')

        for filepath in files:
            if AreaName in filepath:
                testfile.write("%s\n" % filepath)
            else:
                trainfile.write("%s\n" % filepath)
        testfile.close()
        trainfile.close()


