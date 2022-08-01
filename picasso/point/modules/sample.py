import torch, os
from torch.utils.cpp_extension import load
srcDir = os.path.dirname(os.path.abspath(__file__))
module = load(name='pointSample', sources=['%s/source/sample.cpp'%srcDir,
                                           '%s/source/sample_gpu.cu'%srcDir])


def fps_(xyz_in, nv_in, nv_out):
    '''
    input:
        database:   (Np, 3) float32 array, database points
        nvDatabase: (batch_size) int32 vector, number of input points of each batch sample
        nv_out: (batch_size) int32, number of output points of each batch sample
    returns:
        sample_index: (Mp) int32 array, index of sampled neurons in the database
    '''
    nv_in = torch.cumsum(nv_in, dim=-1, dtype=torch.int32)
    nv_out = torch.cumsum(nv_out, dim=-1, dtype=torch.int32)
    sample_index = module.fps(xyz_in, nv_in, nv_out)
    return sample_index

