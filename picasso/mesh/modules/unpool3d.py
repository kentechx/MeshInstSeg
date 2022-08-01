import torch, os
from torch.utils.cpp_extension import load
srcDir = os.path.dirname(os.path.abspath(__file__))
module = load(name='unpool', sources=['%s/source/mesh_unpool3d.cpp'%srcDir,
                                      '%s/source/mesh_unpool3d_gpu.cu'%srcDir])


class InterpolateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vt_replace, vt_map):
        vt_replace.requires_grad = False
        vt_map.requires_grad = False
        output = module.forward(input, vt_replace, vt_map)
        variables = [input, vt_replace, vt_map]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = module.backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, None


def interpolate(input, vt_replace, vt_map):
        return InterpolateFunction.apply(input, vt_replace, vt_map)




