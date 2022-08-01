import torch, os
from torch.utils.cpp_extension import load
srcDir = os.path.dirname(os.path.abspath(__file__))
module = load(name='pointUnpool', sources=['%s/source/pcloud_unpool3d.cpp'%srcDir,
                                           '%s/source/pcloud_unpool3d_gpu.cu'%srcDir])


class InterpolateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, nn_index):
        weight.requires_grad = False
        nn_index.requires_grad = False
        output = module.forward(input, weight, nn_index)
        variables = [input, weight, nn_index]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = module.backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, None


def interpolate(input, weight, nn_index):
        return InterpolateFunction.apply(input, weight, nn_index)

