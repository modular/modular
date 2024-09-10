# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from random import random_float64
from sys import triple_is_nvidia_cuda


fn _filler_impl[
    dtype: DType, filler: fn (i: Int) capturing -> Scalar[dtype]
](tensor: LayoutTensor):
    @parameter
    if tensor.layout.all_dims_known():
        alias num_elements = tensor.layout.size() * tensor.element_size

        @parameter
        for i in range(num_elements):
            var val = filler(i)
            tensor.ptr[i] = val.cast[tensor.dtype]()
    else:
        var num_elements = tensor.runtime_layout.size() * tensor.element_size
        for i in range(num_elements):
            var val = filler(i)
            tensor.ptr[i] = val.cast[tensor.dtype]()


fn arange(
    tensor: LayoutTensor,
    start: Scalar[tensor.dtype] = 0,
    step: Scalar[tensor.dtype] = 1,
):
    fn filler(i: Int) capturing -> Scalar[tensor.dtype]:
        return i * step + start

    # Use layout info for 2D tensors
    @parameter
    if len(tensor.layout) != 2:
        _filler_impl[tensor.dtype, filler](tensor)
    else:
        for m in range(tensor.runtime_layout.shape[0].value[0]):
            for n in range(tensor.runtime_layout.shape[1].value[0]):
                tensor[m, n] = (
                    m * tensor.runtime_layout.shape[1].value[0] + n
                ) * step + start


fn random(
    tensor: LayoutTensor,
    min: Scalar[tensor.dtype] = 0,
    max: Scalar[tensor.dtype] = 1,
):
    constrained[not triple_is_nvidia_cuda(), "Cannot run random on the gpu"]()

    fn filler(i: Int) capturing -> Scalar[tensor.dtype]:
        return random_float64(
            min.cast[DType.float64](), max.cast[DType.float64]()
        ).cast[tensor.dtype]()

    _filler_impl[tensor.dtype, filler](tensor)
