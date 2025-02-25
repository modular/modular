# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from random import random_float64
from sys import is_nvidia_gpu

from utils.numerics import max_finite


fn _filler_impl[
    dtype: DType,
    filler: fn (i: Int) capturing [_] -> Scalar[dtype],
    layout: Layout,
](tensor: LayoutTensor[dtype, layout, mut=True, **_]):
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


fn arange[
    dtype: DType, layout: Layout
](
    tensor: LayoutTensor[dtype, layout, mut=True, **_],
    start: Scalar[tensor.dtype] = 0,
    step: Scalar[tensor.dtype] = 1,
    end: Scalar[tensor.dtype] = max_finite[tensor.dtype](),
):
    @parameter
    fn filler(i: Int) -> Scalar[tensor.dtype]:
        return (i * step + start) % end

    # Use layout info for 2D tensors
    @parameter
    if len(tensor.layout) != 2:
        _filler_impl[tensor.dtype, filler](tensor)
    else:
        for m in range(tensor.runtime_layout.shape[0].value[0]):
            for n in range(tensor.runtime_layout.shape[1].value[0]):
                tensor[m, n] = (
                    (m * tensor.runtime_layout.shape[1].value[0] + n) * step
                    + start
                ) % end


fn random[
    dtype: DType, layout: Layout
](
    tensor: LayoutTensor[dtype, layout, mut=True, **_],
    min: Scalar[tensor.dtype] = 0,
    max: Scalar[tensor.dtype] = 1,
):
    constrained[not is_nvidia_gpu(), "Cannot run random on the gpu"]()

    @parameter
    fn filler(i: Int) -> Scalar[tensor.dtype]:
        return random_float64(
            min.cast[DType.float64](), max.cast[DType.float64]()
        ).cast[tensor.dtype]()

    _filler_impl[tensor.dtype, filler](tensor)
