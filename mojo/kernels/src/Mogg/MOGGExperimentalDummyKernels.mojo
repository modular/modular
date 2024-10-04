# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer.dimlist import DimList
from MOGGExperimental import empty_tensor
from MOGGIntList import IntList
from MOGGTensor import Tensor
from nn.activations import relu
from register import *

from utils import StaticIntTuple, unroll

# Dummy kernels to test fundamental mechanisms without overwriting the normal
# kernels. We duplicate some of the kernels here, allowing FileCheck tests to
# only include this dummy library for testing purpose. In this case, even later
# we implement these dummy kernels in the MOGGExperimental.mojo, those tests
# can remain unchanged.

alias MAX_BENEFIT = 1000


@mogg_register_override("mo.matmul", MAX_BENEFIT)
@export
fn mo_matmul[
    out_shape: DimList,
    transpose_b: Bool,
    packed_b: Bool,
    lambdas_have_fusion: Bool = False,
](x: Tensor, y: Tensor) -> Tensor[x.type, out_shape]:
    var shape = IntList[out_shape](x.shape[0], y.shape[1])
    var out = empty_tensor[x.type](shape)

    print(transpose_b)
    print(packed_b)
    print(lambdas_have_fusion)

    x.shape.print()
    y.shape.print()
    out.shape.print()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return 3.33

    out.for_each[func]()

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    return out


@mogg_register("test_my_cumsum")
@export
fn test_my_cumsum(x: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)
    var dim = 1
    x.enable_fusion()
    out.enable_fusion()

    var start = 0
    var stop = x.shape[dim]
    var step = 1

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return relu(x.simd_load[width](i))

    out.for_each[func]()

    var indices = x.get_nd_indices()
    var val: SIMD[x.type, 1] = 0
    for i in range(start, stop, step):
        indices[dim] = i
        val += x.simd_load[1](indices)
        out.store(indices, val)

    return out


@mogg_register_override("mo.cast", MAX_BENEFIT)
@export
fn mo_cast_si32(
    x: Tensor,
) -> Tensor[DType.int32, x.static_shape]:
    var out = empty_tensor[DType.int32](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return rebind[SIMD[out.type, width]](
            x.simd_load[width](i).cast[DType.int32]()
        )

    out.for_each[func]()
    return out


@mogg_register_override("mo.transpose", MAX_BENEFIT)
@export
fn transpose[
    out_static_strides: DimList,
](
    x: Tensor,
    perm: Tensor,
) -> Tensor[
    x.type, x.same_rank_param(), out_static_strides
]:
    # Currently we don't support alias.
    # alias rank = x.static_rank

    var new_shape = IntList[x.same_rank_param()]()
    var new_stride = IntList[x.same_rank_param()]()

    @always_inline
    @parameter
    fn body[i: Int]():
        var index = IntList[DimList(i)](i)
        var dim = int(perm.simd_load[1](index))
        new_shape[i] = x.shape[dim]
        new_stride[i] = x.strides[dim]

    unroll[body, x.static_rank.value()]()

    return Tensor[x.type, x.same_rank_param(), out_static_strides](
        x.data, new_shape, new_stride, x.refcount()
    )


@mogg_register_override("mo.abs", MAX_BENEFIT)
@export
fn mo_abs(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return abs(x.simd_load[width](i))

    out.for_each[func]()
    return out


@mogg_register_override("mo.static.broadcast_to", MAX_BENEFIT)
@export
fn broadcast[
    rank: Int,
    out_static_strides: DimList,
](x: Tensor, shape: StaticIntTuple[rank]) -> Tensor[
    x.type, DimList.create_unknown[shape.size](), out_static_strides
]:
    var new_shape = IntList[DimList.create_unknown[shape.size]()]()
    var new_stride = IntList[DimList.create_unknown[shape.size]()]()

    var delta = shape.size - x.rank()

    @always_inline
    @__copy_capture(delta)
    @parameter
    fn body[i: Int]():
        new_shape[i] = shape[i]

        if i < delta:
            new_stride[i] = 0
        elif x.shape[i - delta] <= 1:
            new_stride[i] = 0
        else:
            new_stride[i] = x.strides[i - delta]

    unroll[body, shape.size]()

    return Tensor[
        x.type, DimList.create_unknown[shape.size](), out_static_strides
    ](x.data, new_shape, new_stride, x.refcount())


@mogg_register_override("mo.add", MAX_BENEFIT)
@export
fn mo_add(x: Tensor, y: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[x.type, width]:
        var i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return x.simd_load[width](i) + i2

    out.for_each[func]()

    return out
