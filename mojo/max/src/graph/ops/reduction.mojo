# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.graph.type import ElementType, MOTensor, dyn


# ===----------------------------------------------------------------------=== #
# Basic reductions
# ===----------------------------------------------------------------------=== #


fn mean(v: Symbol, axis: Int) raises -> Symbol:
    var g = v.graph()
    var v_type = v.tensor_type()

    var norm_axis = axis
    if norm_axis < 0:
        norm_axis += v_type.rank()
    if norm_axis < 0 or norm_axis >= v_type.rank():
        raise "axis out of range"

    v_type.dims[norm_axis] = 1

    return g.op("mo.mean", (v, g.scalar(Int64(norm_axis))), v_type)


# ===----------------------------------------------------------------------=== #
# Arg* reductions
# ===----------------------------------------------------------------------=== #


fn arg_max(v: Symbol, axis: Int) raises -> Symbol:
    var g = v.graph()
    var v_type = v.tensor_type()

    var norm_axis = axis
    if norm_axis < 0:
        norm_axis += v_type.rank()
    if norm_axis < 0 or norm_axis >= v_type.rank():
        raise "axis out of range"

    v_type.dtype = ElementType(DType.int64)
    v_type.dims[norm_axis] = 1

    return g.op("mo.arg_max", (v, g.scalar(Int64(norm_axis))), v_type)
