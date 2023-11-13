# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import NDBuffer
from algorithm.functional import _elementwise_impl
from utils.index import StaticIntTuple
from utils._annotations import *
from MOGG import simd_load, simd_store


@mogg_register_override("mo.add", 1000)
@mogg_kgen_experiment_kernel()
@export
fn my_add[
    type: DType, rank: Int
](
    out: NDBuffer[rank, DimList.create_unknown[rank](), type],
    x: NDBuffer[rank, DimList.create_unknown[rank](), type],
    y: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    @always_inline
    @parameter
    fn func[width: Int, _rank: Int](i: StaticIntTuple[_rank]):
        let indices = rebind[StaticIntTuple[rank]](i)
        let i1 = simd_load[type, 1, rank, x.shape](x, indices)
        let i2 = simd_load[type, 1, rank, y.shape](y, indices)
        let tmp = i1 + i2
        simd_store[type, width, rank](out, indices, tmp)

    _elementwise_impl[rank, 1, True, func, target="cpu"](
        out.dynamic_shape,
        out_chain,
    )


@mogg_register_override("mo.sub", 1000)
@mogg_kgen_experiment_kernel()
@export
fn my_sub[
    type: DType, rank: Int
](
    out: NDBuffer[rank, DimList.create_unknown[rank](), type],
    x: NDBuffer[rank, DimList.create_unknown[rank](), type],
    y: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    @always_inline
    @parameter
    fn func[width: Int, _rank: Int](i: StaticIntTuple[_rank]):
        let indices = rebind[StaticIntTuple[rank]](i)
        let i1 = simd_load[type, 1, rank, x.shape](x, indices)
        let i2 = simd_load[type, 1, rank, y.shape](y, indices)
        let tmp = i1 - i2
        simd_store[type, width, rank](out, indices, tmp)

    _elementwise_impl[rank, 1, True, func, target="cpu"](
        out.dynamic_shape,
        out_chain,
    )


@mogg_register_override("no-op", 1000)
@mogg_kgen_experiment_kernel()
@export
fn noop[
    type: DType, rank: Int
](
    out: NDBuffer[rank, DimList.create_unknown[rank](), type],
    x: NDBuffer[rank, DimList.create_unknown[rank](), type],
    y: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    pass


@mogg_register_override("diff_dtype", 1000)
@mogg_kgen_experiment_kernel()
@export
fn diff_dtype[
    type1: DType, type2: DType, rank: Int
](
    out: NDBuffer[rank, DimList.create_unknown[rank](), type1],
    x: NDBuffer[rank, DimList.create_unknown[rank](), type1],
    y: NDBuffer[rank, DimList.create_unknown[rank](), type2],
    out_chain: OutputChainPtr,
):
    pass


@mogg_register_override("many_lambdas", 1000)
@mogg_kgen_experiment_kernel()
@export
fn many_lambdas[
    type: DType, rank: Int
](
    out: NDBuffer[rank, DimList.create_unknown[rank](), type],
    x: NDBuffer[rank, DimList.create_unknown[rank](), type],
    y: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    # Lambda in lambda
    @always_inline
    @parameter
    fn func1[width: Int, _rank: Int](i: StaticIntTuple[_rank]):
        let indices = rebind[StaticIntTuple[rank]](i)
        let i1 = simd_load[type, 1, rank, x.shape](x, indices)
        let i2 = simd_load[type, 1, rank, y.shape](y, indices)
        let tmp = i1 + i2
        simd_store[type, width, rank](out, indices, tmp)

    # Second lambda
    @always_inline
    @parameter
    fn func2[width: Int, _rank: Int](i: StaticIntTuple[_rank]):
        let indices = rebind[StaticIntTuple[rank]](i)
        let i1 = simd_load[type, 1, rank, x.shape](x, indices)
        let i2 = simd_load[type, 1, rank, y.shape](y, indices)
        let tmp = i1 - i2
        simd_store[type, width, rank](out, indices, tmp)
