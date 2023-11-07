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
