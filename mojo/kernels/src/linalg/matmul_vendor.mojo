# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from buffer.buffer import NDBuffer
from gpu.host import DeviceContext
from .vendor_blas import matmul as vendor_matmul
from .utils_gpu import MatmulConfig
from .utils import apply_epilogue, elementwise_epilogue_type
from utils import IndexList
from algorithm import elementwise
from sys import simdwidthof
from gpu.host._compile import _get_gpu_target


@always_inline
fn matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType, //,
    use_tensor_core: Bool = False,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    config: OptionalReg[
        MatmulConfig[a_type, b_type, c_type, transpose_b]
    ] = None,
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises:
    """This implements the matmul kernel for the Blackwell architecture. Note
    that we do not currently have pure mojo kernels which would utilize blackwell
    architectures, so in place we just call the CUBLAS library.
    """

    vendor_matmul(ctx, c, a, b, c_row_major=True, transpose_b=transpose_b)

    @parameter
    if elementwise_lambda_fn:
        var m = c.dim[0]()
        var n = c.dim[1]()
        alias epilogue = elementwise_lambda_fn.value()
        alias simd_size = simdwidthof[c.type, target = _get_gpu_target()]()

        @always_inline
        @parameter
        fn epilogue_on_col_chunk[
            simd_width: Int, rank: Int
        ](idx: IndexList[rank]):
            var c_coord = IndexList[2](idx[0], idx[1])
            var c_val = c.load[width=simd_width](c_coord)
            epilogue[c.type, simd_width](c_coord, c_val)

        elementwise[epilogue_on_col_chunk, simd_size, target="gpu"](
            IndexList[2](m, n), ctx
        )
