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
from utils import Index, IndexList
from algorithm import elementwise
from sys import simdwidthof, alignof
from gpu.host._compile import _get_gpu_target


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

    @parameter
    if not elementwise_lambda_fn:
        if not c.data:
            raise "c must be allocated"
        vendor_matmul(ctx, c, a, b, c_row_major=True, transpose_b=transpose_b)
        return
    else:
        alias epilogue = elementwise_lambda_fn.value()
        alias simd_size = simdwidthof[c.type, target = _get_gpu_target()]()

        @parameter
        @__copy_capture(c)
        fn epilogue_wrapper[simd_width: Int, rank: Int](idx: IndexList[rank]):
            var c_coord = Index(idx[0], idx[1])
            var c_val = c.load[
                width=simd_width,
                alignment = alignof[SIMD[c.type, simd_width]](),
            ](c_coord)
            epilogue[c.type, simd_width](c_coord, c_val)

        # If c is already allocated, we can just use the vendor matmul and
        # apply the epilogue.
        if c.data:
            var m = c.dim[0]()
            var n = c.dim[1]()
            vendor_matmul(
                ctx, c, a, b, c_row_major=True, transpose_b=transpose_b
            )
            elementwise[epilogue_wrapper, simd_size, target="gpu"](
                Index(m, n), ctx
            )
            return

        # Otherwise, we need to allocate a new buffer for c and apply the epilogue.
        var tmp_device_buffer = ctx.enqueue_create_buffer[c.type](
            c.num_elements()
        )

        # We do not want to mark c as `mut` in the function signature, so we
        # create a new shallow copy of c as a temporary buffer.
        var c_tmp = c
        c_tmp.data = tmp_device_buffer.unsafe_ptr()

        matmul[
            use_tensor_core=use_tensor_core,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            config=config,
        ](c_tmp, a, b, ctx)

        _ = tmp_device_buffer^
