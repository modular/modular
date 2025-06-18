# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from sys import alignof, simdwidthof

from algorithm import elementwise
from buffer.buffer import NDBuffer
from gpu.host import DeviceContext
from gpu.host import get_gpu_target

from utils import Index, IndexList

from .utils import elementwise_epilogue_type
from .utils_gpu import MatmulConfig
from .vendor_blas import matmul as vendor_matmul


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
    _trace_description: StaticString = "",
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
        alias simd_size = simdwidthof[c.type, target = get_gpu_target()]()

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

            # For D = alpha * A * B + beta * C, vendor matmul currently sets
            # C to null, i.e don't fuse linear operations into gemm, KERN-1774.
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
        c_tmp.data = tmp_device_buffer._unsafe_ptr()

        matmul[
            use_tensor_core=use_tensor_core,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            config=config,
        ](c_tmp, a, b, ctx)

        _ = tmp_device_buffer^
