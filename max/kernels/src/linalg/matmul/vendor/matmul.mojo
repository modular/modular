# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from sys import align_of, simd_width_of, size_of, has_nvidia_gpu_accelerator
from sys.info import _is_sm_100x_or_newer

from algorithm import elementwise
from buffer.buffer import NDBuffer
from gpu.host import DeviceContext, get_gpu_target
from gpu.host.info import B200

from utils import Index, IndexList

from ...utils import elementwise_epilogue_type
from ...utils_gpu import MatmulConfig
from .blas import matmul as vendor_matmul


fn matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    config: Optional[MatmulConfig[a_type, b_type, c_type, transpose_b]] = None,
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[mut=False, a_type, 2, _, _],
    b: NDBuffer[mut=False, b_type, 2, _, _],
    ctx: DeviceContext,
) raises:
    """This implements the matmul kernel for the Blackwell architecture. Note
    that we do not currently have pure mojo kernels which would utilize blackwell
    architectures, so in place we just call the CUBLAS library.
    """

    comptime if not elementwise_lambda_fn:
        if not c.data:
            raise "c must be allocated"
        vendor_matmul[use_tf32=True](
            ctx, c, a, b, c_row_major=True, transpose_b=transpose_b
        )
        return
    else:
        comptime epilogue = elementwise_lambda_fn.value()
        # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
        # arch support 32B load/store to global memory, see KERN-2037.
        comptime use_32b_simd = (
            has_nvidia_gpu_accelerator()
            and ctx.default_device_info.compute >= B200.compute
        )
        comptime simd_size = 32 // size_of[c.type]() if use_32b_simd else (
            simd_width_of[c.type, target = get_gpu_target()]()
        )

        @parameter
        @__copy_capture(c)
        fn epilogue_wrapper[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var c_coord = Index(idx[0], idx[1])
            var c_val = c.load[
                width=simd_width,
                # Load takes alignment in bytes, lambda takes number of elements
                alignment = alignment * size_of[c_type](),
            ](c_coord)
            epilogue[c.type, simd_width, alignment=alignment](c_coord, c_val)

        # If c is already allocated, we can just use the vendor matmul and
        # apply the epilogue.
        if c.data:
            var m = c.dim[0]()
            var n = c.dim[1]()

            # For D = alpha * A * B + beta * C, vendor matmul currently sets
            # C to null, i.e don't fuse linear operations into gemm, KERN-1774.
            vendor_matmul[use_tf32=True](
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

        # Construct a new buffer with external origin pointing to the temporary storage.
        var c_tmp = NDBuffer[c.type, 2, MutExternalOrigin](
            rebind[UnsafePointer[Scalar[c.type], MutExternalOrigin]](
                tmp_device_buffer.unsafe_ptr()
            ),
            IndexList[2](c.dim[0](), c.dim[1]()),
        )

        matmul[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            config=config,
        ](c_tmp, a, b, ctx)

        _ = tmp_device_buffer^
