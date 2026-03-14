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

from std.sys import align_of, simd_width_of, size_of, has_nvidia_gpu_accelerator
from std.sys.info import _is_sm_100x_or_newer

from std.algorithm import elementwise
from buffer.buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.host.info import B200
from layout import TileTensor, coord_to_index_list

from std.utils import Index, IndexList

from ...utils import elementwise_epilogue_type
from .blas import matmul as vendor_matmul


def matmul[
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](c: TileTensor, a: TileTensor, b: TileTensor, ctx: DeviceContext,) raises:
    """TileTensor overload of the vendor matmul dispatch. Constructs
    NDBuffers internally since the vendor BLAS library requires them.
    """
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"

    comptime c_type = c.dtype
    comptime a_type = a.dtype
    comptime b_type = b.dtype

    # Construct NDBuffers for vendor BLAS calls.
    comptime to_dim[i: Int] = Dim(i) if i > -1 else Dim()
    comptime c_ndbuf_shape = DimList[
        to_dim[c.static_shape[0]], to_dim[c.static_shape[1]]
    ]()
    var c_buf = NDBuffer[rank=2, c_type, MutAnyOrigin, c_ndbuf_shape](
        c.ptr.bitcast[Scalar[c_type]]().as_any_origin(),
        rebind[IndexList[2]](coord_to_index_list(c.layout.shape_coord())),
    )
    comptime a_ndbuf_shape = DimList[
        to_dim[a.static_shape[0]], to_dim[a.static_shape[1]]
    ]()
    var a_buf = NDBuffer[rank=2, a_type, MutAnyOrigin, a_ndbuf_shape](
        a.ptr.bitcast[Scalar[a_type]]().as_any_origin(),
        rebind[IndexList[2]](coord_to_index_list(a.layout.shape_coord())),
    )
    comptime b_ndbuf_shape = DimList[
        to_dim[b.static_shape[0]], to_dim[b.static_shape[1]]
    ]()
    var b_buf = NDBuffer[rank=2, b_type, MutAnyOrigin, b_ndbuf_shape](
        b.ptr.bitcast[Scalar[b_type]]().as_any_origin(),
        rebind[IndexList[2]](coord_to_index_list(b.layout.shape_coord())),
    )

    comptime ImmA = NDBuffer[rank=2, a_type, ImmutAnyOrigin, a_ndbuf_shape]
    comptime ImmB = NDBuffer[rank=2, b_type, ImmutAnyOrigin, b_ndbuf_shape]

    comptime if not elementwise_lambda_fn:
        if not c_buf.data:
            raise "c must be allocated"
        vendor_matmul[use_tf32=True](
            ctx,
            c_buf,
            rebind[ImmA](a_buf),
            rebind[ImmB](b_buf),
            c_row_major=True,
            transpose_b=transpose_b,
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
        comptime simd_size = 32 // size_of[c_type]() if use_32b_simd else (
            simd_width_of[c_type, target=get_gpu_target()]()
        )

        @parameter
        @__copy_capture(c_buf)
        def epilogue_wrapper[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var c_coord = Index(idx[0], idx[1])
            var c_val = c_buf.load[
                width=simd_width,
                # Load takes alignment in bytes, lambda takes number of elements
                alignment=alignment * size_of[c_type](),
            ](c_coord)
            epilogue[c_type, simd_width, alignment=alignment](c_coord, c_val)

        # If c is already allocated, we can just use the vendor matmul and
        # apply the epilogue.
        if c_buf.data:
            var m = Int(c.dim[0]())
            var n = Int(c.dim[1]())

            # For D = alpha * A * B + beta * C, vendor matmul currently sets
            # C to null, i.e don't fuse linear operations into gemm, KERN-1774.
            vendor_matmul[use_tf32=True](
                ctx,
                c_buf,
                rebind[ImmA](a_buf),
                rebind[ImmB](b_buf),
                c_row_major=True,
                transpose_b=transpose_b,
            )
            elementwise[epilogue_wrapper, simd_size, target="gpu"](
                Index(m, n), ctx
            )
            return

        # Otherwise, we need to allocate a new buffer for c and apply the epilogue.
        var tmp_device_buffer = ctx.enqueue_create_buffer[c_type](
            c_buf.num_elements()
        )

        # Construct a new buffer with external origin pointing to the temporary storage.
        var c_tmp = NDBuffer[rank=2, c_type, MutExternalOrigin](
            rebind[UnsafePointer[Scalar[c_type], MutExternalOrigin]](
                tmp_device_buffer.unsafe_ptr()
            ),
            IndexList[2](Int(c.dim[0]()), Int(c.dim[1]())),
        )

        matmul[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](
            TileTensor(c_tmp),
            TileTensor(rebind[ImmA](a_buf)),
            TileTensor(rebind[ImmB](b_buf)),
            ctx,
        )

        _ = tmp_device_buffer^
