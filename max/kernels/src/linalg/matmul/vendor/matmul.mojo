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

from std.sys import simd_width_of, size_of, has_nvidia_gpu_accelerator

from std.algorithm import elementwise
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.host.info import B200
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout.tile_tensor import NullableTileTensor

from std.utils import Index, IndexList

from ...utils import elementwise_epilogue_type
from .blas import matmul as vendor_matmul


def matmul[
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NullableTileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    ctx: DeviceContext,
) raises:
    """Vendor matmul dispatch for TileTensor operands."""
    comptime assert c.flat_rank == 2, "c must be of rank 2"
    comptime assert a.flat_rank == 2, "a must be of rank 2"
    comptime assert b.flat_rank == 2, "b must be of rank 2"

    comptime c_type = c.dtype

    comptime if not elementwise_lambda_fn:
        if not c.ptr:
            raise "c must be allocated"
        vendor_matmul[use_tf32=True](
            ctx,
            c,
            a,
            b,
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

        var c_tt = TileTensor(
            rebind[UnsafePointer[Scalar[c_type], MutAnyOrigin]](c.ptr),
            row_major(Coord(Idx(Int(c.dim[0]())), Idx(Int(c.dim[1]())))),
        )

        @parameter
        @__copy_capture(c_tt)
        def epilogue_wrapper[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var c_coord = Index(idx[0], idx[1])
            var c_val = c_tt.load_linear[
                width=simd_width,
                # Load takes alignment in bytes, lambda takes number of elements
                alignment=alignment * size_of[c_type](),
            ](idx)
            epilogue[c_type, simd_width, alignment=alignment](c_coord, c_val)

        # If c is already allocated, we can just use the vendor matmul and
        # apply the epilogue.
        if c.ptr:
            var m = Int(c.dim[0]())
            var n = Int(c.dim[1]())

            # For D = alpha * A * B + beta * C, vendor matmul currently sets
            # C to null, i.e don't fuse linear operations into gemm, KERN-1774.
            vendor_matmul[use_tf32=True](
                ctx,
                c,
                a,
                b,
                c_row_major=True,
                transpose_b=transpose_b,
            )
            elementwise[epilogue_wrapper, simd_size, target="gpu"](
                Index(m, n), ctx
            )
            return

        # Otherwise, we need to allocate a new buffer for c and apply the
        # epilogue.
        var num_elements = Int(c.dim[0]()) * Int(c.dim[1]())
        var tmp_device_buffer = ctx.enqueue_create_buffer[c_type](num_elements)

        var c_tmp = TileTensor(
            tmp_device_buffer,
            row_major(Coord(Idx(Int(c.dim[0]())), Idx(Int(c.dim[1]())))),
        )

        matmul[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c_tmp, a, b, ctx)

        _ = tmp_device_buffer^
