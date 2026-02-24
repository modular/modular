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
"""SM100 Conv2D dispatch for the nn conv op.

This module provides a dtype-safe dispatch function that gates SM100
conv2d kernel instantiation to supported dtypes (bf16/fp16) only.
Importing this module does NOT trigger kernel compilation -- the kernel
is only compiled when `dispatch_sm100_conv2d` is called with a supported
dtype inside a @parameter if guard.
"""

from math import ceildiv
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from memory import LegacyUnsafePointer

comptime MutPointer = LegacyUnsafePointer[mut=True, ...]
from utils.index import IndexList


# =========================================================================
# Filter transpose kernels (no SM100 deps, safe for any dtype)
# =========================================================================


fn _transpose_rscf_to_krsc[
    dtype: DType,
](
    src_ptr: MutPointer[Scalar[dtype]],
    dst_ptr: MutPointer[Scalar[dtype]],
    R: Int,
    S: Int,
    C: Int,
    F: Int,
):
    """GPU kernel: transpose filter RSCF [R,S,C,F] -> KRSC [K,R,S,C]."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = R * S * C * F
    if tid >= total:
        return
    var k = tid // (R * S * C)
    var rem = tid % (R * S * C)
    var r = rem // (S * C)
    rem = rem % (S * C)
    var s = rem // C
    var c = rem % C
    var src_idx = r * S * C * F + s * C * F + c * F + k
    dst_ptr.store(tid, src_ptr.load(src_idx))


fn _transpose_fcrs_to_krsc[
    dtype: DType,
](
    src_ptr: MutPointer[Scalar[dtype]],
    dst_ptr: MutPointer[Scalar[dtype]],
    F: Int,
    C: Int,
    R: Int,
    S: Int,
):
    """GPU kernel: transpose filter FCRS [F,C,R,S] -> KRSC [K,R,S,C]."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = F * C * R * S
    if tid >= total:
        return
    var k = tid // (R * S * C)
    var rem = tid % (R * S * C)
    var r = rem // (S * C)
    rem = rem % (S * C)
    var s = rem // C
    var c = rem % C
    var src_idx = k * C * R * S + c * R * S + r * S + s
    dst_ptr.store(tid, src_ptr.load(src_idx))


# =========================================================================
# SM100 Conv2D dispatch
# =========================================================================


fn dispatch_sm100_conv2d[
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcrs: Bool,
](
    input: LayoutTensor[input_type, input_layout, ...],
    filter: LayoutTensor[filter_type, filter_layout, ...],
    output: LayoutTensor[mut=True, output_type, output_layout, ...],
    symmetric_padding: IndexList[2],
    ctx: DeviceContext,
) raises:
    """Dispatch to SM100 structured conv2d with filter transpose.

    This function gates the SM100 kernel import behind @parameter if
    on dtype, so the kernel is never compiled for unsupported dtypes.
    """

    comptime if input_type == DType.bfloat16:
        from .conv2d import conv2d_fprop
        from .conv_config import Conv2dConfig, Conv2dProblemShape

        # Extract dimensions
        var batch = input.dim[0]()
        var in_h = input.dim[1]()
        var in_w = input.dim[2]()
        var in_c = input.dim[3]()
        var out_h = output.dim[1]()
        var out_w = output.dim[2]()
        var out_c = output.dim[3]()

        var fh: Int
        var fw: Int

        comptime if filter_is_fcrs:
            fh = filter.dim[2]()
            fw = filter.dim[3]()
        else:
            fh = filter.dim[0]()
            fw = filter.dim[1]()

        # Transpose filter to KRSC layout
        var filter_size = filter.size()
        var filter_buf = ctx.enqueue_create_buffer[filter_type](filter_size)
        var filter_krsc_ptr = filter_buf.unsafe_ptr()

        comptime transpose_block = 256
        var grid = ceildiv(filter_size, transpose_block)

        comptime if filter_is_fcrs:
            var F = filter.dim[0]()
            var C = filter.dim[1]()
            var R = filter.dim[2]()
            var S = filter.dim[3]()
            ctx.enqueue_function[
                _transpose_fcrs_to_krsc[filter_type],
                _transpose_fcrs_to_krsc[filter_type],
            ](
                filter.ptr,
                filter_krsc_ptr,
                F,
                C,
                R,
                S,
                grid_dim=grid,
                block_dim=transpose_block,
            )
        else:
            var R = filter.dim[0]()
            var S = filter.dim[1]()
            var C = filter.dim[2]()
            var F = filter.dim[3]()
            ctx.enqueue_function[
                _transpose_rscf_to_krsc[filter_type],
                _transpose_rscf_to_krsc[filter_type],
            ](
                filter.ptr,
                filter_krsc_ptr,
                R,
                S,
                C,
                F,
                grid_dim=grid,
                block_dim=transpose_block,
            )

        # Construct problem shape and NDBuffers
        var problem = Conv2dProblemShape(
            batch=batch,
            in_height=in_h,
            in_width=in_w,
            in_channels=in_c,
            out_channels=out_c,
            filter_h=fh,
            filter_w=fw,
            pad_h=symmetric_padding[0],
            pad_w=symmetric_padding[1],
        )

        comptime static_shape = DimList(-1, -1, -1, -1)
        var act_nd = NDBuffer[input_type, 4, _, static_shape](
            input.ptr, DimList(batch, in_h, in_w, in_c)
        )
        var filter_nd = NDBuffer[filter_type, 4, _, static_shape](
            filter_krsc_ptr, DimList(out_c, fh, fw, in_c)
        )
        var out_nd = NDBuffer[output_type, 4, _, static_shape](
            output.ptr, DimList(batch, out_h, out_w, out_c)
        )

        comptime config = Conv2dConfig[
            input_type, filter_type, output_type
        ].default_bf16_1sm()

        conv2d_fprop[config=config](out_nd, act_nd, filter_nd, problem, ctx)

        _ = filter_buf^
