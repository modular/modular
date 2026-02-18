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

from math import ceildiv, exp
from sys.info import simd_width_of

from gpu.host import DeviceContext
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeTuple,
    RuntimeLayout,
)
from random import rand
from memory import alloc
from state_space.varlen_causal_conv1d import (
    causal_conv1d_varlen_fwd_cpu,
    causal_conv1d_varlen_update_cpu,
    causal_conv1d_varlen_states_cpu,
    causal_conv1d_varlen_fwd_gpu,
    causal_conv1d_varlen_update_gpu,
    causal_conv1d_varlen_states_gpu,
)
from testing import TestSuite, assert_almost_equal

from utils.index import Index, IndexList


# Constants
comptime PAD_SLOT_ID: Int32 = -1


@always_inline
fn silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[DType.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = Scalar[DType.float32](1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


fn run_varlen_causal_conv1d_fwd_gpu[
    dtype: DType,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seq_lengths: IndexList,
    width: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen causal conv1d forward GPU kernel against CPU reference."""
    # Calculate total_seqlen (sum of all sequence lengths)
    var total_seqlen = 0
    for i in range(batch):
        total_seqlen += seq_lengths[i]

    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    # x: (dim, total_seqlen) for varlen - sequences concatenated
    var x_heap = alloc[Scalar[dtype]](dim * total_seqlen)
    var x_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        x_heap, RuntimeLayout[layout_2d].row_major(Index(dim, total_seqlen))
    )

    # weight: (dim, width)
    var weight_heap = alloc[Scalar[dtype]](dim * width)
    var weight_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        weight_heap, RuntimeLayout[layout_2d].row_major(Index(dim, width))
    )

    # bias: (dim,)
    var bias_heap = alloc[Scalar[dtype]](dim)
    var bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        bias_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )

    # query_start_loc: (batch + 1,) - cumulative sequence lengths
    var query_start_loc_heap = alloc[Scalar[DType.int32]](batch + 1)
    var query_start_loc_h = LayoutTensor[DType.int32, layout_1d, MutAnyOrigin](
        query_start_loc_heap,
        RuntimeLayout[layout_1d].row_major(Index(batch + 1)),
    )
    var cumsum = 0
    query_start_loc_h.ptr.store(0, Scalar[DType.int32](0))
    for i in range(batch):
        cumsum += seq_lengths[i]
        query_start_loc_h.ptr.store(i + 1, Scalar[DType.int32](cumsum))

    # cache_indices: (batch,) - identity mapping
    var cache_indices_heap = alloc[Scalar[DType.int32]](batch)
    var cache_indices_h = LayoutTensor[DType.int32, layout_1d, MutAnyOrigin](
        cache_indices_heap, RuntimeLayout[layout_1d].row_major(Index(batch))
    )
    for i in range(batch):
        cache_indices_h.ptr.store(i, Scalar[DType.int32](i))

    # has_initial_state: (batch,) - all False
    var has_initial_state_heap = alloc[Scalar[DType.bool]](batch)
    var has_initial_state_h = LayoutTensor[DType.bool, layout_1d, MutAnyOrigin](
        has_initial_state_heap, RuntimeLayout[layout_1d].row_major(Index(batch))
    )
    for i in range(batch):
        has_initial_state_h.ptr.store(i, Scalar[DType.bool](False))

    # conv_states: (batch, dim, width - 1)
    var state_len = width - 1
    var conv_states_heap = alloc[Scalar[dtype]](batch * dim * state_len)
    var conv_states_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        conv_states_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    ).fill(0)

    # output: (dim, total_seqlen)
    var output_gpu_heap = alloc[Scalar[dtype]](dim * total_seqlen)
    var output_gpu_h = LayoutTensor[dtype, layout_2d](
        output_gpu_heap,
        RuntimeLayout[layout_2d].row_major(Index(dim, total_seqlen)),
    ).fill(0)

    var output_cpu_heap = alloc[Scalar[dtype]](dim * total_seqlen)
    var output_cpu_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        output_cpu_heap,
        RuntimeLayout[layout_2d].row_major(Index(dim, total_seqlen)),
    ).fill(0)

    # Initialize input data
    rand[dtype](x_h.ptr, x_h.size())
    rand[dtype](weight_h.ptr, weight_h.size())
    rand[dtype](bias_h.ptr, bias_h.size())

    var x_buf = x_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var query_start_loc_buf = query_start_loc_h
    var cache_indices_buf = cache_indices_h
    var has_initial_state_buf = has_initial_state_h
    var conv_states_buf = conv_states_h
    var output_gpu_buf = output_gpu_h
    var output_cpu_buf = output_cpu_h

    # Strides for row-major layout
    var x_dim_stride: UInt32 = UInt32(total_seqlen)
    var x_seqlen_stride: UInt32 = 1
    var weight_dim_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var out_dim_stride: UInt32 = UInt32(total_seqlen)
    var out_seqlen_stride: UInt32 = 1
    var conv_states_batch_stride: UInt32 = UInt32(dim * state_len)
    var conv_states_dim_stride: UInt32 = UInt32(state_len)
    var conv_states_width_stride: UInt32 = 1

    var silu_activation = activation == "silu"
    var silu_activation_int8 = Int8(silu_activation)

    # Allocate device buffers
    var x_device = ctx.enqueue_create_buffer[dtype](dim * total_seqlen)
    var weight_device = ctx.enqueue_create_buffer[dtype](dim * width)
    var bias_device = ctx.enqueue_create_buffer[dtype](dim)
    var query_start_loc_device = ctx.enqueue_create_buffer[DType.int32](
        batch + 1
    )
    var cache_indices_device = ctx.enqueue_create_buffer[DType.int32](batch)
    var has_initial_state_device = ctx.enqueue_create_buffer[DType.bool](batch)
    var conv_states_device = ctx.enqueue_create_buffer[dtype](
        batch * dim * state_len
    )
    var output_device = ctx.enqueue_create_buffer[dtype](dim * total_seqlen)

    # Copy data to device
    with ctx.push_context():
        ctx.enqueue_copy(x_device, x_buf.ptr)
        ctx.enqueue_copy(weight_device, weight_buf.ptr)
        ctx.enqueue_copy(bias_device, bias_buf.ptr)
        ctx.enqueue_copy(query_start_loc_device, query_start_loc_buf.ptr)
        ctx.enqueue_copy(cache_indices_device, cache_indices_buf.ptr)
        ctx.enqueue_copy(has_initial_state_device, has_initial_state_buf.ptr)
        ctx.enqueue_copy(conv_states_device, conv_states_buf.ptr)

    # Create device LayoutTensors
    var x_device_tensor = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        x_device.unsafe_ptr(),
        RuntimeLayout[layout_2d].row_major(Index(dim, total_seqlen)),
    )
    var weight_device_tensor = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        weight_device.unsafe_ptr(),
        RuntimeLayout[layout_2d].row_major(Index(dim, width)),
    )
    var bias_device_tensor = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        bias_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(Index(dim)),
    )
    var query_start_loc_device_tensor = LayoutTensor[
        DType.int32, layout_1d, MutAnyOrigin
    ](
        query_start_loc_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(Index(batch + 1)),
    )
    var cache_indices_device_tensor = LayoutTensor[
        DType.int32, layout_1d, MutAnyOrigin
    ](
        cache_indices_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(Index(batch)),
    )
    var has_initial_state_device_tensor = LayoutTensor[
        DType.bool, layout_1d, MutAnyOrigin
    ](
        has_initial_state_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(Index(batch)),
    )
    var conv_states_device_tensor = LayoutTensor[
        dtype, layout_3d, MutAnyOrigin
    ](
        conv_states_device.unsafe_ptr(),
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )
    var output_device_tensor = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        output_device.unsafe_ptr(),
        RuntimeLayout[layout_2d].row_major(Index(dim, total_seqlen)),
    )

    # Run GPU kernel
    comptime BLOCK_DIM = 128
    comptime BLOCK_SEQ = 1

    if width == 1:
        comptime kWidth = 1
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_fwd_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                query_start_loc_device_tensor.dtype,
                query_start_loc_device_tensor.layout,
                cache_indices_device_tensor.dtype,
                cache_indices_device_tensor.layout,
                has_initial_state_device_tensor.dtype,
                has_initial_state_device_tensor.layout,
                conv_states_device_tensor.dtype,
                conv_states_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
            ],
            causal_conv1d_varlen_fwd_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                query_start_loc_device_tensor.dtype,
                query_start_loc_device_tensor.layout,
                cache_indices_device_tensor.dtype,
                cache_indices_device_tensor.layout,
                has_initial_state_device_tensor.dtype,
                has_initial_state_device_tensor.layout,
                conv_states_device_tensor.dtype,
                conv_states_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
            ],
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                dim,
                total_seqlen,
                batch,
                x_device_tensor,
                weight_device_tensor,
                bias_device_tensor,
                query_start_loc_device_tensor,
                cache_indices_device_tensor,
                has_initial_state_device_tensor,
                conv_states_device_tensor,
                output_device_tensor,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                out_dim_stride,
                out_seqlen_stride,
                conv_states_batch_stride,
                conv_states_dim_stride,
                conv_states_width_stride,
                silu_activation_int8,
                PAD_SLOT_ID,
                Int8(1),  # has_cache_indices
                Int8(1),  # has_initial_state_flag
                Int8(1),  # has_conv_states
                Int8(1),  # has_bias
                grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_SEQ),
            )
    elif width == 2:
        comptime kWidth = 2
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_fwd_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                query_start_loc_device_tensor.dtype,
                query_start_loc_device_tensor.layout,
                cache_indices_device_tensor.dtype,
                cache_indices_device_tensor.layout,
                has_initial_state_device_tensor.dtype,
                has_initial_state_device_tensor.layout,
                conv_states_device_tensor.dtype,
                conv_states_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
            ],
            causal_conv1d_varlen_fwd_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                query_start_loc_device_tensor.dtype,
                query_start_loc_device_tensor.layout,
                cache_indices_device_tensor.dtype,
                cache_indices_device_tensor.layout,
                has_initial_state_device_tensor.dtype,
                has_initial_state_device_tensor.layout,
                conv_states_device_tensor.dtype,
                conv_states_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
            ],
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                dim,
                total_seqlen,
                batch,
                x_device_tensor,
                weight_device_tensor,
                bias_device_tensor,
                query_start_loc_device_tensor,
                cache_indices_device_tensor,
                has_initial_state_device_tensor,
                conv_states_device_tensor,
                output_device_tensor,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                out_dim_stride,
                out_seqlen_stride,
                conv_states_batch_stride,
                conv_states_dim_stride,
                conv_states_width_stride,
                silu_activation_int8,
                PAD_SLOT_ID,
                Int8(1),  # has_cache_indices
                Int8(1),  # has_initial_state_flag
                Int8(1),  # has_conv_states
                Int8(1),  # has_bias
                grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_SEQ),
            )
    elif width == 3:
        comptime kWidth = 3
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_fwd_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                query_start_loc_device_tensor.dtype,
                query_start_loc_device_tensor.layout,
                cache_indices_device_tensor.dtype,
                cache_indices_device_tensor.layout,
                has_initial_state_device_tensor.dtype,
                has_initial_state_device_tensor.layout,
                conv_states_device_tensor.dtype,
                conv_states_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
            ],
            causal_conv1d_varlen_fwd_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                query_start_loc_device_tensor.dtype,
                query_start_loc_device_tensor.layout,
                cache_indices_device_tensor.dtype,
                cache_indices_device_tensor.layout,
                has_initial_state_device_tensor.dtype,
                has_initial_state_device_tensor.layout,
                conv_states_device_tensor.dtype,
                conv_states_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
            ],
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                dim,
                total_seqlen,
                batch,
                x_device_tensor,
                weight_device_tensor,
                bias_device_tensor,
                query_start_loc_device_tensor,
                cache_indices_device_tensor,
                has_initial_state_device_tensor,
                conv_states_device_tensor,
                output_device_tensor,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                out_dim_stride,
                out_seqlen_stride,
                conv_states_batch_stride,
                conv_states_dim_stride,
                conv_states_width_stride,
                silu_activation_int8,
                PAD_SLOT_ID,
                Int8(1),  # has_cache_indices
                Int8(1),  # has_initial_state_flag
                Int8(1),  # has_conv_states
                Int8(1),  # has_bias
                grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_SEQ),
            )
    elif width == 4:
        comptime kWidth = 4
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_fwd_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                query_start_loc_device_tensor.dtype,
                query_start_loc_device_tensor.layout,
                cache_indices_device_tensor.dtype,
                cache_indices_device_tensor.layout,
                has_initial_state_device_tensor.dtype,
                has_initial_state_device_tensor.layout,
                conv_states_device_tensor.dtype,
                conv_states_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
            ],
            causal_conv1d_varlen_fwd_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                query_start_loc_device_tensor.dtype,
                query_start_loc_device_tensor.layout,
                cache_indices_device_tensor.dtype,
                cache_indices_device_tensor.layout,
                has_initial_state_device_tensor.dtype,
                has_initial_state_device_tensor.layout,
                conv_states_device_tensor.dtype,
                conv_states_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
            ],
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                dim,
                total_seqlen,
                batch,
                x_device_tensor,
                weight_device_tensor,
                bias_device_tensor,
                query_start_loc_device_tensor,
                cache_indices_device_tensor,
                has_initial_state_device_tensor,
                conv_states_device_tensor,
                output_device_tensor,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                out_dim_stride,
                out_seqlen_stride,
                conv_states_batch_stride,
                conv_states_dim_stride,
                conv_states_width_stride,
                silu_activation_int8,
                PAD_SLOT_ID,
                Int8(1),  # has_cache_indices
                Int8(1),  # has_initial_state_flag
                Int8(1),  # has_conv_states
                Int8(1),  # has_bias
                grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_SEQ),
            )
    else:
        raise Error(
            "Unsupported kernel width: only widths 1, 2, 3, 4 are supported"
        )

    # Copy GPU results back to host
    with ctx.push_context():
        ctx.enqueue_copy(output_gpu_buf.ptr, output_device)
    ctx.synchronize()

    # Run CPU reference
    causal_conv1d_varlen_fwd_cpu[
        x_buf.dtype,
        x_buf.layout,
        weight_buf.dtype,
        weight_buf.layout,
        bias_buf.dtype,
        bias_buf.layout,
        output_cpu_buf.dtype,
        output_cpu_buf.layout,
        query_start_loc_buf.dtype,
        query_start_loc_buf.layout,
        cache_indices_buf.dtype,
        cache_indices_buf.layout,
        has_initial_state_buf.dtype,
        has_initial_state_buf.layout,
        conv_states_buf.dtype,
        conv_states_buf.layout,
    ](
        dim,
        total_seqlen,
        width,
        batch,
        x_buf,
        weight_buf,
        bias_buf,
        query_start_loc_buf,
        cache_indices_buf,
        has_initial_state_buf,
        conv_states_buf,
        output_cpu_buf,
        x_dim_stride,
        x_seqlen_stride,
        weight_dim_stride,
        weight_width_stride,
        out_dim_stride,
        out_seqlen_stride,
        conv_states_batch_stride,
        conv_states_dim_stride,
        conv_states_width_stride,
        silu_activation,
        PAD_SLOT_ID,
        True,  # has_cache_indices
        True,  # has_initial_state_flag
        True,  # has_conv_states
        True,  # has_bias
    )

    # Compare results
    var flattened_size = dim * total_seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_gpu_h.ptr[i],
            output_cpu_h.ptr[i],
            rtol=rtol,
        )

    # Cleanup
    x_heap.free()
    weight_heap.free()
    bias_heap.free()
    query_start_loc_heap.free()
    cache_indices_heap.free()
    has_initial_state_heap.free()
    conv_states_heap.free()
    output_gpu_heap.free()
    output_cpu_heap.free()


fn run_varlen_causal_conv1d_update_gpu[
    dtype: DType,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen causal conv1d update GPU kernel against CPU reference."""
    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    # x: (batch, dim, seqlen)
    var x_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var x_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        x_heap, RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen))
    )

    # weight: (dim, width)
    var weight_heap = alloc[Scalar[dtype]](dim * width)
    var weight_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        weight_heap, RuntimeLayout[layout_2d].row_major(Index(dim, width))
    )

    # bias: (dim,)
    var bias_heap = alloc[Scalar[dtype]](dim)
    var bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        bias_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )

    # conv_state: (batch, dim, state_len)
    var conv_state_heap = alloc[Scalar[dtype]](batch * dim * state_len)
    var conv_state_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        conv_state_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )

    # cache_seqlens: (batch,) - all zeros
    var cache_seqlens_heap = alloc[Scalar[DType.int32]](batch)
    var cache_seqlens_h = LayoutTensor[DType.int32, layout_1d, MutAnyOrigin](
        cache_seqlens_heap, RuntimeLayout[layout_1d].row_major(Index(batch))
    )
    for i in range(batch):
        cache_seqlens_h.ptr.store(i, Scalar[DType.int32](0))

    # conv_state_indices: (batch,) - identity mapping
    var conv_state_indices_heap = alloc[Scalar[DType.int32]](batch)
    var conv_state_indices_h = LayoutTensor[
        DType.int32, layout_1d, MutAnyOrigin
    ](conv_state_indices_heap, RuntimeLayout[layout_1d].row_major(Index(batch)))
    for i in range(batch):
        conv_state_indices_h.ptr.store(i, Scalar[DType.int32](i))

    # output: (batch, dim, seqlen)
    var output_gpu_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var output_gpu_h = LayoutTensor[dtype, layout_3d](
        output_gpu_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    ).fill(0)

    var output_cpu_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var output_cpu_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_cpu_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    ).fill(0)

    # Copy of conv_state for CPU and GPU
    var conv_state_cpu_heap = alloc[Scalar[dtype]](batch * dim * state_len)
    var conv_state_cpu_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        conv_state_cpu_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )

    var conv_state_gpu_heap = alloc[Scalar[dtype]](batch * dim * state_len)
    var conv_state_gpu_h = LayoutTensor[dtype, layout_3d](
        conv_state_gpu_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )

    # Initialize input data
    rand[dtype](x_h.ptr, x_h.size())
    rand[dtype](conv_state_h.ptr, conv_state_h.size())
    rand[dtype](weight_h.ptr, weight_h.size())
    rand[dtype](bias_h.ptr, bias_h.size())

    # Copy conv_state for CPU and GPU
    for i in range(batch * dim * state_len):
        conv_state_cpu_h.ptr[i] = conv_state_h.ptr[i]
        conv_state_gpu_h.ptr[i] = conv_state_h.ptr[i]

    var x_buf = x_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var conv_state_cpu_buf = conv_state_cpu_h
    var conv_state_gpu_buf = conv_state_gpu_h
    var cache_seqlens_buf = cache_seqlens_h
    var conv_state_indices_buf = conv_state_indices_h
    var output_gpu_buf = output_gpu_h
    var output_cpu_buf = output_cpu_h

    # Strides for row-major layout
    var x_batch_stride: UInt32 = UInt32(dim * seqlen)
    var x_dim_stride: UInt32 = UInt32(seqlen)
    var x_seqlen_stride: UInt32 = 1
    var weight_dim_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var conv_state_batch_stride: UInt32 = UInt32(dim * state_len)
    var conv_state_dim_stride: UInt32 = UInt32(state_len)
    var conv_state_seqlen_stride: UInt32 = 1
    var out_batch_stride: UInt32 = UInt32(dim * seqlen)
    var out_dim_stride: UInt32 = UInt32(seqlen)
    var out_seqlen_stride: UInt32 = 1

    var silu_activation = activation == "silu"
    var silu_activation_int8 = Int8(silu_activation)

    # Allocate device buffers
    var x_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var weight_device = ctx.enqueue_create_buffer[dtype](dim * width)
    var bias_device = ctx.enqueue_create_buffer[dtype](dim)
    var conv_state_device = ctx.enqueue_create_buffer[dtype](
        batch * dim * state_len
    )
    var cache_seqlens_device = ctx.enqueue_create_buffer[DType.int32](batch)
    var conv_state_indices_device = ctx.enqueue_create_buffer[DType.int32](
        batch
    )
    var output_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)

    # Copy data to device
    with ctx.push_context():
        ctx.enqueue_copy(x_device, x_buf.ptr)
        ctx.enqueue_copy(weight_device, weight_buf.ptr)
        ctx.enqueue_copy(bias_device, bias_buf.ptr)
        ctx.enqueue_copy(conv_state_device, conv_state_gpu_buf.ptr)
        ctx.enqueue_copy(cache_seqlens_device, cache_seqlens_buf.ptr)
        ctx.enqueue_copy(conv_state_indices_device, conv_state_indices_buf.ptr)

    # Create device LayoutTensors
    var x_device_tensor = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        x_device.unsafe_ptr(),
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var weight_device_tensor = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        weight_device.unsafe_ptr(),
        RuntimeLayout[layout_2d].row_major(Index(dim, width)),
    )
    var bias_device_tensor = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        bias_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(Index(dim)),
    )
    var conv_state_device_tensor = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        conv_state_device.unsafe_ptr(),
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )
    var cache_seqlens_device_tensor = LayoutTensor[
        DType.int32, layout_1d, MutAnyOrigin
    ](
        cache_seqlens_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(Index(batch)),
    )
    var conv_state_indices_device_tensor = LayoutTensor[
        DType.int32, layout_1d, MutAnyOrigin
    ](
        conv_state_indices_device.unsafe_ptr(),
        RuntimeLayout[layout_1d].row_major(Index(batch)),
    )
    var output_device_tensor = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_device.unsafe_ptr(),
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    # Run GPU kernel
    comptime BLOCK_DIM = 128

    if width == 1:
        comptime kWidth = 1
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_update_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                conv_state_device_tensor.dtype,
                conv_state_device_tensor.layout,
                cache_seqlens_device_tensor.dtype,
                cache_seqlens_device_tensor.layout,
                conv_state_indices_device_tensor.dtype,
                conv_state_indices_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
            ],
            causal_conv1d_varlen_update_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                conv_state_device_tensor.dtype,
                conv_state_device_tensor.layout,
                cache_seqlens_device_tensor.dtype,
                cache_seqlens_device_tensor.layout,
                conv_state_indices_device_tensor.dtype,
                conv_state_indices_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
            ],
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                batch,
                dim,
                seqlen,
                state_len,
                x_device_tensor,
                weight_device_tensor,
                bias_device_tensor,
                conv_state_device_tensor,
                cache_seqlens_device_tensor,
                conv_state_indices_device_tensor,
                output_device_tensor,
                x_batch_stride,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                conv_state_batch_stride,
                conv_state_dim_stride,
                conv_state_seqlen_stride,
                out_batch_stride,
                out_dim_stride,
                out_seqlen_stride,
                silu_activation_int8,
                PAD_SLOT_ID,
                Int8(1),  # has_conv_state_indices
                Int8(1),  # has_cache_seqlens
                Int8(1),  # has_bias
                grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                block_dim=(BLOCK_DIM),
            )
    elif width == 2:
        comptime kWidth = 2
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_update_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                conv_state_device_tensor.dtype,
                conv_state_device_tensor.layout,
                cache_seqlens_device_tensor.dtype,
                cache_seqlens_device_tensor.layout,
                conv_state_indices_device_tensor.dtype,
                conv_state_indices_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
            ],
            causal_conv1d_varlen_update_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                conv_state_device_tensor.dtype,
                conv_state_device_tensor.layout,
                cache_seqlens_device_tensor.dtype,
                cache_seqlens_device_tensor.layout,
                conv_state_indices_device_tensor.dtype,
                conv_state_indices_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
            ],
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                batch,
                dim,
                seqlen,
                state_len,
                x_device_tensor,
                weight_device_tensor,
                bias_device_tensor,
                conv_state_device_tensor,
                cache_seqlens_device_tensor,
                conv_state_indices_device_tensor,
                output_device_tensor,
                x_batch_stride,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                conv_state_batch_stride,
                conv_state_dim_stride,
                conv_state_seqlen_stride,
                out_batch_stride,
                out_dim_stride,
                out_seqlen_stride,
                silu_activation_int8,
                PAD_SLOT_ID,
                Int8(1),  # has_conv_state_indices
                Int8(1),  # has_cache_seqlens
                Int8(1),  # has_bias
                grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                block_dim=(BLOCK_DIM),
            )
    elif width == 3:
        comptime kWidth = 3
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_update_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                conv_state_device_tensor.dtype,
                conv_state_device_tensor.layout,
                cache_seqlens_device_tensor.dtype,
                cache_seqlens_device_tensor.layout,
                conv_state_indices_device_tensor.dtype,
                conv_state_indices_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
            ],
            causal_conv1d_varlen_update_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                conv_state_device_tensor.dtype,
                conv_state_device_tensor.layout,
                cache_seqlens_device_tensor.dtype,
                cache_seqlens_device_tensor.layout,
                conv_state_indices_device_tensor.dtype,
                conv_state_indices_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
            ],
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                batch,
                dim,
                seqlen,
                state_len,
                x_device_tensor,
                weight_device_tensor,
                bias_device_tensor,
                conv_state_device_tensor,
                cache_seqlens_device_tensor,
                conv_state_indices_device_tensor,
                output_device_tensor,
                x_batch_stride,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                conv_state_batch_stride,
                conv_state_dim_stride,
                conv_state_seqlen_stride,
                out_batch_stride,
                out_dim_stride,
                out_seqlen_stride,
                silu_activation_int8,
                PAD_SLOT_ID,
                Int8(1),  # has_conv_state_indices
                Int8(1),  # has_cache_seqlens
                Int8(1),  # has_bias
                grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                block_dim=(BLOCK_DIM),
            )
    elif width == 4:
        comptime kWidth = 4
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_update_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                conv_state_device_tensor.dtype,
                conv_state_device_tensor.layout,
                cache_seqlens_device_tensor.dtype,
                cache_seqlens_device_tensor.layout,
                conv_state_indices_device_tensor.dtype,
                conv_state_indices_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
            ],
            causal_conv1d_varlen_update_gpu[
                x_device_tensor.dtype,
                x_device_tensor.layout,
                weight_device_tensor.dtype,
                weight_device_tensor.layout,
                bias_device_tensor.dtype,
                bias_device_tensor.layout,
                output_device_tensor.dtype,
                output_device_tensor.layout,
                conv_state_device_tensor.dtype,
                conv_state_device_tensor.layout,
                cache_seqlens_device_tensor.dtype,
                cache_seqlens_device_tensor.layout,
                conv_state_indices_device_tensor.dtype,
                conv_state_indices_device_tensor.layout,
                kWidth,
                BLOCK_DIM,
            ],
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                batch,
                dim,
                seqlen,
                state_len,
                x_device_tensor,
                weight_device_tensor,
                bias_device_tensor,
                conv_state_device_tensor,
                cache_seqlens_device_tensor,
                conv_state_indices_device_tensor,
                output_device_tensor,
                x_batch_stride,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                conv_state_batch_stride,
                conv_state_dim_stride,
                conv_state_seqlen_stride,
                out_batch_stride,
                out_dim_stride,
                out_seqlen_stride,
                silu_activation_int8,
                PAD_SLOT_ID,
                Int8(1),  # has_conv_state_indices
                Int8(1),  # has_cache_seqlens
                Int8(1),  # has_bias
                grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                block_dim=(BLOCK_DIM),
            )
    else:
        raise Error(
            "Unsupported kernel width: only widths 1, 2, 3, 4 are supported"
        )

    # Copy results back from device
    with ctx.push_context():
        ctx.enqueue_copy(output_gpu_buf.ptr, output_device)
        ctx.enqueue_copy(conv_state_gpu_buf.ptr, conv_state_device)
    ctx.synchronize()

    # Run CPU reference
    causal_conv1d_varlen_update_cpu[
        x_buf.dtype,
        x_buf.layout,
        weight_buf.dtype,
        weight_buf.layout,
        bias_buf.dtype,
        bias_buf.layout,
        output_cpu_buf.dtype,
        output_cpu_buf.layout,
        conv_state_cpu_buf.dtype,
        conv_state_cpu_buf.layout,
        cache_seqlens_buf.dtype,
        cache_seqlens_buf.layout,
        conv_state_indices_buf.dtype,
        conv_state_indices_buf.layout,
    ](
        batch,
        dim,
        seqlen,
        width,
        state_len,
        x_buf,
        weight_buf,
        bias_buf,
        conv_state_cpu_buf,
        cache_seqlens_buf,
        conv_state_indices_buf,
        output_cpu_buf,
        x_batch_stride,
        x_dim_stride,
        x_seqlen_stride,
        weight_dim_stride,
        weight_width_stride,
        conv_state_batch_stride,
        conv_state_dim_stride,
        conv_state_seqlen_stride,
        out_batch_stride,
        out_dim_stride,
        out_seqlen_stride,
        silu_activation,
        PAD_SLOT_ID,
        True,  # has_conv_state_indices
        True,  # has_cache_seqlens
        True,  # has_bias
    )

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_gpu_h.ptr[i],
            output_cpu_h.ptr[i],
            rtol=rtol,
        )

    # Compare conv_state updates
    var conv_state_size = batch * dim * state_len
    for i in range(conv_state_size):
        assert_almost_equal(
            conv_state_gpu_h.ptr[i],
            conv_state_cpu_h.ptr[i],
            rtol=rtol,
        )

    # Cleanup
    x_heap.free()
    weight_heap.free()
    bias_heap.free()
    conv_state_heap.free()
    conv_state_cpu_heap.free()
    conv_state_gpu_heap.free()
    cache_seqlens_heap.free()
    conv_state_indices_heap.free()
    output_gpu_heap.free()
    output_cpu_heap.free()


# =============================================================================
# Test functions for varlen causal conv1d forward on GPU
# =============================================================================


fn test_varlen_causal_conv1d_fwd_gpu_equal_lengths() raises:
    """Test varlen causal conv1d forward GPU with equal-length sequences."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[DType.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=3, ctx=ctx
    )


fn test_varlen_causal_conv1d_fwd_gpu_variable_lengths() raises:
    """Test varlen causal conv1d forward GPU with variable-length sequences."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[DType.float32, "none"](
        batch=3, dim=4, seq_lengths=Index(10, 6, 1), width=3, ctx=ctx
    )


fn test_varlen_causal_conv1d_fwd_gpu_with_silu() raises:
    """Test varlen causal conv1d forward GPU with SiLU activation."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[DType.float32, "silu"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=3, ctx=ctx
    )


fn test_varlen_causal_conv1d_fwd_gpu_various_widths() raises:
    """Test varlen causal conv1d forward GPU with various kernel widths."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[DType.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=2, ctx=ctx
    )
    run_varlen_causal_conv1d_fwd_gpu[DType.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=4, ctx=ctx
    )


# =============================================================================
# Test functions for varlen causal conv1d update on GPU
# =============================================================================


fn test_varlen_causal_conv1d_update_gpu_basic() raises:
    """Test basic varlen causal conv1d update on GPU."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_update_gpu[DType.float32, "none"](
        batch=2, dim=4, seqlen=1, width=3, state_len=4, ctx=ctx
    )


fn test_varlen_causal_conv1d_update_gpu_with_silu() raises:
    """Test varlen causal conv1d update GPU with SiLU activation."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_update_gpu[DType.float32, "silu"](
        batch=2, dim=4, seqlen=1, width=3, state_len=4, ctx=ctx
    )


fn test_varlen_causal_conv1d_update_gpu_seqlen_gt_1() raises:
    """Test varlen causal conv1d update GPU with seqlen > 1."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_update_gpu[DType.float32, "none"](
        batch=2, dim=4, seqlen=4, width=3, state_len=4, ctx=ctx
    )


fn test_varlen_causal_conv1d_update_gpu_various_widths() raises:
    """Test varlen causal conv1d update GPU with various kernel widths."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_update_gpu[DType.float32, "none"](
        batch=2, dim=4, seqlen=1, width=2, state_len=3, ctx=ctx
    )
    run_varlen_causal_conv1d_update_gpu[DType.float32, "none"](
        batch=2, dim=4, seqlen=1, width=4, state_len=5, ctx=ctx
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
