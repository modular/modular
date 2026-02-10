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

from math import exp
from sys.info import simd_width_of

from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeTuple,
    RuntimeLayout,
)
from layout._fillers import random
from memory import alloc
from state_space.varlen_causal_conv1d import (
    causal_conv1d_varlen_fwd_cpu,
    causal_conv1d_varlen_update_cpu,
    causal_conv1d_varlen_states_cpu,
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


fn run_varlen_causal_conv1d_fwd[
    dtype: DType,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seq_lengths: IndexList,
    width: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen causal conv1d forward kernel against reference implementation.
    """
    # Calculate total_seqlen (sum of all sequence lengths)
    var total_seqlen = 0
    for i in range(batch):
        total_seqlen += seq_lengths[i]

    # Allocate host memory
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
    var conv_states_h = LayoutTensor[
        dtype, Layout.row_major[3](), MutAnyOrigin
    ](
        conv_states_heap,
        RuntimeLayout[Layout.row_major[3]()].row_major(
            Index(batch, dim, state_len)
        ),
    ).fill(
        0
    )

    # output: (dim, total_seqlen)
    var output_heap = alloc[Scalar[dtype]](dim * total_seqlen)
    var output_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        output_heap,
        RuntimeLayout[layout_2d].row_major(Index(dim, total_seqlen)),
    ).fill(0)

    # reference output: (dim, total_seqlen)
    var output_ref_heap = alloc[Scalar[dtype]](dim * total_seqlen)
    var output_ref_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        output_ref_heap,
        RuntimeLayout[layout_2d].row_major(Index(dim, total_seqlen)),
    ).fill(0)

    # Initialize input data
    random(x_h)
    random(weight_h)
    random(bias_h)

    var x_buf = x_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var query_start_loc_buf = query_start_loc_h
    var cache_indices_buf = cache_indices_h
    var has_initial_state_buf = has_initial_state_h
    var conv_states_buf = conv_states_h
    var output_buf = output_h
    var output_ref_buf = output_ref_h

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

    # Test kernel
    causal_conv1d_varlen_fwd_cpu[
        x_buf.dtype,
        x_buf.layout,
        weight_buf.dtype,
        weight_buf.layout,
        bias_buf.dtype,
        bias_buf.layout,
        output_buf.dtype,
        output_buf.layout,
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
        output_buf,
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

    # Reference implementation
    var width_minus_1: Int = width - 1
    for b in range(batch):
        var seq_start = Int(query_start_loc_buf.ptr.load(b))
        var seq_end = Int(query_start_loc_buf.ptr.load(b + 1))
        var seqlen = seq_end - seq_start

        for d in range(dim):
            var bias_val = bias_buf.ptr.load(d)

            for l in range(seqlen):
                var conv_sum: Scalar[dtype] = bias_val

                for w_idx in range(width):
                    var input_l = l - (width_minus_1 - w_idx)
                    var input_val: Scalar[dtype] = Scalar[dtype](0.0)

                    if input_l >= 0:
                        var x_offset = (
                            UInt32(d) * x_dim_stride
                            + UInt32((seq_start + input_l)) * x_seqlen_stride
                        )
                        input_val = x_buf.ptr.load(x_offset)

                    var weight_offset = (
                        UInt32(d) * weight_dim_stride
                        + UInt32(w_idx) * weight_width_stride
                    )
                    var weight_val = weight_buf.ptr.load(weight_offset)
                    conv_sum = conv_sum + input_val * weight_val

                var out_val = conv_sum
                if silu_activation:
                    out_val = silu_ref[dtype](out_val)

                var out_offset = (
                    UInt32(d) * out_dim_stride
                    + UInt32((seq_start + l)) * out_seqlen_stride
                )
                output_ref_buf.ptr.store(out_offset, out_val)

    # Compare results
    var flattened_size = dim * total_seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_h.ptr[i],
            output_ref_h.ptr[i],
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
    output_heap.free()
    output_ref_heap.free()


fn run_varlen_causal_conv1d_update[
    dtype: DType,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen causal conv1d update kernel against reference implementation.
    """
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

    # cache_seqlens: (batch,) - can be empty
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
    var output_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var output_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    ).fill(0)

    # reference output: (batch, dim, seqlen)
    var output_ref_heap = alloc[Scalar[dtype]](batch * dim * seqlen)
    var output_ref_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_ref_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    ).fill(0)

    # Copy of conv_state for reference
    var conv_state_ref_heap = alloc[Scalar[dtype]](batch * dim * state_len)
    var conv_state_ref_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        conv_state_ref_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )

    # Initialize input data
    random(x_h)
    random(conv_state_h)
    random(weight_h)
    random(bias_h)

    # Copy conv_state for reference
    for i in range(batch * dim * state_len):
        conv_state_ref_h.ptr[i] = conv_state_h.ptr[i]

    var x_buf = x_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var conv_state_buf = conv_state_h
    var cache_seqlens_buf = cache_seqlens_h
    var conv_state_indices_buf = conv_state_indices_h
    var output_buf = output_h
    var output_ref_buf = output_ref_h
    var conv_state_ref_buf = conv_state_ref_h

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

    # Test kernel
    causal_conv1d_varlen_update_cpu[
        x_buf.dtype,
        x_buf.layout,
        weight_buf.dtype,
        weight_buf.layout,
        bias_buf.dtype,
        bias_buf.layout,
        output_buf.dtype,
        output_buf.layout,
        conv_state_buf.dtype,
        conv_state_buf.layout,
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
        conv_state_buf,
        cache_seqlens_buf,
        conv_state_indices_buf,
        output_buf,
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

    # Reference implementation
    var width_minus_1: Int = width - 1
    for b in range(batch):
        var state_batch_idx = b

        for d in range(dim):
            var bias_val = bias_buf.ptr.load(d)

            for l in range(seqlen):
                var conv_sum: Scalar[dtype] = bias_val

                for w_idx in range(width):
                    var rel_pos = w_idx - width_minus_1
                    var input_val: Scalar[dtype] = Scalar[dtype](0.0)

                    if rel_pos + l < 0:
                        # Read from state
                        var state_pos: Int
                        # has_cache_seqlens is True in our test, so use circular buffer
                        var cache_seqlen = Int(cache_seqlens_buf.ptr.load(b))
                        state_pos = (
                            cache_seqlen + rel_pos + l + state_len
                        ) % state_len

                        if state_pos >= 0 and state_pos < state_len:
                            var state_offset = (
                                UInt32(state_batch_idx)
                                * conv_state_batch_stride
                                + UInt32(d) * conv_state_dim_stride
                                + UInt32(state_pos) * conv_state_seqlen_stride
                            )
                            input_val = conv_state_ref_buf.ptr.load(
                                state_offset
                            )
                    else:
                        # Read from x
                        var x_l = rel_pos + l
                        if x_l >= 0 and x_l < seqlen:
                            var x_offset = (
                                UInt32(b) * x_batch_stride
                                + UInt32(d) * x_dim_stride
                                + UInt32(x_l) * x_seqlen_stride
                            )
                            input_val = x_buf.ptr.load(x_offset)

                    var weight_offset = (
                        UInt32(d) * weight_dim_stride
                        + UInt32(w_idx) * weight_width_stride
                    )
                    var weight_val = weight_buf.ptr.load(weight_offset)
                    conv_sum = conv_sum + input_val * weight_val

                var out_val = conv_sum
                if silu_activation:
                    out_val = silu_ref[dtype](out_val)

                var out_offset = (
                    UInt32(b) * out_batch_stride
                    + UInt32(d) * out_dim_stride
                    + UInt32(l) * out_seqlen_stride
                )
                output_ref_buf.ptr.store(out_offset, out_val)

            # Update state with new x values
            # This matches the CPU implementation logic exactly
            for l in range(seqlen):
                var x_offset = (
                    UInt32(b) * x_batch_stride
                    + UInt32(d) * x_dim_stride
                    + UInt32(l) * x_seqlen_stride
                )
                var x_val = x_buf.ptr.load(x_offset)

                var state_pos: Int
                # has_cache_seqlens is True in our test, so use circular buffer
                var cache_seqlen = Int(cache_seqlens_buf.ptr.load(b))
                state_pos = (cache_seqlen + l) % state_len

                var state_offset = (
                    UInt32(state_batch_idx) * conv_state_batch_stride
                    + UInt32(d) * conv_state_dim_stride
                    + UInt32(state_pos) * conv_state_seqlen_stride
                )
                conv_state_ref_buf.ptr.store(state_offset, x_val)

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_h.ptr[i],
            output_ref_h.ptr[i],
            rtol=rtol,
        )

    # Compare conv_state updates
    var conv_state_size = batch * dim * state_len
    for i in range(conv_state_size):
        assert_almost_equal(
            conv_state_h.ptr[i],
            conv_state_ref_h.ptr[i],
            rtol=rtol,
        )

    # Cleanup
    x_heap.free()
    weight_heap.free()
    bias_heap.free()
    conv_state_heap.free()
    conv_state_ref_heap.free()
    cache_seqlens_heap.free()
    conv_state_indices_heap.free()
    output_heap.free()
    output_ref_heap.free()


fn run_varlen_causal_conv1d_states[
    dtype: DType,
](
    batch: Int,
    dim: Int,
    seq_lengths: IndexList,
    state_len: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen causal conv1d states extraction kernel."""
    # Calculate total_tokens (sum of all sequence lengths)
    var total_tokens = 0
    for i in range(batch):
        total_tokens += seq_lengths[i]

    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    # x: (total_tokens, dim) - sequences concatenated
    var x_heap = alloc[Scalar[dtype]](total_tokens * dim)
    var x_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        x_heap, RuntimeLayout[layout_2d].row_major(Index(total_tokens, dim))
    )

    # cu_seqlens: (batch + 1,) - cumulative sequence lengths
    var cu_seqlens_heap = alloc[Scalar[DType.int32]](batch + 1)
    var cu_seqlens_h = LayoutTensor[DType.int32, layout_1d, MutAnyOrigin](
        cu_seqlens_heap, RuntimeLayout[layout_1d].row_major(Index(batch + 1))
    )
    var cumsum = 0
    cu_seqlens_h.ptr.store(0, Scalar[DType.int32](0))
    for i in range(batch):
        cumsum += seq_lengths[i]
        cu_seqlens_h.ptr.store(i + 1, Scalar[DType.int32](cumsum))

    # states: (batch, dim, state_len)
    var states_heap = alloc[Scalar[dtype]](batch * dim * state_len)
    var states_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        states_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    ).fill(0)

    # reference states: (batch, dim, state_len)
    var states_ref_heap = alloc[Scalar[dtype]](batch * dim * state_len)
    var states_ref_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        states_ref_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    ).fill(0)

    # Initialize input data
    random(x_h)

    var x_buf = x_h
    var cu_seqlens_buf = cu_seqlens_h
    var states_buf = states_h
    var states_ref_buf = states_ref_h

    # Strides for row-major layout
    var x_seqlen_stride: UInt32 = UInt32(dim)
    var x_dim_stride: UInt32 = 1
    var states_batch_stride: UInt32 = UInt32(dim * state_len)
    var states_dim_stride: UInt32 = UInt32(state_len)
    var states_seqlen_stride: UInt32 = 1

    # Test kernel
    causal_conv1d_varlen_states_cpu[
        x_buf.dtype,
        x_buf.layout,
        cu_seqlens_buf.dtype,
        cu_seqlens_buf.layout,
        states_buf.dtype,
        states_buf.layout,
    ](
        total_tokens,
        dim,
        batch,
        state_len,
        x_buf,
        cu_seqlens_buf,
        states_buf,
        x_seqlen_stride,
        x_dim_stride,
        states_batch_stride,
        states_dim_stride,
        states_seqlen_stride,
    )

    # Reference implementation
    for b in range(batch):
        var end_idx = Int(cu_seqlens_buf.ptr.load(b + 1))
        var start_idx_seq = Int(cu_seqlens_buf.ptr.load(b))
        var start_idx = max(start_idx_seq, end_idx - state_len)
        var num_elements = end_idx - start_idx

        for i in range(num_elements):
            var x_seq_idx = start_idx + i
            var states_seq_idx = state_len - num_elements + i

            for d in range(dim):
                var x_offset = (
                    UInt32(x_seq_idx) * x_seqlen_stride
                    + UInt32(d) * x_dim_stride
                )
                var states_offset = (
                    UInt32(b) * states_batch_stride
                    + UInt32(d) * states_dim_stride
                    + UInt32(states_seq_idx) * states_seqlen_stride
                )
                var val = x_buf.ptr.load(x_offset)
                states_ref_buf.ptr.store(states_offset, val)

    # Compare results
    var flattened_size = batch * dim * state_len
    for i in range(flattened_size):
        assert_almost_equal(
            states_h.ptr[i],
            states_ref_h.ptr[i],
            rtol=rtol,
        )

    # Cleanup
    x_heap.free()
    cu_seqlens_heap.free()
    states_heap.free()
    states_ref_heap.free()


# =============================================================================
# Test functions for varlen causal conv1d forward
# =============================================================================


fn test_varlen_causal_conv1d_fwd_equal_lengths() raises:
    """Test varlen causal conv1d forward with equal-length sequences."""
    run_varlen_causal_conv1d_fwd[DType.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=3
    )


fn test_varlen_causal_conv1d_fwd_variable_lengths() raises:
    """Test varlen causal conv1d forward with variable-length sequences."""
    run_varlen_causal_conv1d_fwd[DType.float32, "none"](
        batch=3, dim=4, seq_lengths=Index(10, 6, 1), width=3
    )


fn test_varlen_causal_conv1d_fwd_with_silu() raises:
    """Test varlen causal conv1d forward with SiLU activation."""
    run_varlen_causal_conv1d_fwd[DType.float32, "silu"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=3
    )


fn test_varlen_causal_conv1d_fwd_various_widths() raises:
    """Test varlen causal conv1d forward with various kernel widths."""
    run_varlen_causal_conv1d_fwd[DType.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=2
    )
    run_varlen_causal_conv1d_fwd[DType.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=4
    )


# =============================================================================
# Test functions for varlen causal conv1d update
# =============================================================================


fn test_varlen_causal_conv1d_update_basic() raises:
    """Test basic varlen causal conv1d update."""
    run_varlen_causal_conv1d_update[DType.float32, "none"](
        batch=2, dim=4, seqlen=1, width=3, state_len=4
    )


fn test_varlen_causal_conv1d_update_with_silu() raises:
    """Test varlen causal conv1d update with SiLU activation."""
    run_varlen_causal_conv1d_update[DType.float32, "silu"](
        batch=2, dim=4, seqlen=1, width=3, state_len=4
    )


fn test_varlen_causal_conv1d_update_seqlen_gt_1() raises:
    """Test varlen causal conv1d update with seqlen > 1."""
    run_varlen_causal_conv1d_update[DType.float32, "none"](
        batch=2, dim=4, seqlen=4, width=3, state_len=4
    )


# =============================================================================
# Test functions for varlen causal conv1d states
# =============================================================================


fn test_varlen_causal_conv1d_states_basic() raises:
    """Test basic varlen causal conv1d states extraction."""
    run_varlen_causal_conv1d_states[DType.float32](
        batch=2, dim=4, seq_lengths=Index(8, 8), state_len=3
    )


fn test_varlen_causal_conv1d_states_variable_lengths() raises:
    """Test varlen causal conv1d states with variable-length sequences."""
    run_varlen_causal_conv1d_states[DType.float32](
        batch=3, dim=4, seq_lengths=Index(10, 6, 1), state_len=3
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
