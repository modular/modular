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

from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_true

from extensibility import IOSpec, ManagedTensorSlice, StaticTensorSpec
from layout import MixedLayout, TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.causal_conv1d import enqueue_causal_conv1d_update_gpu
from state_space.causal_conv1d_ops import CausalConv1DUpdate


def _run[
    dtype: DType, strided: Bool, alias_state: Bool, registered: Bool = False
](
    ctx: DeviceContext, seqlen: Int, state_len: Int, silu_activation: Bool
) raises:
    comptime batch = 2
    comptime channels = 131
    comptime width = 4
    var elements = max(1, batch * channels * seqlen)
    var state_elements = batch * channels * state_len
    var source_channel_stride = state_len * 2 + 3 if strided else state_len
    var source_batch_stride = (
        channels * source_channel_stride
        + 5 if strided else channels * state_len
    )
    var source_inner_stride = 2 if strided else 1
    var dest_channel_stride = state_len * 3 + 2 if strided else state_len
    var dest_batch_stride = (
        channels * dest_channel_stride + 7 if strided else channels * state_len
    )
    var dest_inner_stride = 3 if strided else 1
    comptime if alias_state:
        dest_channel_stride = source_channel_stride
        dest_batch_stride = source_batch_stride
        dest_inner_stride = source_inner_stride
    var source_size = batch * source_batch_stride
    var dest_size = batch * dest_batch_stride
    var input_h = ctx.enqueue_create_host_buffer[dtype](elements)
    var weights_h = ctx.enqueue_create_host_buffer[dtype](channels * width)
    var bias_h = ctx.enqueue_create_host_buffer[dtype](channels)
    var seed_h = ctx.enqueue_create_host_buffer[dtype](state_elements)
    var previous_h = ctx.enqueue_create_host_buffer[dtype](source_size)
    var dest_h = ctx.enqueue_create_host_buffer[dtype](dest_size)
    var baseline_state_h = ctx.enqueue_create_host_buffer[dtype](state_elements)
    var baseline_output_h = ctx.enqueue_create_host_buffer[dtype](elements)
    var candidate_output_h = ctx.enqueue_create_host_buffer[dtype](elements)
    for i in range(elements):
        input_h[i] = Scalar[dtype](Float32(i % 29 - 14) * 0.03125)
        baseline_output_h[i] = 0
        candidate_output_h[i] = 0
    for i in range(channels * width):
        weights_h[i] = Scalar[dtype](Float32(i % 11 - 5) * 0.0625)
    for i in range(channels):
        bias_h[i] = Scalar[dtype](Float32(i % 7 - 3) * 0.0625)
    for i in range(source_size):
        previous_h[i] = -7
    for i in range(dest_size):
        dest_h[i] = -7
    for b in range(batch):
        for c in range(channels):
            for s in range(state_len):
                var index = (b * channels + c) * state_len + s
                seed_h[index] = Scalar[dtype](
                    Float32(index % 31 - 15) * 0.03125
                )
                previous_h[
                    b * source_batch_stride
                    + c * source_channel_stride
                    + s * source_inner_stride
                ] = seed_h[index]
    var input_d = ctx.enqueue_create_buffer[dtype](elements)
    var weights_d = ctx.enqueue_create_buffer[dtype](channels * width)
    var bias_d = ctx.enqueue_create_buffer[dtype](channels)
    var seed_d = ctx.enqueue_create_buffer[dtype](state_elements)
    var baseline_state_d = ctx.enqueue_create_buffer[dtype](state_elements)
    var previous_d = ctx.enqueue_create_buffer[dtype](source_size)
    var dest_d = ctx.enqueue_create_buffer[dtype](dest_size)
    var baseline_output_d = ctx.enqueue_create_buffer[dtype](elements)
    var candidate_output_d = ctx.enqueue_create_buffer[dtype](elements)
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(weights_d, weights_h)
    ctx.enqueue_copy(bias_d, bias_h)
    ctx.enqueue_copy(seed_d, seed_h)
    ctx.enqueue_copy(baseline_state_d, seed_h)
    ctx.enqueue_copy(previous_d, previous_h)
    ctx.enqueue_copy(dest_d, dest_h)
    ctx.enqueue_copy(baseline_output_d, baseline_output_h)
    ctx.enqueue_copy(candidate_output_d, candidate_output_h)
    var input_tt = TileTensor(input_d, row_major(batch, channels, seqlen))
    var weights_tt = TileTensor(weights_d, row_major(channels, width))
    var bias_tt = TileTensor(bias_d, row_major(channels))
    var seed_tt = TileTensor(seed_d, row_major(batch, channels, state_len))
    var baseline_state_tt = TileTensor(
        baseline_state_d, row_major(batch, channels, state_len)
    )
    var baseline_output_tt = TileTensor(
        baseline_output_d, row_major(batch, channels, seqlen)
    )
    var candidate_output_tt = TileTensor(
        candidate_output_d, row_major(batch, channels, seqlen)
    )
    var previous_tt = TileTensor(
        previous_d,
        MixedLayout(
            (batch, channels, state_len),
            (source_batch_stride, source_channel_stride, source_inner_stride),
        ),
    )
    var dest_ptr = dest_d.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    comptime if alias_state:
        dest_ptr = previous_d.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
    var dest_tt = TileTensor(
        dest_ptr,
        MixedLayout(
            (batch, channels, state_len),
            (dest_batch_stride, dest_channel_stride, dest_inner_stride),
        ),
    )
    enqueue_causal_conv1d_update_gpu[separate_state=False](
        ctx,
        batch,
        channels,
        seqlen,
        width,
        state_len,
        input_tt,
        seed_tt,
        baseline_state_tt,
        weights_tt,
        baseline_output_tt,
        bias_tt,
        silu_activation,
    )
    comptime if registered:
        comptime assert not strided and not alias_state
        comptime spec3 = StaticTensorSpec[dtype, 3].get_unknown()
        comptime spec2 = StaticTensorSpec[dtype, 2].get_unknown()
        comptime spec1 = StaticTensorSpec[dtype, 1].get_unknown()
        var input_arg = ManagedTensorSlice[
            io_spec=IOSpec.Input, static_spec=spec3
        ](input_d.unsafe_ptr(), (batch, channels, seqlen))
        var previous_arg = ManagedTensorSlice[
            io_spec=IOSpec.Input, static_spec=spec3
        ](previous_d.unsafe_ptr(), (batch, channels, state_len))
        var weights_arg = ManagedTensorSlice[
            io_spec=IOSpec.Input, static_spec=spec2
        ](weights_d.unsafe_ptr(), (channels, width))
        var bias_arg = ManagedTensorSlice[
            io_spec=IOSpec.Input, static_spec=spec1
        ](bias_d.unsafe_ptr(), (channels,))
        var output_arg = ManagedTensorSlice[
            io_spec=IOSpec.Output, static_spec=spec3
        ](candidate_output_d.unsafe_ptr(), (batch, channels, seqlen))
        var state_arg = ManagedTensorSlice[
            io_spec=IOSpec.Output, static_spec=spec3
        ](dest_d.unsafe_ptr(), (batch, channels, state_len))
        if silu_activation:
            CausalConv1DUpdate["silu"].execute[target="gpu"](
                output_arg,
                state_arg,
                input_arg,
                previous_arg,
                weights_arg,
                bias_arg,
                ctx,
            )
        else:
            CausalConv1DUpdate["none"].execute[target="gpu"](
                output_arg,
                state_arg,
                input_arg,
                previous_arg,
                weights_arg,
                bias_arg,
                ctx,
            )
    else:
        enqueue_causal_conv1d_update_gpu[separate_state=True](
            ctx,
            batch,
            channels,
            seqlen,
            width,
            state_len,
            input_tt,
            previous_tt,
            dest_tt,
            weights_tt,
            candidate_output_tt,
            bias_tt,
            silu_activation,
        )
    ctx.enqueue_copy(baseline_state_h, baseline_state_d)
    ctx.enqueue_copy(baseline_output_h, baseline_output_d)
    ctx.enqueue_copy(candidate_output_h, candidate_output_d)
    var actual_state_h = ctx.enqueue_create_host_buffer[dtype](dest_size)
    var previous_after_h = ctx.enqueue_create_host_buffer[dtype](source_size)
    ctx.enqueue_copy(
        actual_state_h.unsafe_ptr(), dest_tt.unsafe_ptr(), dest_size
    )
    ctx.enqueue_copy(previous_after_h, previous_d)
    ctx.synchronize()
    comptime bits_dtype = DType.uint16 if dtype == DType.bfloat16 else DType.uint32
    for i in range(batch * channels * seqlen):
        assert_equal(
            bitcast[bits_dtype](candidate_output_h[i]),
            bitcast[bits_dtype](baseline_output_h[i]),
        )
    for b in range(batch):
        for c in range(channels):
            for s in range(state_len):
                var compact_index = (b * channels + c) * state_len + s
                var dest_index = (
                    b * dest_batch_stride
                    + c * dest_channel_stride
                    + s * dest_inner_stride
                )
                dest_h[dest_index] = baseline_state_h[compact_index]
    # Sentinel padding must remain untouched in both layouts.
    for i in range(dest_size):
        assert_equal(
            bitcast[bits_dtype](actual_state_h[i]),
            bitcast[bits_dtype](dest_h[i]),
        )
    comptime if not alias_state:
        for i in range(source_size):
            assert_equal(
                bitcast[bits_dtype](previous_after_h[i]),
                bitcast[bits_dtype](previous_h[i]),
            )
    _ = input_d^
    _ = weights_d^
    _ = bias_d^
    _ = seed_d^
    _ = baseline_state_d^
    _ = previous_d^
    _ = dest_d^
    _ = baseline_output_d^
    _ = candidate_output_d^


def test_functional_causal_conv1d_update() raises:
    with DeviceContext() as ctx:
        assert_true(ctx.is_compatible(), "A compatible GPU is required")
        comptime for dtype in [DType.bfloat16, DType.float32]:
            for state_len in [4, 7]:
                for seqlen in [0, 1, 3, 4, 7, 8]:
                    for silu_activation in [False, True]:
                        _run[dtype, False, False](
                            ctx, seqlen, state_len, silu_activation
                        )
                        _run[dtype, True, False](
                            ctx, seqlen, state_len, silu_activation
                        )
                        _run[dtype, False, True](
                            ctx, seqlen, state_len, silu_activation
                        )
                        _run[dtype, True, True](
                            ctx, seqlen, state_len, silu_activation
                        )


def test_registered_causal_conv1d_update() raises:
    with DeviceContext() as ctx:
        assert_true(ctx.is_compatible(), "A compatible GPU is required")
        comptime for dtype in [DType.bfloat16, DType.float32]:
            for state_len in [4, 7]:
                for seqlen in [0, 1, 8]:
                    for silu_activation in [False, True]:
                        _run[dtype, False, False, True](
                            ctx, seqlen, state_len, silu_activation
                        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
