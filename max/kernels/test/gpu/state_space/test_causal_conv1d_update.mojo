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

from std.math import ceildiv, exp

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.random import rand
from state_space.causal_conv1d import (
    causal_conv1d_update_cpu,
    causal_conv1d_update_cpu_no_bias,
    causal_conv1d_update_gpu,
    causal_conv1d_update_gpu_no_bias,
)
from std.testing import TestSuite, assert_almost_equal, assert_true


def main() raises:
    var suite = TestSuite()
    suite.test[test_gpu_causal_conv1d_update_basic]()
    suite.test[test_gpu_causal_conv1d_update_with_silu]()
    suite.test[test_gpu_causal_conv1d_update_without_bias]()
    suite.test[test_gpu_causal_conv1d_update_seqlen_greater_than_one]()
    suite.test[test_gpu_causal_conv1d_update_various_widths]()
    suite^.run()


@inline(.always)
def silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = Float32(1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


def run_causal_conv1d_update_gpu[
    dtype: DType,
    has_bias: Bool,
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
    """Test causal conv1d update GPU kernel against CPU reference."""
    # Allocate host memory

    # Input x: (B, C, L)
    var input_heap = ctx.enqueue_create_host_buffer[dtype](batch * dim * seqlen)
    var input_h = TileTensor(
        input_heap,
        row_major((batch, dim, seqlen)),
    )

    # Conv state: (B, C, S)
    var conv_state_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * state_len
    )
    var conv_state_h = TileTensor(
        conv_state_heap,
        row_major((batch, dim, state_len)),
    )

    # Weight: (C, W)
    var weight_heap = ctx.enqueue_create_host_buffer[dtype](dim * width)
    var weight_h = TileTensor(weight_heap, row_major((dim, width)))

    # Bias: (C,)
    var bias_heap = ctx.enqueue_create_host_buffer[dtype](dim)
    var bias_h = TileTensor(bias_heap, row_major((dim)))

    # Output: (B, C, L)
    var result_gpu_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * seqlen
    )
    var result_gpu_h = TileTensor(
        result_gpu_heap,
        row_major((batch, dim, seqlen)),
    )

    var result_cpu_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * seqlen
    )
    var result_cpu_h = TileTensor(
        result_cpu_heap,
        row_major((batch, dim, seqlen)),
    )

    # Copy of conv_state for CPU reference
    var conv_state_cpu_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * state_len
    )
    var conv_state_cpu_h = TileTensor(
        conv_state_cpu_heap,
        row_major((batch, dim, state_len)),
    )

    # Copy of conv_state for GPU
    var conv_state_gpu_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * state_len
    )
    var conv_state_gpu_h = TileTensor(
        conv_state_gpu_heap,
        row_major((batch, dim, state_len)),
    )

    # Initialize input data
    rand[dtype](input_h.unsafe_ptr(), input_h.num_elements())
    rand[dtype](conv_state_h.unsafe_ptr(), conv_state_h.num_elements())
    rand[dtype](weight_h.unsafe_ptr(), weight_h.num_elements())
    rand[dtype](bias_h.unsafe_ptr(), bias_h.num_elements())

    # Copy conv_state for CPU and GPU
    for i in range(batch * dim * state_len):
        conv_state_cpu_h.unsafe_ptr()[i] = conv_state_h.unsafe_ptr()[i]
        conv_state_gpu_h.unsafe_ptr()[i] = conv_state_h.unsafe_ptr()[i]

    var input_buf = input_h
    var conv_state_cpu_buf = conv_state_cpu_h
    var conv_state_gpu_buf = conv_state_gpu_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var result_gpu_buf = result_gpu_h
    var result_cpu_buf = result_cpu_h

    var silu_activation = activation == "silu"
    var silu_activation_int8 = Int8(silu_activation)

    # Allocate device buffers
    var input_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var conv_state_device = ctx.enqueue_create_buffer[dtype](
        batch * dim * state_len
    )
    var weight_device = ctx.enqueue_create_buffer[dtype](dim * width)
    var bias_device = ctx.enqueue_create_buffer[dtype](dim)
    var output_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)

    # Copy data to device
    with ctx.push_context():
        ctx.enqueue_copy(input_device, input_buf.unsafe_ptr())
        ctx.enqueue_copy(conv_state_device, conv_state_gpu_buf.unsafe_ptr())
        ctx.enqueue_copy(weight_device, weight_buf.unsafe_ptr())
        ctx.enqueue_copy(bias_device, bias_buf.unsafe_ptr())

    # Create TileTensors for GPU kernel
    var input_device_tt = TileTensor(
        input_device,
        row_major(batch, dim, seqlen),
    )
    var conv_state_device_tt = TileTensor(
        conv_state_device,
        row_major(batch, dim, state_len),
    )
    var weight_device_tt = TileTensor(
        weight_device,
        row_major(dim, width),
    )
    var bias_device_tt = TileTensor(
        bias_device,
        row_major(
            dim,
        ),
    )
    var output_device_tt = TileTensor(
        output_device,
        row_major(batch, dim, seqlen),
    )

    # Run GPU kernel
    comptime kNThreads = 128
    with ctx.push_context():
        if has_bias:
            var compiled_func = ctx.compile_function[
                causal_conv1d_update_gpu[
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    kNThreads,
                    input_device_tt.LayoutType,
                    conv_state_device_tt.LayoutType,
                    weight_device_tt.LayoutType,
                    output_device_tt.LayoutType,
                    bias_device_tt.LayoutType,
                    input_device_tt.Engine,
                    conv_state_device_tt.Engine,
                    weight_device_tt.Engine,
                    output_device_tt.Engine,
                    bias_device_tt.Engine,
                ]
            ]()
            ctx.enqueue_function(
                compiled_func,
                Int32(batch),
                Int32(dim),
                Int32(seqlen),
                Int32(width),
                Int32(state_len),
                input_device_tt,
                conv_state_device_tt,
                weight_device_tt,
                output_device_tt,
                bias_device_tt,
                silu_activation_int8,
                grid_dim=(batch, ceildiv(dim, kNThreads)),
                block_dim=(kNThreads),
            )
        else:
            var compiled_func = ctx.compile_function[
                causal_conv1d_update_gpu_no_bias[
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    kNThreads,
                    input_device_tt.LayoutType,
                    conv_state_device_tt.LayoutType,
                    weight_device_tt.LayoutType,
                    output_device_tt.LayoutType,
                    input_device_tt.Engine,
                    conv_state_device_tt.Engine,
                    weight_device_tt.Engine,
                    output_device_tt.Engine,
                ]
            ]()
            ctx.enqueue_function(
                compiled_func,
                Int32(batch),
                Int32(dim),
                Int32(seqlen),
                Int32(width),
                Int32(state_len),
                input_device_tt,
                conv_state_device_tt,
                weight_device_tt,
                output_device_tt,
                silu_activation_int8,
                grid_dim=(batch, ceildiv(dim, kNThreads)),
                block_dim=(kNThreads),
            )

    # Copy results back from device
    with ctx.push_context():
        ctx.enqueue_copy(result_gpu_buf.unsafe_ptr(), output_device)
        ctx.enqueue_copy(conv_state_gpu_buf.unsafe_ptr(), conv_state_device)
    ctx.synchronize()

    # Run CPU reference
    if has_bias:
        causal_conv1d_update_cpu[
            dtype,
            dtype,
            dtype,
            dtype,
            dtype,
        ](
            batch,
            dim,
            seqlen,
            width,
            state_len,
            input_buf,
            conv_state_cpu_buf,
            weight_buf,
            result_cpu_buf,
            bias_buf,
            silu_activation,
        )
    else:
        causal_conv1d_update_cpu_no_bias[
            dtype,
            dtype,
            dtype,
            dtype,
        ](
            batch,
            dim,
            seqlen,
            width,
            state_len,
            input_buf,
            conv_state_cpu_buf,
            weight_buf,
            result_cpu_buf,
            silu_activation,
        )

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            result_gpu_h.unsafe_ptr()[i],
            result_cpu_h.unsafe_ptr()[i],
            rtol=rtol,
        )

    # Compare conv_state updates
    var conv_state_size = batch * dim * state_len
    for i in range(conv_state_size):
        assert_almost_equal(
            conv_state_gpu_h.unsafe_ptr()[i],
            conv_state_cpu_h.unsafe_ptr()[i],
            rtol=rtol,
        )


def test_gpu_causal_conv1d_update_basic() raises:
    """Test basic GPU causal conv1d update with bias."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_update_gpu[.float32, True, "none"](2, 8, 1, 3, 4, ctx=ctx)


def test_gpu_causal_conv1d_update_with_silu() raises:
    """Test GPU causal conv1d update with SiLU activation."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_update_gpu[.float32, True, "silu"](2, 8, 1, 3, 4, ctx=ctx)


def test_gpu_causal_conv1d_update_without_bias() raises:
    """Test GPU causal conv1d update without bias."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_update_gpu[.float32, False, "none"](
        2, 8, 1, 3, 4, ctx=ctx
    )


def test_gpu_causal_conv1d_update_seqlen_greater_than_one() raises:
    """Test GPU causal conv1d update with seqlen > 1."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_update_gpu[.float32, True, "none"](2, 8, 4, 3, 4, ctx=ctx)


def test_gpu_causal_conv1d_update_various_widths() raises:
    """Test GPU causal conv1d update with various kernel widths."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_update_gpu[.float32, True, "none"](2, 8, 1, 2, 3, ctx=ctx)
    run_causal_conv1d_update_gpu[.float32, True, "none"](2, 8, 1, 3, 4, ctx=ctx)
    run_causal_conv1d_update_gpu[.float32, True, "none"](2, 8, 1, 4, 5, ctx=ctx)
