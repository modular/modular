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
    causal_conv1d_channel_first_fwd_cpu,
    causal_conv1d_channel_first_fwd_gpu,
)
from std.testing import TestSuite, assert_almost_equal, assert_true


def main() raises:
    var suite = TestSuite()
    suite.test[test_basic_gpu_causal_conv1d]()
    suite.test[test_gpu_causal_conv1d_with_silu]()
    suite.test[test_gpu_causal_conv1d_width_1]()
    suite.test[test_gpu_causal_conv1d_width_2]()
    suite.test[test_gpu_causal_conv1d_width_3]()
    suite.test[test_gpu_causal_conv1d_width_4]()
    suite.test[test_gpu_causal_conv1d_large_sequence]()
    suite.test[test_gpu_causal_conv1d_mamba_dimensions]()
    suite.test[test_gpu_causal_conv1d_strict_tolerance]()
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


def run_causal_conv1d_gpu[
    dtype: DType,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test causal conv1d GPU kernel against CPU reference."""
    # Allocate host memory

    var input_heap = ctx.enqueue_create_host_buffer[dtype](batch * dim * seqlen)
    var input_h = TileTensor(
        input_heap,
        row_major((batch, dim, seqlen)),
    )
    var weight_heap = ctx.enqueue_create_host_buffer[dtype](dim * width)
    var weight_h = TileTensor(weight_heap, row_major((dim, width)))
    var bias_heap = ctx.enqueue_create_host_buffer[dtype](dim)
    var bias_h = TileTensor(bias_heap, row_major((dim)))
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

    # Initialize input data
    rand[dtype](input_h.unsafe_ptr(), input_h.num_elements())
    rand[dtype](weight_h.unsafe_ptr(), weight_h.num_elements())
    rand[dtype](bias_h.unsafe_ptr(), bias_h.num_elements())

    var input_buf = input_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var result_cpu_buf = result_cpu_h

    var silu_activation = activation == "silu"

    # Run CPU reference
    causal_conv1d_channel_first_fwd_cpu[
        dtype,
        dtype,
        dtype,
        dtype,
    ](
        batch,
        dim,
        seqlen,
        width,
        input_buf,
        weight_buf,
        result_cpu_buf,
        bias_buf,
        silu_activation,
    )

    # Allocate device buffers
    var input_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var weight_device = ctx.enqueue_create_buffer[dtype](dim * width)
    var bias_device = ctx.enqueue_create_buffer[dtype](dim)
    var output_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)

    # Copy data to device
    with ctx.push_context():
        ctx.enqueue_copy(input_device, input_buf.unsafe_ptr())
        ctx.enqueue_copy(weight_device, weight_buf.unsafe_ptr())
        ctx.enqueue_copy(bias_device, bias_buf.unsafe_ptr())

    # Create TileTensors for GPU kernel
    var input_device_tt = TileTensor(
        input_device,
        row_major(batch, dim, seqlen),
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
    comptime kNElts = 4

    if width == 1:
        comptime kWidth = 1
        var compiled_func = ctx.compile_function[
            causal_conv1d_channel_first_fwd_gpu[
                dtype,
                dtype,
                dtype,
                kNThreads,
                kWidth,
                kNElts,
                dtype,
                input_device_tt.LayoutType,
                weight_device_tt.LayoutType,
                output_device_tt.LayoutType,
                bias_device_tt.LayoutType,
                input_device_tt.Engine,
                weight_device_tt.Engine,
                output_device_tt.Engine,
                bias_device_tt.Engine,
            ]
        ]()
        var silu_activation_int8 = Int8(silu_activation)
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                Int32(batch),
                Int32(dim),
                Int32(seqlen),
                Int32(width),
                input_device_tt,
                weight_device_tt,
                output_device_tt,
                bias_device_tt,
                silu_activation_int8,
                grid_dim=(ceildiv(seqlen, kNThreads * kNElts), dim, batch),
                block_dim=(kNThreads),
            )
    elif width == 2:
        comptime kWidth = 2
        var compiled_func = ctx.compile_function[
            causal_conv1d_channel_first_fwd_gpu[
                dtype,
                dtype,
                dtype,
                kNThreads,
                kWidth,
                kNElts,
                dtype,
                input_device_tt.LayoutType,
                weight_device_tt.LayoutType,
                output_device_tt.LayoutType,
                bias_device_tt.LayoutType,
                input_device_tt.Engine,
                weight_device_tt.Engine,
                output_device_tt.Engine,
                bias_device_tt.Engine,
            ]
        ]()
        var silu_activation_int8 = Int8(silu_activation)
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                Int32(batch),
                Int32(dim),
                Int32(seqlen),
                Int32(width),
                input_device_tt,
                weight_device_tt,
                output_device_tt,
                bias_device_tt,
                silu_activation_int8,
                grid_dim=(ceildiv(seqlen, kNThreads * kNElts), dim, batch),
                block_dim=(kNThreads),
            )
    elif width == 3:
        comptime kWidth = 3
        var compiled_func = ctx.compile_function[
            causal_conv1d_channel_first_fwd_gpu[
                dtype,
                dtype,
                dtype,
                kNThreads,
                kWidth,
                kNElts,
                dtype,
                input_device_tt.LayoutType,
                weight_device_tt.LayoutType,
                output_device_tt.LayoutType,
                bias_device_tt.LayoutType,
                input_device_tt.Engine,
                weight_device_tt.Engine,
                output_device_tt.Engine,
                bias_device_tt.Engine,
            ]
        ]()
        var silu_activation_int8 = Int8(silu_activation)
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                Int32(batch),
                Int32(dim),
                Int32(seqlen),
                Int32(width),
                input_device_tt,
                weight_device_tt,
                output_device_tt,
                bias_device_tt,
                silu_activation_int8,
                grid_dim=(ceildiv(seqlen, kNThreads * kNElts), dim, batch),
                block_dim=(kNThreads),
            )
    elif width == 4:
        comptime kWidth = 4
        var compiled_func = ctx.compile_function[
            causal_conv1d_channel_first_fwd_gpu[
                dtype,
                dtype,
                dtype,
                kNThreads,
                kWidth,
                kNElts,
                dtype,
                input_device_tt.LayoutType,
                weight_device_tt.LayoutType,
                output_device_tt.LayoutType,
                bias_device_tt.LayoutType,
                input_device_tt.Engine,
                weight_device_tt.Engine,
                output_device_tt.Engine,
                bias_device_tt.Engine,
            ]
        ]()
        var silu_activation_int8 = Int8(silu_activation)
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                Int32(batch),
                Int32(dim),
                Int32(seqlen),
                Int32(width),
                input_device_tt,
                weight_device_tt,
                output_device_tt,
                bias_device_tt,
                silu_activation_int8,
                grid_dim=(ceildiv(seqlen, kNThreads * kNElts), dim, batch),
                block_dim=(kNThreads),
            )
    else:
        raise Error(
            "Unsupported kernel width: only widths 1, 2, 3, 4 are supported"
        )

    # Copy GPU results back to host
    with ctx.push_context():
        ctx.enqueue_copy(result_gpu_h.unsafe_ptr(), output_device)
    ctx.synchronize()

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            result_gpu_h.unsafe_ptr()[i],
            result_cpu_h.unsafe_ptr()[i],
            rtol=rtol,
        )


def test_basic_gpu_causal_conv1d() raises:
    """Test basic GPU causal conv1d without activation."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_gpu[.float32, "none"](2, 4, 8, 3, ctx=ctx)


def test_gpu_causal_conv1d_with_silu() raises:
    """Test GPU causal conv1d with SiLU activation."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_gpu[.float32, "silu"](2, 4, 8, 3, ctx=ctx)


def test_gpu_causal_conv1d_width_1() raises:
    """Test GPU causal conv1d with kernel width 1."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_gpu[.float32, "none"](2, 8, 16, 1, ctx=ctx)


def test_gpu_causal_conv1d_width_2() raises:
    """Test GPU causal conv1d with kernel width 2."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_gpu[.float32, "none"](2, 8, 16, 2, ctx=ctx)


def test_gpu_causal_conv1d_width_3() raises:
    """Test GPU causal conv1d with kernel width 3."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_gpu[.float32, "none"](2, 8, 16, 3, ctx=ctx)


def test_gpu_causal_conv1d_width_4() raises:
    """Test GPU causal conv1d with kernel width 4."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_gpu[.float32, "none"](2, 8, 16, 4, ctx=ctx)


def test_gpu_causal_conv1d_large_sequence() raises:
    """Test GPU causal conv1d with larger sequence length."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_gpu[.float32, "none"](2, 16, 128, 3, ctx=ctx)


def test_gpu_causal_conv1d_mamba_dimensions() raises:
    """Test GPU causal conv1d with mamba-130m-hf realistic dimensions."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    # dim=1536, width=4 (conv_kernel)
    for seqlen in [5, 6, 7]:
        run_causal_conv1d_gpu[.float32, "silu"](1, 1536, seqlen, 4, ctx=ctx)


def test_gpu_causal_conv1d_strict_tolerance() raises:
    """Test GPU causal conv1d with strict tolerance (0.01%)."""
    var ctx = DeviceContext()
    assert_true(ctx.is_compatible(), "The GPU context must be compatible")
    run_causal_conv1d_gpu[.float32, "silu"](1, 1536, 7, 4, ctx=ctx, rtol=0.0001)
