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
"""GPU tests for the channel-last causal conv1d kernel.

Runs `causal_conv1d_channel_last_fwd_gpu` over (B, L, C) inputs and compares
against the shared CPU core (`causal_conv1d_fwd_cpu`) driven with channel-last
strides. Covers widths 1-4 and SiLU on/off.
"""

from std.math import ceildiv

from std.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.random import rand
from state_space.causal_conv1d import (
    causal_conv1d_channel_last_fwd_gpu,
    causal_conv1d_fwd_cpu,
)
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def run_causal_conv1d_channel_last_gpu[
    dtype: DType,
    activation: StaticString,
    kNThreads: Int = 128,
    kNElts: Int = 4,
    has_seq_idx: Bool = False,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Compare the channel-last GPU kernel against the CPU core.

    When `has_seq_idx` is set, a two-segment packed-sequence mask (split at
    seqlen//2) is supplied so the per-tap seq_idx scalar path — which the
    coalesced fast path defers to — is validated against the CPU reference.
    """
    var n = batch * seqlen * dim

    var input_heap = ctx.enqueue_create_host_buffer[dtype](n)
    var weight_heap = ctx.enqueue_create_host_buffer[dtype](dim * width)
    var bias_heap = ctx.enqueue_create_host_buffer[dtype](dim)
    var seq_idx_heap = ctx.enqueue_create_host_buffer[dtype](batch * seqlen)
    var result_gpu_heap = ctx.enqueue_create_host_buffer[dtype](n)
    var result_cpu_heap = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.synchronize()

    rand[dtype](input_heap.unsafe_ptr(), n)
    rand[dtype](weight_heap.unsafe_ptr(), dim * width)
    rand[dtype](bias_heap.unsafe_ptr(), dim)
    var split = seqlen // 2
    for bb in range(batch):
        for ll in range(seqlen):
            seq_idx_heap.unsafe_ptr()[bb * seqlen + ll] = Scalar[dtype](
                1 if (has_seq_idx and ll >= split) else 0
            )

    # Channel-last strides for (B, L, C).
    var x_batch_stride: UInt32 = UInt32(seqlen * dim)
    var x_l_stride: UInt32 = UInt32(dim)
    var x_c_stride: UInt32 = 1
    var weight_c_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var bias_stride: UInt32 = 1
    var seq_idx_batch_stride: UInt32 = UInt32(seqlen)
    var seq_idx_l_stride: UInt32 = 1

    var silu_activation = activation == "silu"

    # CPU reference via the shared stride-driven core (channel-last strides).
    var input_tt = TileTensor(
        input_heap.unsafe_ptr(), row_major(batch, seqlen, dim)
    )
    var weight_tt = TileTensor(weight_heap.unsafe_ptr(), row_major(dim, width))
    var bias_tt = TileTensor(bias_heap.unsafe_ptr(), row_major(dim))
    var seq_idx_tt = TileTensor(
        seq_idx_heap.unsafe_ptr(), row_major(batch, seqlen)
    )
    var result_cpu_tt = TileTensor(
        result_cpu_heap.unsafe_ptr(), row_major(batch, seqlen, dim)
    )

    causal_conv1d_fwd_cpu[dtype, dtype, dtype, dtype, dtype, True, has_seq_idx](
        batch,
        dim,
        seqlen,
        width,
        input_tt.as_immut(),
        weight_tt.as_immut(),
        result_cpu_tt,
        bias_tt.as_immut(),
        seq_idx_tt.as_immut(),
        x_batch_stride,
        x_c_stride,
        x_l_stride,
        weight_c_stride,
        weight_width_stride,
        x_batch_stride,
        x_c_stride,
        x_l_stride,
        bias_stride,
        seq_idx_batch_stride,
        seq_idx_l_stride,
        silu_activation,
    )

    # Device buffers.
    var input_device = ctx.enqueue_create_buffer[dtype](n)
    var weight_device = ctx.enqueue_create_buffer[dtype](dim * width)
    var bias_device = ctx.enqueue_create_buffer[dtype](dim)
    var seq_idx_device = ctx.enqueue_create_buffer[dtype](batch * seqlen)
    var output_device = ctx.enqueue_create_buffer[dtype](n)

    with ctx.push_context():
        ctx.enqueue_copy(input_device, input_heap.unsafe_ptr())
        ctx.enqueue_copy(weight_device, weight_heap.unsafe_ptr())
        ctx.enqueue_copy(bias_device, bias_heap.unsafe_ptr())
        ctx.enqueue_copy(seq_idx_device, seq_idx_heap.unsafe_ptr())

    var input_device_tt = TileTensor(
        input_device, row_major(batch, seqlen, dim)
    )
    var weight_device_tt = TileTensor(weight_device, row_major(dim, width))
    var bias_device_tt = TileTensor(bias_device, row_major(dim))
    var seq_idx_device_tt = TileTensor(seq_idx_device, row_major(batch, seqlen))
    var output_device_tt = TileTensor(
        output_device, row_major(batch, seqlen, dim)
    )
    var silu_activation_int8 = Int8(silu_activation)

    @parameter
    @always_inline
    def launch[kWidth: Int]() raises:
        var compiled_func = ctx.compile_function[
            causal_conv1d_channel_last_fwd_gpu[
                dtype,
                dtype,
                dtype,
                kNThreads,
                kWidth,
                kNElts,
                dtype,
                dtype,
                input_device_tt.LayoutType,
                weight_device_tt.LayoutType,
                output_device_tt.LayoutType,
                bias_device_tt.LayoutType,
                seq_idx_device_tt.LayoutType,
            ]
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                batch,
                dim,
                seqlen,
                width,
                input_device_tt.as_immut(),
                weight_device_tt.as_immut(),
                output_device_tt,
                bias_device_tt.as_immut(),
                seq_idx_device_tt.as_immut(),
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                bias_stride,
                seq_idx_batch_stride,
                seq_idx_l_stride,
                Int8(True),
                Int8(has_seq_idx),
                silu_activation_int8,
                grid_dim=(
                    ceildiv(seqlen, kNElts),
                    ceildiv(dim, kNThreads),
                    batch,
                ),
                block_dim=(kNThreads),
            )

    if width == 1:
        launch[1]()
    elif width == 2:
        launch[2]()
    elif width == 3:
        launch[3]()
    elif width == 4:
        launch[4]()
    else:
        raise Error(
            "Unsupported kernel width: only widths 1, 2, 3, 4 are supported"
        )

    with ctx.push_context():
        ctx.enqueue_copy(result_gpu_heap.unsafe_ptr(), output_device)
    ctx.synchronize()

    for i in range(n):
        assert_almost_equal(
            result_gpu_heap.unsafe_ptr()[i],
            result_cpu_heap.unsafe_ptr()[i],
            rtol=rtol,
        )


def test_channel_last_gpu_basic() raises:
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_channel_last_gpu[DType.float32, "none"](
        2, 4, 8, 3, ctx=ctx
    )


def test_channel_last_gpu_silu() raises:
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_channel_last_gpu[DType.float32, "silu"](
        2, 4, 8, 3, ctx=ctx
    )


def test_channel_last_gpu_width_1() raises:
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_channel_last_gpu[DType.float32, "none"](
        2, 8, 16, 1, ctx=ctx
    )


def test_channel_last_gpu_width_2() raises:
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_channel_last_gpu[DType.float32, "none"](
        2, 8, 16, 2, ctx=ctx
    )


def test_channel_last_gpu_width_3() raises:
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_channel_last_gpu[DType.float32, "none"](
        2, 8, 16, 3, ctx=ctx
    )


def test_channel_last_gpu_width_4() raises:
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_channel_last_gpu[DType.float32, "none"](
        2, 8, 16, 4, ctx=ctx
    )


def test_channel_last_gpu_silu_width_4() raises:
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_channel_last_gpu[DType.float32, "silu"](
        2, 8, 16, 4, ctx=ctx
    )


def test_channel_last_gpu_large() raises:
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_channel_last_gpu[DType.float32, "none"](
        2, 16, 64, 4, ctx=ctx
    )


def test_channel_last_gpu_ragged_channels() raises:
    """dim not a multiple of kNElts exercises the scalar fallback (the last
    channel chunk is partial, so the vectorized channel load is skipped)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    for d in [6, 10, 1535]:
        run_causal_conv1d_channel_last_gpu[DType.float32, "silu"](
            1, d, 64, 4, ctx=ctx
        )


def test_channel_last_gpu_mamba_dims() raises:
    """Vectorized channel-last fast path at mamba dims (dim=1536)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_channel_last_gpu[DType.float32, "silu"](
        1, 1536, 256, 4, ctx=ctx
    )


def test_channel_last_gpu_bf16() raises:
    """bf16 channel-last fast path (kNElts=8, 128-bit channel vector)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    for width in [2, 4]:
        run_causal_conv1d_channel_last_gpu[DType.bfloat16, "silu", 128, 8](
            1, 1536, 256, width, ctx=ctx, rtol=0.03
        )


def test_channel_last_gpu_seq_idx() raises:
    """Packed-sequence (seq_idx) path: the coalesced fast path defers to the
    per-tap scalar scan when seq_idx is active. Validates that masked scan
    against the CPU reference across all widths."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    for width in [1, 2, 3, 4]:
        run_causal_conv1d_channel_last_gpu[DType.float32, "silu", 64, 8, True](
            2, 16, 64, width, ctx=ctx
        )
        run_causal_conv1d_channel_last_gpu[DType.float32, "silu", 64, 8, True](
            1, 1536, 256, width, ctx=ctx
        )
