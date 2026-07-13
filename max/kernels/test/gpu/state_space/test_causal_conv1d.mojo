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

from std.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.random import rand
from state_space.causal_conv1d import (
    causal_conv1d_channel_first_fwd_gpu,
    causal_conv1d_fwd_cpu,
)
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


@always_inline
def silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[DType.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = Scalar[DType.float32](1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


def run_causal_conv1d_gpu[
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
    """Test causal conv1d GPU kernel against the CPU reference.

    When `has_seq_idx` is set, a packed-sequence mask is supplied (two segments
    split at seqlen//2) so the GPU scalar fallback — which the vectorized fast
    path defers to whenever seq_idx is active — is exercised against the CPU
    reference computing the same masked convolution.
    """
    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    var input_heap = ctx.enqueue_create_host_buffer[dtype](batch * dim * seqlen)
    var input_h = LayoutTensor[dtype, layout_3d, _](
        input_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var weight_heap = ctx.enqueue_create_host_buffer[dtype](dim * width)
    var weight_h = LayoutTensor[dtype, layout_2d, _](
        weight_heap, RuntimeLayout[layout_2d].row_major(Index(dim, width))
    )
    var bias_heap = ctx.enqueue_create_host_buffer[dtype](dim)
    var bias_h = LayoutTensor[dtype, layout_1d, _](
        bias_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )
    # seq_idx (B, L) packed-sequence tags. Always allocated (passed to both the
    # CPU ref and the GPU kernel); only read when has_seq_idx is True.
    var seq_idx_heap = ctx.enqueue_create_host_buffer[dtype](batch * seqlen)
    var seq_idx_h = LayoutTensor[dtype, layout_2d, _](
        seq_idx_heap, RuntimeLayout[layout_2d].row_major(Index(batch, seqlen))
    )
    var result_gpu_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * seqlen
    )
    var result_gpu_h = LayoutTensor[dtype, layout_3d, _](
        result_gpu_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )
    var result_cpu_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * seqlen
    )
    var result_cpu_h = LayoutTensor[dtype, layout_3d, _](
        result_cpu_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    # Initialize input data
    rand[dtype](input_h.ptr, input_h.size())
    rand[dtype](weight_h.ptr, weight_h.size())
    rand[dtype](bias_h.ptr, bias_h.size())
    # Two packed segments split at seqlen // 2 (so taps near the split are
    # masked); a single segment (all zeros) otherwise.
    var split = seqlen // 2
    for b in range(batch):
        for l in range(seqlen):
            seq_idx_h.ptr[b * seqlen + l] = Scalar[dtype](
                1 if (has_seq_idx and l >= split) else 0
            )

    var input_buf = input_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var result_cpu_buf = result_cpu_h

    # Strides for channel-first layout (B, C, L)
    var x_batch_stride: UInt32 = UInt32(dim * seqlen)
    var x_c_stride: UInt32 = UInt32(seqlen)
    var x_l_stride: UInt32 = 1
    var weight_c_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var out_batch_stride: UInt32 = UInt32(dim * seqlen)
    var out_c_stride: UInt32 = UInt32(seqlen)
    var out_l_stride: UInt32 = 1
    var bias_stride: UInt32 = 1
    var seq_idx_batch_stride: UInt32 = UInt32(seqlen)
    var seq_idx_l_stride: UInt32 = 1

    var silu_activation = activation == "silu"

    # Create TileTensors for CPU reference
    var input_tt = TileTensor(input_buf.ptr, row_major(batch, dim, seqlen))
    var weight_tt = TileTensor(weight_buf.ptr, row_major(dim, width))
    var bias_tt = TileTensor(
        bias_buf.ptr,
        row_major(
            dim,
        ),
    )
    var result_cpu_tt = TileTensor(
        result_cpu_buf.ptr, row_major(batch, dim, seqlen)
    )
    var seq_idx_tt = TileTensor(seq_idx_h.ptr, row_major(batch, seqlen))

    # Run CPU reference (same has_seq_idx as the GPU launch below).
    causal_conv1d_fwd_cpu[
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        True,
        has_seq_idx,
    ](
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
        out_batch_stride,
        out_c_stride,
        out_l_stride,
        bias_stride,
        seq_idx_batch_stride,
        seq_idx_l_stride,
        silu_activation,
    )

    # Allocate device buffers
    var input_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var weight_device = ctx.enqueue_create_buffer[dtype](dim * width)
    var bias_device = ctx.enqueue_create_buffer[dtype](dim)
    var seq_idx_device = ctx.enqueue_create_buffer[dtype](batch * seqlen)
    var output_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)

    # Copy data to device
    with ctx.push_context():
        ctx.enqueue_copy(input_device, input_buf.ptr)
        ctx.enqueue_copy(weight_device, weight_buf.ptr)
        ctx.enqueue_copy(bias_device, bias_buf.ptr)
        ctx.enqueue_copy(seq_idx_device, seq_idx_h.ptr)

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
    var seq_idx_device_tt = TileTensor(
        seq_idx_device,
        row_major(batch, seqlen),
    )
    var output_device_tt = TileTensor(
        output_device,
        row_major(batch, dim, seqlen),
    )

    # Run GPU kernel. seq_idx is unused (has_seq_idx=False); bias_device_tt
    # stands in as a valid tensor argument and is never dereferenced.
    var silu_activation_int8 = Int8(silu_activation)

    @parameter
    @always_inline
    def launch[kWidth: Int]() raises:
        var compiled_func = ctx.compile_function[
            causal_conv1d_channel_first_fwd_gpu[
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
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                bias_stride,
                seq_idx_batch_stride,
                seq_idx_l_stride,
                Int8(True),
                Int8(has_seq_idx),
                silu_activation_int8,
                grid_dim=(ceildiv(seqlen, kNThreads * kNElts), dim, batch),
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

    # Copy GPU results back to host
    with ctx.push_context():
        ctx.enqueue_copy(result_gpu_h.ptr, output_device)
    ctx.synchronize()

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            result_gpu_h.ptr[i],
            result_cpu_h.ptr[i],
            rtol=rtol,
        )


def test_basic_gpu_causal_conv1d() raises:
    """Test basic GPU causal conv1d without activation."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_gpu[DType.float32, "none"](2, 4, 8, 3, ctx=ctx)


def test_gpu_causal_conv1d_with_silu() raises:
    """Test GPU causal conv1d with SiLU activation."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_gpu[DType.float32, "silu"](2, 4, 8, 3, ctx=ctx)


def test_gpu_causal_conv1d_width_1() raises:
    """Test GPU causal conv1d with kernel width 1."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_gpu[DType.float32, "none"](2, 8, 16, 1, ctx=ctx)


def test_gpu_causal_conv1d_width_2() raises:
    """Test GPU causal conv1d with kernel width 2."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_gpu[DType.float32, "none"](2, 8, 16, 2, ctx=ctx)


def test_gpu_causal_conv1d_width_3() raises:
    """Test GPU causal conv1d with kernel width 3."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_gpu[DType.float32, "none"](2, 8, 16, 3, ctx=ctx)


def test_gpu_causal_conv1d_width_4() raises:
    """Test GPU causal conv1d with kernel width 4."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_gpu[DType.float32, "none"](2, 8, 16, 4, ctx=ctx)


def test_gpu_causal_conv1d_large_sequence() raises:
    """Test GPU causal conv1d with larger sequence length."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_gpu[DType.float32, "none"](2, 16, 128, 3, ctx=ctx)


def test_gpu_causal_conv1d_mamba_dimensions() raises:
    """Test GPU causal conv1d with mamba-130m-hf realistic dimensions."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    # dim=1536, width=4 (conv_kernel)
    for seqlen in [5, 6, 7]:
        run_causal_conv1d_gpu[DType.float32, "silu"](
            1, 1536, seqlen, 4, ctx=ctx
        )


def test_gpu_causal_conv1d_strict_tolerance() raises:
    """Test GPU causal conv1d with strict tolerance (0.01%)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_gpu[DType.float32, "silu"](
        1, 1536, 7, 4, ctx=ctx, rtol=0.0001
    )


def test_gpu_causal_conv1d_vectorized_fast_path() raises:
    """Exercise the vectorized fast path (vector load + warp-shuffle halo).

    The fast path engages only when L-contiguous, no seq_idx, kNElts-aligned,
    and blocks are full: seqlen % (kNThreads * kNElts) == 0. The launcher here
    uses kNThreads=128, kNElts=4, so seqlen must be a multiple of 512. Covers
    every width (the halo spans width-1 prior elements, pulled across lanes by
    warp shuffle) and both activations, with multiple channels/batches so the
    per-(batch, channel) base offsets are exercised.
    """
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    for width in [1, 2, 3, 4]:
        # Single full block (seqlen == kNThreads*kNElts) and two full blocks.
        run_causal_conv1d_gpu[DType.float32, "none"](2, 8, 512, width, ctx=ctx)
        run_causal_conv1d_gpu[DType.float32, "silu"](2, 8, 512, width, ctx=ctx)
        run_causal_conv1d_gpu[DType.float32, "silu"](1, 4, 1024, width, ctx=ctx)


def test_gpu_causal_conv1d_vectorized_mamba_prefill() raises:
    """Vectorized fast path at mamba prefill dimensions (dim=1536, L=512)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_gpu[DType.float32, "silu"](1, 1536, 512, 4, ctx=ctx)


def test_gpu_causal_conv1d_bf16_fast_path() raises:
    """bf16 fast path: 128-bit load is kNElts=8, accumulation stays float32.

    64 threads x 8 bf16 == 512-position tile, full utilization. bf16 carries
    ~8 mantissa bits, so the tolerance is looser than fp32 but the float32
    accumulator keeps it well within a few bf16 ULPs.
    """
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    for width in [1, 2, 3, 4]:
        run_causal_conv1d_gpu[DType.bfloat16, "none", 64, 8](
            2, 8, 512, width, ctx=ctx, rtol=0.03
        )
        run_causal_conv1d_gpu[DType.bfloat16, "silu", 64, 8](
            1, 1536, 512, width, ctx=ctx, rtol=0.03
        )


def test_gpu_causal_conv1d_fp16_fast_path() raises:
    """fp16 fast path: kNElts=8 (128-bit), float32 accumulation."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    for width in [2, 4]:
        run_causal_conv1d_gpu[DType.float16, "silu", 64, 8](
            1, 1536, 512, width, ctx=ctx, rtol=0.01
        )


def test_gpu_causal_conv1d_seq_idx() raises:
    """Packed-sequence (seq_idx) path: the vectorized fast path defers to the
    scalar fallback whenever seq_idx is active, so this validates that fallback
    (with masking across the segment boundary) against the CPU reference.
    Covers all widths and both a full-tile (L=512) and a small (L=16) shape.
    """
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    for width in [1, 2, 3, 4]:
        run_causal_conv1d_gpu[DType.float32, "silu", 128, 4, True](
            2, 8, 16, width, ctx=ctx
        )
        run_causal_conv1d_gpu[DType.float32, "silu", 128, 4, True](
            1, 1536, 512, width, ctx=ctx
        )


def test_gpu_causal_conv1d_datacenter_scale() raises:
    """Correctness at larger-model / datacenter shapes.

    Wide channels (grid.y = dim) and long sequences (grid.x scales with L) and
    multi-stream batches (grid.z = batch). Confirms the kernel stays correct and
    within grid limits (dim, batch <= 65535) at scale; the op launches the same
    64x4 config, whose grid.x grows with L so every block stays full.
    """
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    # Wide model (mamba-1.4B-ish dim) with a long prefill.
    run_causal_conv1d_gpu[DType.float32, "silu"](1, 4096, 2048, 4, ctx=ctx)
    # Batched prefill (multi-stream serving).
    run_causal_conv1d_gpu[DType.float32, "silu"](8, 1536, 512, 4, ctx=ctx)
    # bf16 wide model.
    run_causal_conv1d_gpu[DType.bfloat16, "silu", 64, 8](
        2, 5120, 512, 4, ctx=ctx, rtol=0.03
    )
