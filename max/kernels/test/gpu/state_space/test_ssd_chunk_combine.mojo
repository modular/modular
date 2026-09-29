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
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from std.random import rand
from state_space.ssd_chunk_combine import (
    ssd_output_recombination_fwd_gpu,
    ssd_output_recombination_fwd_gpu_fused,
    ssd_output_recombination_fwd_gpu_static,
)
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index


def run_ssd_output_recombination_gpu[
    dtype: DType,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Run the SSD output-recombination GPU kernel and check vs a CPU ref.

    One GPU thread per ``(batch, chunk, head, l, p)`` output scalar; this
    validates the result against the same naive recombination used by the CPU
    test.
    """
    comptime SCALAR_BLOCK_SIZE = 128
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var c_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    var a_count = batch * n_chunks * n_heads * chunk_len
    var y_count = batch * n_chunks * n_heads * chunk_len * head_dim

    # ── Host tensors ─────────────────────────────────────────────────────────
    var c_heap = ctx.enqueue_create_host_buffer[dtype](c_count)
    var c_h = LayoutTensor[dtype, layout_5d, _](
        c_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    rand[dtype](c_h.ptr, c_h.size())

    var ent_heap = ctx.enqueue_create_host_buffer[dtype](ent_count)
    var ent_h = LayoutTensor[dtype, layout_5d, _](
        ent_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )
    rand[dtype](ent_h.ptr, ent_h.size())

    var a_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    var a_h = LayoutTensor[dtype, layout_4d, _](
        a_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    rand[dtype](a_h.ptr, a_h.size())
    # Make A negative (a real SSM decays). `rand` fills in [0, 1); map into
    # [-0.6, -0.1] like a real per-token in-chunk log-decay.
    for i in range(a_count):
        var v = a_h.ptr[i].cast[DType.float32]()
        # Map [0, 1) -> [-0.6, -0.1].
        var scaled = Float32(-0.6) + Float32(0.5) * v
        a_h.ptr[i] = Scalar[dtype](scaled)

    var yd_heap = ctx.enqueue_create_host_buffer[dtype](y_count)
    var yd_h = LayoutTensor[dtype, layout_5d, _](
        yd_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    rand[dtype](yd_h.ptr, yd_h.size())

    var y_gpu_heap = ctx.enqueue_create_host_buffer[dtype](y_count)

    # ── Device buffers ───────────────────────────────────────────────────────
    var c_device = ctx.enqueue_create_buffer[dtype](c_count)
    var ent_device = ctx.enqueue_create_buffer[dtype](ent_count)
    var a_device = ctx.enqueue_create_buffer[dtype](a_count)
    var yd_device = ctx.enqueue_create_buffer[dtype](y_count)
    var y_device = ctx.enqueue_create_buffer[dtype](y_count)

    with ctx.push_context():
        ctx.enqueue_copy(c_device, c_h.ptr)
        ctx.enqueue_copy(ent_device, ent_h.ptr)
        ctx.enqueue_copy(a_device, a_h.ptr)
        ctx.enqueue_copy(yd_device, yd_h.ptr)

    var c_tt = TileTensor(
        c_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var ent_tt = TileTensor(
        ent_device,
        row_major(batch, n_chunks, n_heads, head_dim, state_dim),
    )
    var a_tt = TileTensor(
        a_device,
        row_major(batch, n_chunks, n_heads, chunk_len),
    )
    var yd_tt = TileTensor(
        yd_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )
    var y_tt = TileTensor(
        y_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )

    var total_threads = batch * n_chunks * n_heads * chunk_len * head_dim

    var compiled_func = ctx.compile_function[
        ssd_output_recombination_fwd_gpu[
            dtype,
            c_tt.LayoutType,
            ent_tt.LayoutType,
            a_tt.LayoutType,
            yd_tt.LayoutType,
            y_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len),
            Int32(head_dim),
            Int32(state_dim),
            c_tt,
            ent_tt,
            a_tt,
            yd_tt,
            y_tt,
            grid_dim=(ceildiv(total_threads, SCALAR_BLOCK_SIZE),),
            block_dim=(SCALAR_BLOCK_SIZE,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(y_gpu_heap, y_device)
    ctx.synchronize()

    # ── CPU reference: naive output recombination ────────────────────────────
    var ref_y = ctx.enqueue_create_host_buffer[dtype](y_count)

    var c_l_stride = state_dim
    var c_h_stride = chunk_len * c_l_stride
    var c_c_stride = n_heads * c_h_stride
    var c_b_stride = n_chunks * c_c_stride

    var ent_p_stride = state_dim
    var ent_h_stride = head_dim * ent_p_stride
    var ent_c_stride = n_heads * ent_h_stride
    var ent_b_stride = n_chunks * ent_c_stride

    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    var y_l_stride = head_dim
    var y_h_stride = chunk_len * y_l_stride
    var y_c_stride = n_heads * y_h_stride
    var y_b_stride = n_chunks * y_c_stride

    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var a_base = b * a_b_stride + c * a_c_stride + h * a_h_stride
                var c_base = b * c_b_stride + c * c_c_stride + h * c_h_stride
                var ent_base = (
                    b * ent_b_stride + c * ent_c_stride + h * ent_h_stride
                )
                var y_base = b * y_b_stride + c * y_c_stride + h * y_h_stride

                var a_cumsum = Float32(0.0)
                for l in range(chunk_len):
                    a_cumsum += a_h.ptr[a_base + l].cast[DType.float32]()
                    var decay = exp(a_cumsum)
                    for p in range(head_dim):
                        var acc = Float32(0.0)
                        for n in range(state_dim):
                            var c_val = c_h.ptr[
                                c_base + l * c_l_stride + n
                            ].cast[DType.float32]()
                            var ent_val = ent_h.ptr[
                                ent_base + p * ent_p_stride + n
                            ].cast[DType.float32]()
                            acc += c_val * ent_val
                        var y_off = decay * acc
                        var yd_val = yd_h.ptr[y_base + l * y_l_stride + p].cast[
                            DType.float32
                        ]()
                        ref_y[y_base + l * y_l_stride + p] = Scalar[dtype](
                            yd_val + y_off
                        )

    # ── Compare GPU vs CPU reference ─────────────────────────────────────────
    for i in range(y_count):
        assert_almost_equal(y_gpu_heap[i], ref_y[i], rtol=rtol)


def test_ssd_output_recombination_gpu_golden() raises:
    """Golden-parity test on GPU against the canonical PyTorch SSD reference.

    Same FIXED tiny input as the CPU golden test (batch=1, n_chunks=1,
    n_heads=1, L=4, N=3, P=2). Pins the GPU kernel to the stage-4 output
    recombination of ``ssd_minimal_discrete``.
    """
    var ctx = DeviceContext()

    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 1
    comptime n_heads = 1
    comptime chunk_len = 4
    comptime state_dim = 3
    comptime head_dim = 2

    var c_vals = [
        Float32(0.35),
        -0.57,
        -0.39,
        -0.29,
        -0.45,
        0.92,
        -0.76,
        1.06,
        -0.12,
        0.58,
        0.47,
        0.30,
    ]
    var ent_vals = [
        Float32(0.56),
        2.29,
        2.11,
        0.30,
        0.03,
        -0.57,
    ]
    var a_vals = [Float32(-0.51), -0.14, -0.32, -0.54]
    var yd_vals = [
        Float32(0.24),
        0.69,
        1.40,
        -0.14,
        -0.35,
        -2.05,
        -0.44,
        -1.53,
    ]
    var y_expected = [
        Float32(-0.920277),
        0.876274,
        1.790647,
        -0.466226,
        0.312865,
        -2.098447,
        0.009353,
        -1.526222,
    ]

    var c_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    var a_count = batch * n_chunks * n_heads * chunk_len
    var y_count = batch * n_chunks * n_heads * chunk_len * head_dim

    # Host buffers from fixed literals.
    var c_heap = ctx.enqueue_create_host_buffer[dtype](c_count)
    for i in range(c_count):
        c_heap[i] = Scalar[dtype](c_vals[i])
    var ent_heap = ctx.enqueue_create_host_buffer[dtype](ent_count)
    for i in range(ent_count):
        ent_heap[i] = Scalar[dtype](ent_vals[i])
    var a_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    for i in range(a_count):
        a_heap[i] = Scalar[dtype](a_vals[i])
    var yd_heap = ctx.enqueue_create_host_buffer[dtype](y_count)
    for i in range(y_count):
        yd_heap[i] = Scalar[dtype](yd_vals[i])
    var y_heap = ctx.enqueue_create_host_buffer[dtype](y_count)

    var c_device = ctx.enqueue_create_buffer[dtype](c_count)
    var ent_device = ctx.enqueue_create_buffer[dtype](ent_count)
    var a_device = ctx.enqueue_create_buffer[dtype](a_count)
    var yd_device = ctx.enqueue_create_buffer[dtype](y_count)
    var y_device = ctx.enqueue_create_buffer[dtype](y_count)

    with ctx.push_context():
        ctx.enqueue_copy(c_device, c_heap)
        ctx.enqueue_copy(ent_device, ent_heap)
        ctx.enqueue_copy(a_device, a_heap)
        ctx.enqueue_copy(yd_device, yd_heap)

    var c_tt = TileTensor(
        c_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var ent_tt = TileTensor(
        ent_device,
        row_major(batch, n_chunks, n_heads, head_dim, state_dim),
    )
    var a_tt = TileTensor(
        a_device,
        row_major(batch, n_chunks, n_heads, chunk_len),
    )
    var yd_tt = TileTensor(
        yd_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )
    var y_tt = TileTensor(
        y_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )

    var total_threads = batch * n_chunks * n_heads * chunk_len * head_dim

    var compiled_func = ctx.compile_function[
        ssd_output_recombination_fwd_gpu[
            dtype,
            c_tt.LayoutType,
            ent_tt.LayoutType,
            a_tt.LayoutType,
            yd_tt.LayoutType,
            y_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len),
            Int32(head_dim),
            Int32(state_dim),
            c_tt,
            ent_tt,
            a_tt,
            yd_tt,
            y_tt,
            grid_dim=(1,),
            block_dim=(total_threads,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(y_heap, y_device)
    ctx.synchronize()

    for i in range(y_count):
        assert_almost_equal(
            y_heap[i],
            Scalar[dtype](y_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )


def test_ssd_output_recombination_gpu_basic() raises:
    """Basic single-batch case against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_output_recombination_gpu[DType.float32](
        batch=1,
        n_chunks=2,
        n_heads=2,
        chunk_len=8,
        head_dim=8,
        state_dim=16,
        ctx=ctx,
    )


def test_ssd_output_recombination_gpu_multi_batch() raises:
    """Multiple batch elements and heads against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_output_recombination_gpu[DType.float32](
        batch=2,
        n_chunks=3,
        n_heads=3,
        chunk_len=5,
        head_dim=4,
        state_dim=8,
        ctx=ctx,
    )


def run_ssd_output_recombination_gpu_static[
    dtype: DType,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
    use_fused: Bool = False,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Validate the static / fused tensor-core path vs the naive CPU reference.

    Uses gate-hitting compile-time dims (``chunk_len % 128 == 0``,
    ``state_dim % 32 == 0`` and ``>= 128``) so ``batched_matmul`` takes the A100
    multistage path. With ``use_fused=True``, exercises the single-pass fused
    MMA path instead.
    """
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var c_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    var a_count = batch * n_chunks * n_heads * chunk_len
    var y_count = batch * n_chunks * n_heads * chunk_len * head_dim

    var c_heap = ctx.enqueue_create_host_buffer[dtype](c_count)
    var c_h = LayoutTensor[dtype, layout_5d, _](
        c_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    rand[dtype](c_h.ptr, c_h.size())

    var ent_heap = ctx.enqueue_create_host_buffer[dtype](ent_count)
    var ent_h = LayoutTensor[dtype, layout_5d, _](
        ent_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )
    rand[dtype](ent_h.ptr, ent_h.size())

    var a_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    var a_h = LayoutTensor[dtype, layout_4d, _](
        a_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    rand[dtype](a_h.ptr, a_h.size())
    for i in range(a_count):
        var v = a_h.ptr[i].cast[DType.float32]()
        a_h.ptr[i] = Scalar[dtype](Float32(-0.6) + Float32(0.5) * v)

    var yd_heap = ctx.enqueue_create_host_buffer[dtype](y_count)
    var yd_h = LayoutTensor[dtype, layout_5d, _](
        yd_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    rand[dtype](yd_h.ptr, yd_h.size())

    var y_gpu_heap = ctx.enqueue_create_host_buffer[dtype](y_count)

    var c_device = ctx.enqueue_create_buffer[dtype](c_count)
    var ent_device = ctx.enqueue_create_buffer[dtype](ent_count)
    var a_device = ctx.enqueue_create_buffer[dtype](a_count)
    var yd_device = ctx.enqueue_create_buffer[dtype](y_count)
    var y_device = ctx.enqueue_create_buffer[dtype](y_count)

    with ctx.push_context():
        ctx.enqueue_copy(c_device, c_h.ptr)
        ctx.enqueue_copy(ent_device, ent_h.ptr)
        ctx.enqueue_copy(a_device, a_h.ptr)
        ctx.enqueue_copy(yd_device, yd_h.ptr)

    var c_tt = TileTensor(
        c_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var ent_tt = TileTensor(
        ent_device, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )
    var a_tt = TileTensor(
        a_device, row_major(batch, n_chunks, n_heads, chunk_len)
    )
    var yd_tt = TileTensor(
        yd_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var y_tt = TileTensor(
        y_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )

    comptime if use_fused:
        ssd_output_recombination_fwd_gpu_fused[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            c_tt.LayoutType,
            ent_tt.LayoutType,
            a_tt.LayoutType,
            yd_tt.LayoutType,
            y_tt.LayoutType,
        ](batch, n_chunks, n_heads, c_tt, ent_tt, a_tt, yd_tt, y_tt, ctx)
    else:
        ssd_output_recombination_fwd_gpu_static[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            c_tt.LayoutType,
            ent_tt.LayoutType,
            a_tt.LayoutType,
            yd_tt.LayoutType,
            y_tt.LayoutType,
        ](batch, n_chunks, n_heads, c_tt, ent_tt, a_tt, yd_tt, y_tt, ctx)

    with ctx.push_context():
        ctx.enqueue_copy(y_gpu_heap, y_device)
    ctx.synchronize()

    # ── CPU reference: naive output recombination ────────────────────────────
    var ref_y = ctx.enqueue_create_host_buffer[dtype](y_count)
    var c_l_stride = state_dim
    var c_h_stride = chunk_len * c_l_stride
    var c_c_stride = n_heads * c_h_stride
    var c_b_stride = n_chunks * c_c_stride
    var ent_p_stride = state_dim
    var ent_h_stride = head_dim * ent_p_stride
    var ent_c_stride = n_heads * ent_h_stride
    var ent_b_stride = n_chunks * ent_c_stride
    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride
    var y_l_stride = head_dim
    var y_h_stride = chunk_len * y_l_stride
    var y_c_stride = n_heads * y_h_stride
    var y_b_stride = n_chunks * y_c_stride

    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var a_base = b * a_b_stride + c * a_c_stride + h * a_h_stride
                var c_base = b * c_b_stride + c * c_c_stride + h * c_h_stride
                var ent_base = (
                    b * ent_b_stride + c * ent_c_stride + h * ent_h_stride
                )
                var y_base = b * y_b_stride + c * y_c_stride + h * y_h_stride
                var a_cumsum = Float32(0.0)
                for l in range(chunk_len):
                    a_cumsum += a_h.ptr[a_base + l].cast[DType.float32]()
                    var decay = exp(a_cumsum)
                    for p in range(head_dim):
                        var acc = Float32(0.0)
                        for n in range(state_dim):
                            var c_val = c_h.ptr[
                                c_base + l * c_l_stride + n
                            ].cast[DType.float32]()
                            var ent_val = ent_h.ptr[
                                ent_base + p * ent_p_stride + n
                            ].cast[DType.float32]()
                            acc += c_val * ent_val
                        var yd_val = yd_h.ptr[y_base + l * y_l_stride + p].cast[
                            DType.float32
                        ]()
                        ref_y[y_base + l * y_l_stride + p] = Scalar[dtype](
                            yd_val + decay * acc
                        )

    for i in range(y_count):
        assert_almost_equal(y_gpu_heap[i], ref_y[i], rtol=rtol)


def test_ssd_output_recombination_gpu_static_golden() raises:
    """Static tensor-core path vs naive reference, gate-hitting dims."""
    var ctx = DeviceContext()
    run_ssd_output_recombination_gpu_static[
        DType.float32, chunk_len=128, head_dim=64, state_dim=128
    ](batch=1, n_chunks=2, n_heads=2, ctx=ctx)


def test_ssd_output_recombination_gpu_static_mamba2() raises:
    """Static path at the Mamba2-130m prefill profile (L=256, P=64, N=128)."""
    var ctx = DeviceContext()
    run_ssd_output_recombination_gpu_static[
        DType.float32, chunk_len=256, head_dim=64, state_dim=128
    ](batch=1, n_chunks=2, n_heads=4, ctx=ctx)


def test_ssd_output_recombination_gpu_fused_golden() raises:
    """Fused single-pass MMA path vs naive reference, gate-hitting dims."""
    var ctx = DeviceContext()
    run_ssd_output_recombination_gpu_static[
        DType.float32,
        chunk_len=128,
        head_dim=64,
        state_dim=128,
        use_fused=True,
    ](batch=1, n_chunks=2, n_heads=2, ctx=ctx)


def test_ssd_output_recombination_gpu_fused_mamba2() raises:
    """Fused path at the Mamba2-130m prefill profile (L=256, P=64, N=128)."""
    var ctx = DeviceContext()
    run_ssd_output_recombination_gpu_static[
        DType.float32,
        chunk_len=256,
        head_dim=64,
        state_dim=128,
        use_fused=True,
    ](batch=1, n_chunks=2, n_heads=4, ctx=ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
