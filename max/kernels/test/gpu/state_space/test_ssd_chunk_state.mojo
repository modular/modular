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

from std.math import exp

from max.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from std.random import rand
from state_space.ssd_chunk_state import (
    ssd_chunk_state_fwd_gpu,
    ssd_chunk_state_fwd_gpu_static,
    ssd_chunk_state_fwd_gpu_fused,
)
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index


def run_ssd_chunk_state_gpu[
    dtype: DType,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
    num_p_tiles: Int = 1,
) raises:
    """Run the SSD chunk-state GPU kernel and check it against a CPU reference.

    One block per ``(batch, chunk, head)`` slice (``block_dim == chunk_len``)
    reduces the chunk to its ``[head_dim, state_dim]`` end-state; this validates
    the result against the same naive reference used by the CPU test.
    """
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var b_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var x_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len
    var state_count = batch * n_chunks * n_heads * head_dim * state_dim

    # ── Host tensors ─────────────────────────────────────────────────────────
    var B_heap = ctx.enqueue_create_host_buffer[dtype](b_count)
    var B_h = LayoutTensor[dtype, layout_5d, _](
        B_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    rand[dtype](B_h.ptr, B_h.size())

    var X_heap = ctx.enqueue_create_host_buffer[dtype](x_count)
    var X_h = LayoutTensor[dtype, layout_5d, _](
        X_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    rand[dtype](X_h.ptr, X_h.size())

    var A_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    var A_h = LayoutTensor[dtype, layout_4d, _](
        A_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    rand[dtype](A_h.ptr, A_h.size())
    # Make A negative-ish (a real SSM decays) so exp(cumsum diffs) stays
    # bounded. `rand` fills in [0, 1); map into a small negative range.
    for i in range(a_count):
        var v = A_h.ptr[i].cast[DType.float32]()
        # Map [0, 1) -> [-0.6, -0.1].
        var scaled = Float32(-0.6) + Float32(0.5) * v
        A_h.ptr[i] = Scalar[dtype](scaled)

    var state_gpu_heap = ctx.enqueue_create_host_buffer[dtype](state_count)
    var state_gpu_h = LayoutTensor[dtype, layout_5d, _](
        state_gpu_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )

    # ── Device buffers ───────────────────────────────────────────────────────
    var B_device = ctx.enqueue_create_buffer[dtype](b_count)
    var X_device = ctx.enqueue_create_buffer[dtype](x_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var state_device = ctx.enqueue_create_buffer[dtype](state_count)

    with ctx.push_context():
        ctx.enqueue_copy(B_device, B_h.ptr)
        ctx.enqueue_copy(X_device, X_h.ptr)
        ctx.enqueue_copy(A_device, A_h.ptr)

    var B_tt = TileTensor(
        B_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var X_tt = TileTensor(
        X_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )
    var A_tt = TileTensor(
        A_device,
        row_major(batch, n_chunks, n_heads, chunk_len),
    )
    var state_tt = TileTensor(
        state_device,
        row_major(batch, n_chunks, n_heads, head_dim, state_dim),
    )

    var total_slices = batch * n_chunks * n_heads

    var compiled_func = ctx.compile_function[
        ssd_chunk_state_fwd_gpu[
            dtype,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            state_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len),
            Int32(state_dim),
            Int32(head_dim),
            B_tt,
            X_tt,
            A_tt,
            state_tt,
            grid_dim=(total_slices, num_p_tiles),
            block_dim=(chunk_len,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(state_gpu_h.ptr, state_device)
    ctx.synchronize()

    # ── CPU reference: naive reduction of the SSD chunk-state formula ────────
    var ref_heap = ctx.enqueue_create_host_buffer[dtype](state_count)

    var b_h_stride = chunk_len * state_dim
    var b_c_stride = n_heads * b_h_stride
    var b_b_stride = n_chunks * b_c_stride

    var x_h_stride = chunk_len * head_dim
    var x_c_stride = n_heads * x_h_stride
    var x_b_stride = n_chunks * x_c_stride

    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    var state_h_stride = head_dim * state_dim
    var state_c_stride = n_heads * state_h_stride
    var state_b_stride = n_chunks * state_c_stride

    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var b_base = b * b_b_stride + c * b_c_stride + h * b_h_stride
                var x_base = b * x_b_stride + c * x_c_stride + h * x_h_stride
                var a_base = b * a_b_stride + c * a_c_stride + h * a_h_stride
                var state_base = (
                    b * state_b_stride + c * state_c_stride + h * state_h_stride
                )

                var cumsum = List[Float32](length=chunk_len, fill=0.0)
                var acc = Float32(0.0)
                for l in range(chunk_len):
                    acc += A_h.ptr[a_base + l].cast[DType.float32]()
                    cumsum[l] = acc

                var cum_end = cumsum[chunk_len - 1]

                for p in range(head_dim):
                    for n in range(state_dim):
                        var s_acc = Float32(0.0)
                        for l in range(chunk_len):
                            var bv = B_h.ptr[b_base + l * state_dim + n].cast[
                                DType.float32
                            ]()
                            var xv = X_h.ptr[x_base + l * head_dim + p].cast[
                                DType.float32
                            ]()
                            var decay = exp(cum_end - cumsum[l])
                            s_acc += bv * decay * xv
                        ref_heap[state_base + p * state_dim + n] = Scalar[
                            dtype
                        ](s_acc)

    # ── Compare GPU vs CPU reference ─────────────────────────────────────────
    for i in range(state_count):
        assert_almost_equal(state_gpu_h.ptr[i], ref_heap[i], rtol=rtol)


def run_ssd_chunk_state_gpu_static[
    dtype: DType,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    fused: Bool = False,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Run a static tensor-core chunk-state entry and check it vs a CPU ref.

    With ``fused=False`` runs ``ssd_chunk_state_fwd_gpu_static`` (2-pass
    decay-transpose + batched_matmul); with ``fused=True`` runs the fused
    single-pass kernel ``ssd_chunk_state_fwd_gpu_fused``. Both take compile-time
    dims and the same argument list.
    """
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var b_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var x_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len
    var state_count = batch * n_chunks * n_heads * head_dim * state_dim

    var B_heap = ctx.enqueue_create_host_buffer[dtype](b_count)
    var B_h = LayoutTensor[dtype, layout_5d, _](
        B_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    rand[dtype](B_h.ptr, B_h.size())

    var X_heap = ctx.enqueue_create_host_buffer[dtype](x_count)
    var X_h = LayoutTensor[dtype, layout_5d, _](
        X_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    rand[dtype](X_h.ptr, X_h.size())

    var A_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    var A_h = LayoutTensor[dtype, layout_4d, _](
        A_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    rand[dtype](A_h.ptr, A_h.size())
    for i in range(a_count):
        var v = A_h.ptr[i].cast[DType.float32]()
        A_h.ptr[i] = Scalar[dtype](Float32(-0.6) + Float32(0.5) * v)

    var state_gpu_heap = ctx.enqueue_create_host_buffer[dtype](state_count)

    var B_device = ctx.enqueue_create_buffer[dtype](b_count)
    var X_device = ctx.enqueue_create_buffer[dtype](x_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var state_device = ctx.enqueue_create_buffer[dtype](state_count)

    with ctx.push_context():
        ctx.enqueue_copy(B_device, B_h.ptr)
        ctx.enqueue_copy(X_device, X_h.ptr)
        ctx.enqueue_copy(A_device, A_h.ptr)

    var B_tt = TileTensor(
        B_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var X_tt = TileTensor(
        X_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var A_tt = TileTensor(
        A_device, row_major(batch, n_chunks, n_heads, chunk_len)
    )
    var state_tt = TileTensor(
        state_device, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )

    comptime if fused:
        ssd_chunk_state_fwd_gpu_fused[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            state_tt.LayoutType,
        ](batch, n_chunks, n_heads, B_tt, X_tt, A_tt, state_tt, ctx)
    else:
        ssd_chunk_state_fwd_gpu_static[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            state_tt.LayoutType,
        ](batch, n_chunks, n_heads, B_tt, X_tt, A_tt, state_tt, ctx)

    with ctx.push_context():
        ctx.enqueue_copy(state_gpu_heap, state_device)
    ctx.synchronize()

    # ── CPU reference ────────────────────────────────────────────────────────
    var ref_heap = ctx.enqueue_create_host_buffer[dtype](state_count)
    var b_h_stride = chunk_len * state_dim
    var b_c_stride = n_heads * b_h_stride
    var b_b_stride = n_chunks * b_c_stride
    var x_h_stride = chunk_len * head_dim
    var x_c_stride = n_heads * x_h_stride
    var x_b_stride = n_chunks * x_c_stride
    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride
    var state_h_stride = head_dim * state_dim
    var state_c_stride = n_heads * state_h_stride
    var state_b_stride = n_chunks * state_c_stride

    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var b_base = b * b_b_stride + c * b_c_stride + h * b_h_stride
                var x_base = b * x_b_stride + c * x_c_stride + h * x_h_stride
                var a_base = b * a_b_stride + c * a_c_stride + h * a_h_stride
                var state_base = (
                    b * state_b_stride + c * state_c_stride + h * state_h_stride
                )
                var cumsum = List[Float32](length=chunk_len, fill=0.0)
                var acc = Float32(0.0)
                for l in range(chunk_len):
                    acc += A_h.ptr[a_base + l].cast[DType.float32]()
                    cumsum[l] = acc
                var cum_end = cumsum[chunk_len - 1]
                for p in range(head_dim):
                    for n in range(state_dim):
                        var s_acc = Float32(0.0)
                        for l in range(chunk_len):
                            var bv = B_h.ptr[b_base + l * state_dim + n].cast[
                                DType.float32
                            ]()
                            var xv = X_h.ptr[x_base + l * head_dim + p].cast[
                                DType.float32
                            ]()
                            s_acc += bv * exp(cum_end - cumsum[l]) * xv
                        ref_heap[state_base + p * state_dim + n] = Scalar[
                            dtype
                        ](s_acc)

    for i in range(state_count):
        assert_almost_equal(state_gpu_heap[i], ref_heap[i], rtol=rtol)


def test_ssd_chunk_state_gpu_golden() raises:
    """Golden-parity test on GPU against the canonical PyTorch SSD reference.

    Same FIXED tiny input as the CPU golden test (batch=1, n_chunks=1,
    n_heads=1, L=4, N=3, P=2). Pins the GPU kernel to the chunk end-state
    reduction (stage 2 of ``ssd_minimal_discrete``).
    """
    var ctx = DeviceContext()

    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 1
    comptime n_heads = 1
    comptime chunk_len = 4
    comptime state_dim = 3
    comptime head_dim = 2

    var B_vals = [
        Float32(0.74),
        1.95,
        -0.70,
        -1.30,
        -0.51,
        -0.27,
        0.25,
        0.48,
        0.45,
        -0.96,
        1.50,
        -0.31,
    ]
    var X_vals = [
        Float32(-0.23),
        -1.07,
        0.16,
        0.12,
        0.38,
        -0.12,
        0.89,
        -0.49,
    ]
    var A_vals = [Float32(-0.45), -0.18, -0.36, -0.50]
    var state_expected = [
        Float32(-0.944955),
        1.252577,
        -0.133558,
        0.106325,
        -1.533317,
        0.370175,
    ]

    var b_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var x_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len
    var state_count = batch * n_chunks * n_heads * head_dim * state_dim

    # Host buffers from fixed literals.
    var B_heap = ctx.enqueue_create_host_buffer[dtype](b_count)
    for i in range(b_count):
        B_heap[i] = Scalar[dtype](B_vals[i])
    var X_heap = ctx.enqueue_create_host_buffer[dtype](x_count)
    for i in range(x_count):
        X_heap[i] = Scalar[dtype](X_vals[i])
    var A_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    for i in range(a_count):
        A_heap[i] = Scalar[dtype](A_vals[i])
    var state_heap = ctx.enqueue_create_host_buffer[dtype](state_count)

    var B_device = ctx.enqueue_create_buffer[dtype](b_count)
    var X_device = ctx.enqueue_create_buffer[dtype](x_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var state_device = ctx.enqueue_create_buffer[dtype](state_count)

    with ctx.push_context():
        ctx.enqueue_copy(B_device, B_heap)
        ctx.enqueue_copy(X_device, X_heap)
        ctx.enqueue_copy(A_device, A_heap)

    var B_tt = TileTensor(
        B_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var X_tt = TileTensor(
        X_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )
    var A_tt = TileTensor(
        A_device,
        row_major(batch, n_chunks, n_heads, chunk_len),
    )
    var state_tt = TileTensor(
        state_device,
        row_major(batch, n_chunks, n_heads, head_dim, state_dim),
    )

    var total_slices = batch * n_chunks * n_heads

    var compiled_func = ctx.compile_function[
        ssd_chunk_state_fwd_gpu[
            dtype,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            state_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len),
            Int32(state_dim),
            Int32(head_dim),
            B_tt,
            X_tt,
            A_tt,
            state_tt,
            grid_dim=(total_slices,),
            block_dim=(chunk_len,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(state_heap, state_device)
    ctx.synchronize()

    for i in range(state_count):
        assert_almost_equal(
            state_heap[i],
            Scalar[dtype](state_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )


def test_ssd_chunk_state_gpu_basic() raises:
    """Basic single-batch, single-chunk case against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_chunk_state_gpu[DType.float32](
        batch=1,
        n_chunks=1,
        n_heads=2,
        chunk_len=8,
        state_dim=16,
        head_dim=8,
        ctx=ctx,
    )


def test_ssd_chunk_state_gpu_multi_chunk() raises:
    """Multiple chunks and heads against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_chunk_state_gpu[DType.float32](
        batch=1,
        n_chunks=3,
        n_heads=2,
        chunk_len=16,
        state_dim=16,
        head_dim=8,
        ctx=ctx,
    )


def test_ssd_chunk_state_gpu_multi_batch() raises:
    """Multiple batch elements against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_chunk_state_gpu[DType.float32](
        batch=2,
        n_chunks=2,
        n_heads=2,
        chunk_len=8,
        state_dim=8,
        head_dim=4,
        ctx=ctx,
    )


def test_ssd_chunk_state_gpu_ptiled() raises:
    """Multi-block-per-slice (grid_dim.y > 1) path, head_dim divisible by tiles.

    Splits each slice across 4 p-tiles (head_dim=16, tile height 4); checks the
    tiled result still matches the naive reference.
    """
    var ctx = DeviceContext()
    run_ssd_chunk_state_gpu[DType.float32](
        batch=1,
        n_chunks=2,
        n_heads=3,
        chunk_len=16,
        state_dim=16,
        head_dim=16,
        ctx=ctx,
        num_p_tiles=4,
    )


def test_ssd_chunk_state_gpu_ptiled_ragged() raises:
    """Multi-block-per-slice path with head_dim NOT divisible by the tile count.

    head_dim=10 over 4 p-tiles → tile height ceildiv(10,4)=3, so tiles cover
    rows [0,3) [3,6) [6,9) [9,10); the last tile is a partial row and an extra
    block would early-return. Exercises the ragged tail and bounds handling.
    """
    var ctx = DeviceContext()
    run_ssd_chunk_state_gpu[DType.float32](
        batch=1,
        n_chunks=1,
        n_heads=2,
        chunk_len=8,
        state_dim=8,
        head_dim=10,
        ctx=ctx,
        num_p_tiles=4,
    )


def test_ssd_chunk_state_gpu_static_mma() raises:
    """Tensor-core batched-matmul path at gate-satisfying dims.

    state_dim=128 (matmul N % 128 == 0), chunk_len=128 (K % 32 == 0, >= 128) so
    ssd_chunk_state_fwd_gpu_static takes the A100 multistage path. Checked vs
    the naive CPU reference.
    """
    var ctx = DeviceContext()
    run_ssd_chunk_state_gpu_static[
        DType.float32, chunk_len=128, state_dim=128, head_dim=64
    ](batch=1, n_chunks=2, n_heads=4, ctx=ctx)


def test_ssd_chunk_state_gpu_static_mma_profile() raises:
    """Tensor-core path at the Mamba2-130m prefill profile (L=256, P=64, N=128).
    """
    var ctx = DeviceContext()
    run_ssd_chunk_state_gpu_static[
        DType.float32, chunk_len=256, state_dim=128, head_dim=64
    ](batch=1, n_chunks=4, n_heads=24, ctx=ctx)


def test_ssd_chunk_state_gpu_fused_mma() raises:
    """Fused single-pass tensor-core path at gate-satisfying dims."""
    var ctx = DeviceContext()
    run_ssd_chunk_state_gpu_static[
        DType.float32,
        chunk_len=128,
        state_dim=128,
        head_dim=64,
        fused=True,
    ](batch=1, n_chunks=2, n_heads=4, ctx=ctx)


def test_ssd_chunk_state_gpu_fused_mma_profile() raises:
    """Fused path at the Mamba2-130m prefill profile (L=256, P=64, N=128)."""
    var ctx = DeviceContext()
    run_ssd_chunk_state_gpu_static[
        DType.float32,
        chunk_len=256,
        state_dim=128,
        head_dim=64,
        fused=True,
    ](batch=1, n_chunks=4, n_heads=24, ctx=ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
