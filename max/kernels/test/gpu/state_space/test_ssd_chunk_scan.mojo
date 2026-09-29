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
from state_space.ssd_chunk_scan import ssd_chunk_scan_fwd_gpu
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index


def run_ssd_chunk_scan_gpu[
    dtype: DType,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Run the SSD inter-chunk scan GPU kernel and check it against a CPU ref.

    One GPU thread per ``(batch, head, p, n)`` scalar walks the chunk axis
    sequentially; this validates the result against the same naive recurrence
    used by the CPU test.
    """
    comptime SCALAR_BLOCK_SIZE = 128
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var cs_count = batch * n_chunks * n_heads * head_dim * state_dim
    var cd_count = batch * n_chunks * n_heads
    var ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    var final_count = batch * n_heads * head_dim * state_dim

    # ── Host tensors ─────────────────────────────────────────────────────────
    var cs_heap = ctx.enqueue_create_host_buffer[dtype](cs_count)
    var cs_h = LayoutTensor[dtype, layout_5d, _](
        cs_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )
    rand[dtype](cs_h.ptr, cs_h.size())

    var cd_heap = ctx.enqueue_create_host_buffer[dtype](cd_count)
    var cd_h = LayoutTensor[dtype, layout_3d, _](
        cd_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, n_chunks, n_heads)),
    )
    rand[dtype](cd_h.ptr, cd_h.size())
    # Make chunk_decays negative (a real SSM decays). `rand` fills in [0, 1);
    # map into [-1.2, -0.3] like a real per-chunk total decay.
    for i in range(cd_count):
        var v = cd_h.ptr[i].cast[DType.float32]()
        # Map [0, 1) -> [-1.2, -0.3].
        var scaled = Float32(-1.2) + Float32(0.9) * v
        cd_h.ptr[i] = Scalar[dtype](scaled)

    var ent_gpu_heap = ctx.enqueue_create_host_buffer[dtype](ent_count)
    var ent_gpu_h = LayoutTensor[dtype, layout_5d, _](
        ent_gpu_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )
    var final_gpu_heap = ctx.enqueue_create_host_buffer[dtype](final_count)
    var final_gpu_h = LayoutTensor[dtype, layout_4d, _](
        final_gpu_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_heads, head_dim, state_dim)
        ),
    )

    # ── Device buffers ───────────────────────────────────────────────────────
    var cs_device = ctx.enqueue_create_buffer[dtype](cs_count)
    var cd_device = ctx.enqueue_create_buffer[dtype](cd_count)
    var ent_device = ctx.enqueue_create_buffer[dtype](ent_count)
    var final_device = ctx.enqueue_create_buffer[dtype](final_count)

    with ctx.push_context():
        ctx.enqueue_copy(cs_device, cs_h.ptr)
        ctx.enqueue_copy(cd_device, cd_h.ptr)

    var cs_tt = TileTensor(
        cs_device,
        row_major(batch, n_chunks, n_heads, head_dim, state_dim),
    )
    var cd_tt = TileTensor(
        cd_device,
        row_major(batch, n_chunks, n_heads),
    )
    var ent_tt = TileTensor(
        ent_device,
        row_major(batch, n_chunks, n_heads, head_dim, state_dim),
    )
    var final_tt = TileTensor(
        final_device,
        row_major(batch, n_heads, head_dim, state_dim),
    )

    var total_threads = batch * n_heads * head_dim * state_dim

    var compiled_func = ctx.compile_function[
        ssd_chunk_scan_fwd_gpu[
            dtype,
            cs_tt.LayoutType,
            cd_tt.LayoutType,
            ent_tt.LayoutType,
            final_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(head_dim),
            Int32(state_dim),
            cs_tt,
            cd_tt,
            ent_tt,
            final_tt,
            grid_dim=(ceildiv(total_threads, SCALAR_BLOCK_SIZE),),
            block_dim=(SCALAR_BLOCK_SIZE,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(ent_gpu_h.ptr, ent_device)
        ctx.enqueue_copy(final_gpu_h.ptr, final_device)
    ctx.synchronize()

    # ── CPU reference: naive sequential recurrence over chunks ───────────────
    var ref_ent = ctx.enqueue_create_host_buffer[dtype](ent_count)
    var ref_final = ctx.enqueue_create_host_buffer[dtype](final_count)

    var cs_h_stride = head_dim * state_dim
    var cs_c_stride = n_heads * cs_h_stride
    var cs_b_stride = n_chunks * cs_c_stride

    var cd_c_stride = n_heads
    var cd_b_stride = n_chunks * cd_c_stride

    var ent_h_stride = head_dim * state_dim
    var ent_c_stride = n_heads * ent_h_stride
    var ent_b_stride = n_chunks * ent_c_stride

    var final_h_stride = head_dim * state_dim
    var final_b_stride = n_heads * final_h_stride

    for b in range(batch):
        for h in range(n_heads):
            var state = List[Float32](length=head_dim * state_dim, fill=0.0)
            for c in range(n_chunks):
                var ent_base = (
                    b * ent_b_stride + c * ent_c_stride + h * ent_h_stride
                )
                for pn in range(head_dim * state_dim):
                    ref_ent[ent_base + pn] = Scalar[dtype](state[pn])

                var cd_off = b * cd_b_stride + c * cd_c_stride + h
                var decay = exp(cd_h.ptr[cd_off].cast[DType.float32]())

                var cs_base = (
                    b * cs_b_stride + c * cs_c_stride + h * cs_h_stride
                )
                for pn in range(head_dim * state_dim):
                    var cs_val = cs_h.ptr[cs_base + pn].cast[DType.float32]()
                    state[pn] = cs_val + decay * state[pn]

            var final_base = b * final_b_stride + h * final_h_stride
            for pn in range(head_dim * state_dim):
                ref_final[final_base + pn] = Scalar[dtype](state[pn])

    # ── Compare GPU vs CPU reference ─────────────────────────────────────────
    for i in range(ent_count):
        assert_almost_equal(ent_gpu_h.ptr[i], ref_ent[i], rtol=rtol)
    for i in range(final_count):
        assert_almost_equal(final_gpu_h.ptr[i], ref_final[i], rtol=rtol)


def test_ssd_chunk_scan_gpu_golden() raises:
    """Golden-parity test on GPU against the canonical PyTorch SSD reference.

    Same FIXED tiny input as the CPU golden test (batch=1, n_chunks=3,
    n_heads=1, P=2, N=3). Pins the GPU kernel to the inter-chunk recurrence
    (stage 3 of ``ssd_minimal_discrete``).
    """
    var ctx = DeviceContext()

    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 3
    comptime n_heads = 1
    comptime head_dim = 2
    comptime state_dim = 3

    var cs_vals = [
        Float32(-0.90),
        0.57,
        0.91,
        0.43,
        1.29,
        -0.17,
        -0.86,
        -0.66,
        -0.70,
        -0.10,
        -0.62,
        -0.63,
        -0.58,
        0.59,
        0.12,
        0.09,
        1.04,
        -0.18,
    ]
    var cd_vals = [Float32(-0.61), -1.03, -0.79]
    var entering_expected = [
        Float32(0.0),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.90,
        0.57,
        0.91,
        0.43,
        1.29,
        -0.17,
        -1.181306,
        -0.456506,
        -0.375124,
        0.053513,
        -0.159461,
        -0.690691,
    ]
    var final_expected = [
        Float32(-1.116130),
        0.382817,
        -0.050248,
        0.114287,
        0.967629,
        -0.493467,
    ]

    var cs_count = batch * n_chunks * n_heads * head_dim * state_dim
    var cd_count = batch * n_chunks * n_heads
    var ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    var final_count = batch * n_heads * head_dim * state_dim

    # Host buffers from fixed literals.
    var cs_heap = ctx.enqueue_create_host_buffer[dtype](cs_count)
    for i in range(cs_count):
        cs_heap[i] = Scalar[dtype](cs_vals[i])
    var cd_heap = ctx.enqueue_create_host_buffer[dtype](cd_count)
    for i in range(cd_count):
        cd_heap[i] = Scalar[dtype](cd_vals[i])
    var ent_heap = ctx.enqueue_create_host_buffer[dtype](ent_count)
    var final_heap = ctx.enqueue_create_host_buffer[dtype](final_count)

    var cs_device = ctx.enqueue_create_buffer[dtype](cs_count)
    var cd_device = ctx.enqueue_create_buffer[dtype](cd_count)
    var ent_device = ctx.enqueue_create_buffer[dtype](ent_count)
    var final_device = ctx.enqueue_create_buffer[dtype](final_count)

    with ctx.push_context():
        ctx.enqueue_copy(cs_device, cs_heap)
        ctx.enqueue_copy(cd_device, cd_heap)

    var cs_tt = TileTensor(
        cs_device,
        row_major(batch, n_chunks, n_heads, head_dim, state_dim),
    )
    var cd_tt = TileTensor(
        cd_device,
        row_major(batch, n_chunks, n_heads),
    )
    var ent_tt = TileTensor(
        ent_device,
        row_major(batch, n_chunks, n_heads, head_dim, state_dim),
    )
    var final_tt = TileTensor(
        final_device,
        row_major(batch, n_heads, head_dim, state_dim),
    )

    var total_threads = batch * n_heads * head_dim * state_dim

    var compiled_func = ctx.compile_function[
        ssd_chunk_scan_fwd_gpu[
            dtype,
            cs_tt.LayoutType,
            cd_tt.LayoutType,
            ent_tt.LayoutType,
            final_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(head_dim),
            Int32(state_dim),
            cs_tt,
            cd_tt,
            ent_tt,
            final_tt,
            grid_dim=(1,),
            block_dim=(total_threads,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(ent_heap, ent_device)
        ctx.enqueue_copy(final_heap, final_device)
    ctx.synchronize()

    for i in range(ent_count):
        assert_almost_equal(
            ent_heap[i],
            Scalar[dtype](entering_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )
    for i in range(final_count):
        assert_almost_equal(
            final_heap[i],
            Scalar[dtype](final_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )


def test_ssd_chunk_scan_gpu_basic() raises:
    """Basic single-batch case against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_chunk_scan_gpu[DType.float32](
        batch=1,
        n_chunks=4,
        n_heads=2,
        head_dim=8,
        state_dim=16,
        ctx=ctx,
    )


def test_ssd_chunk_scan_gpu_multi_batch() raises:
    """Multiple batch elements and heads against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_chunk_scan_gpu[DType.float32](
        batch=2,
        n_chunks=5,
        n_heads=3,
        head_dim=4,
        state_dim=8,
        ctx=ctx,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
