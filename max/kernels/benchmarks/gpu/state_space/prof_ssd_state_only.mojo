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
"""Single-kernel profiling driver for ``ssd_chunk_state_fwd_gpu``.

Runs ONLY the chunk-state kernel in a long loop at the Mamba2-130m prefill
profile (B=1, n_chunks=4, n_heads=24, L=256, P=64, N=128) so per-kernel ncu
rows are unambiguous and the nsys 10 kHz sampler lands on a sustained busy
window. See ``.planning/ssd-intra-optimization-notes.md`` for the GB10
profiling method (ncu single-metric + ``-c N`` is safe; ``--set full`` crashes
the box).

Iter count is read from argv[1] (default 20000) so the same binary serves both
a quick latency check and a multi-second nsys capture window.
"""

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from std.sys import argv
from layout import TileTensor, row_major

from state_space.ssd_chunk_state import ssd_chunk_state_fwd_gpu


def main() raises:
    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 4
    comptime n_heads = 24
    comptime chunk_len = 256
    comptime head_dim = 64
    comptime state_dim = 128

    var iters = 20000
    var num_p_tiles = 8  # GB10 sweet spot at this profile (see bench sweep).
    var args = argv()
    if len(args) > 1:
        iters = atol(args[1])
    if len(args) > 2:
        num_p_tiles = atol(args[2])

    var ctx = DeviceContext()

    comptime b_count = batch * n_chunks * n_heads * chunk_len * state_dim
    comptime x_count = batch * n_chunks * n_heads * chunk_len * head_dim
    comptime a_count = batch * n_chunks * n_heads * chunk_len
    comptime state_count = batch * n_chunks * n_heads * head_dim * state_dim

    var B_device = ctx.enqueue_create_buffer[dtype](b_count)
    var X_device = ctx.enqueue_create_buffer[dtype](x_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var state_device = ctx.enqueue_create_buffer[dtype](state_count)

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

    # Warmup.
    for _ in range(5):
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
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
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
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var per_iter_ms = Float64(Int(t1 - t0)) / Float64(iters) / 1.0e6
    print("ssd_chunk_state prof: ", iters, " iters, ", per_iter_ms, " ms/iter")
