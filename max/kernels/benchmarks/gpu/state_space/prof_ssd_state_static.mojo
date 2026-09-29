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
"""Single-path profiling driver for the static tensor-core chunk-state entry.

Runs ssd_chunk_state_fwd_gpu_static (decay-transpose kernel + batched_matmul)
in a long loop at the Mamba2-130m prefill profile so ncu rows for the two
kernels (the decay-transpose and the matmul) are unambiguous. argv[1] = iters.
"""

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from std.sys import argv
from layout import TileTensor, row_major

from state_space.ssd_chunk_state import ssd_chunk_state_fwd_gpu_static


def main() raises:
    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 4
    comptime n_heads = 24
    comptime chunk_len = 256
    comptime head_dim = 64
    comptime state_dim = 128

    var iters = 2000
    var args = argv()
    if len(args) > 1:
        iters = atol(args[1])

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

    for _ in range(5):
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
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
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
    ctx.synchronize()
    var t1 = perf_counter_ns()

    print(
        "ssd_chunk_state static prof: ",
        iters,
        " iters, ",
        Float64(Int(t1 - t0)) / Float64(iters) / 1.0e6,
        " ms/iter",
    )
