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
"""Router-gate GEMV microbenchmark for SM100 (B200).

Times the MiniMax-M3 router gate (`[M, 6144] bf16 x [128, 6144] fp32^T`) over
`NUM_LAYERS` distinct weight copies, so each launch reads a weight the previous
launches evicted, as the MoE layers of a decode step do. Arms:

- `cast+matmul`: the standard path, a bf16->fp32 cast then the fp32 matmul
  dispatch.
- `fused`: `router_gate_mixed_gemv`, one launch.
- `copy`: the operand floor, a plain read of the same weight bytes.

The host launch rate bounds a back-to-back sequence of tiny kernels, so the
printed host time is only a screen. Read kernel durations from an nsys trace:
a one-block `marker` kernel precedes each arm so the trace splits per arm.
"""

from std.math import ceildiv
from std.random import randn, random_float64, seed
from std.sys import get_defined_int
from std.time import monotonic

from layout import Idx, TileTensor, row_major
from linalg.gemv import router_gate_mixed_gemv
from linalg.matmul import matmul
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

comptime N = 128
comptime K = 6144
comptime NUM_LAYERS = get_defined_int["NUM_LAYERS", 60]()
comptime REPS = get_defined_int["REPS", 20]()


def marker(x: MutPointer[Scalar[DType.float32], MutAnyOrigin]):
    x.unsafe_store(0, Float32(0))


def cast_bf16_to_f32(
    src: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    n: Int32,
):
    var i = (block_idx.x * block_dim.x + thread_idx.x) * 8
    if i < Int(n):
        dst.unsafe_store(i, src.unsafe_load[width=8](i).cast[DType.float32]())


def copy_f32(
    src: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    dst: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    n: Int32,
):
    var i = (block_idx.x * block_dim.x + thread_idx.x) * 4
    if i < Int(n):
        dst.unsafe_store(i, src.unsafe_load[width=4](i))


struct Operands(Movable):
    var weights: List[DeviceBuffer[DType.float32]]
    var a_bf16: DeviceBuffer[DType.bfloat16]
    var a_f32: DeviceBuffer[DType.float32]
    var c: DeviceBuffer[DType.float32]
    var scratch: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, m: Int) raises:
        self.weights = List[DeviceBuffer[DType.float32]]()
        for _ in range(NUM_LAYERS):
            self.weights.append(ctx.enqueue_create_buffer[DType.float32](N * K))
        self.a_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](m * K)
        self.a_f32 = ctx.enqueue_create_buffer[DType.float32](m * K)
        self.c = ctx.enqueue_create_buffer[DType.float32](m * N)
        self.scratch = ctx.enqueue_create_buffer[DType.float32](N * K)

        seed(7)
        var a_host = ctx.enqueue_create_host_buffer[DType.bfloat16](m * K)
        for i in range(m * K):
            a_host[i] = random_float64(min=-1.0, max=1.0).cast[DType.bfloat16]()
        ctx.enqueue_copy(self.a_bf16, a_host)
        var w_host = ctx.enqueue_create_host_buffer[DType.float32](N * K)
        for layer in range(NUM_LAYERS):
            randn(w_host.unsafe_ptr(), N * K)
            ctx.enqueue_copy(self.weights[layer], w_host)
        ctx.synchronize()


def time_launches[
    LaunchFn: def(Int) raises -> None
](
    ctx: DeviceContext,
    marker_buf: DeviceBuffer[DType.float32],
    label: String,
    launch: LaunchFn,
) raises:
    ctx.enqueue_function[marker](
        marker_buf.unsafe_ptr(), grid_dim=1, block_dim=1
    )
    for layer in range(NUM_LAYERS):
        launch(layer)
    ctx.synchronize()
    var start = monotonic()
    for _ in range(REPS):
        for layer in range(NUM_LAYERS):
            launch(layer)
    ctx.synchronize()
    var us = Float64(monotonic() - start) / 1e3 / Float64(NUM_LAYERS * REPS)
    print(label, us, "us (host, per launch)")


def run[m: Int](ctx: DeviceContext) raises:
    var ops = Operands(ctx, m)
    var a_bf16_tt = TileTensor(ops.a_bf16, row_major(m, Idx[K])).as_imm()
    var a_f32_tt = TileTensor(ops.a_f32, row_major(m, Idx[K])).as_imm()
    var c_tt = TileTensor(ops.c, row_major(m, Idx[N]))
    var marker_buf = ctx.enqueue_create_buffer[DType.float32](1)

    def cast_matmul(layer: Int) raises {imm}:
        ctx.enqueue_function[cast_bf16_to_f32](
            ops.a_bf16.unsafe_ptr(),
            ops.a_f32.unsafe_ptr(),
            Int32(m * K),
            grid_dim=ceildiv(m * K, 8 * 256),
            block_dim=256,
        )
        var b_tt = TileTensor(
            ops.weights[layer], row_major(Idx[N], Idx[K])
        ).as_imm()
        matmul[transpose_b=True, target="gpu"](c_tt, a_f32_tt, b_tt, ctx)

    def fused(layer: Int) raises {imm}:
        var b_tt = TileTensor(
            ops.weights[layer], row_major(Idx[N], Idx[K])
        ).as_imm()
        router_gate_mixed_gemv[N](c_tt, a_bf16_tt, b_tt, m, N, K, ctx)

    def copy(layer: Int) raises {imm}:
        ctx.enqueue_function[copy_f32](
            ops.weights[layer].unsafe_ptr(),
            ops.scratch.unsafe_ptr(),
            Int32(N * K),
            grid_dim=ceildiv(N * K, 4 * 256),
            block_dim=256,
        )

    time_launches(
        ctx, marker_buf, "m=" + String(m) + " cast+matmul", cast_matmul
    )
    time_launches(ctx, marker_buf, "m=" + String(m) + " fused      ", fused)
    time_launches(ctx, marker_buf, "m=" + String(m) + " copy floor ", copy)


def main() raises:
    with DeviceContext() as ctx:
        comptime for m in [8, 16, 24, 32, 36, 48, 64, 96, 128]:
            run[m](ctx)
