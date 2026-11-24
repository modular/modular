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
"""Numerics for `moe_exit_act_concat`, the K3 MoE exit's glue kernel.

The K3 latent MoE's exit reads `[normed latent | shared activation]` as one
K axis. `moe_exit_act_concat` writes both halves of that destination in one
launch: the latent half as a copy, the shared half as the shared expert's
SiTU-and-mul of its fused `[gate ; up]` projection. The latent is copied bit
for bit out of a wider row (the merged entry GEMM writes it leading other
columns, so a stride-`wide_cols` view is what production passes), and the
activation matches the graph's SiTU chain to the destination dtype.
"""

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from layout._fillers import random
from nn.activations import sigmoid
from nn.moe import moe_exit_act_concat
from std.math import tanh
from std.memory import bitcast
from std.testing import assert_almost_equal, assert_equal


def _check[
    tokens: Int,
    latent_cols: Int,
    half_cols: Int,
    simd_width: Int,
](
    wide_cols: Int,
    beta: Float32,
    linear_beta: Float32,
    ctx: DeviceContext,
) raises:
    """Runs the kernel at one shape and checks both halves of its output.

    `wide_cols` is the latent input's row width; the kernel reads the leading
    `latent_cols` of it, the layout production hands it when the merged entry
    GEMM writes the latent beside the shared expert's columns.

    This is an INTERNAL consistency check: the latent half must be a copy, and
    the activation half must be the same launch's float32 value rounded once,
    so the two destinations agree. It is deliberately not a reference for the
    graph -- comparing against host scalar `tanh`/`sigmoid` is wrong (they
    differ from the device's vector ones by an architecture-dependent step,
    which is what broke this test on B200), and comparing against this
    kernel's own output is circular. The graph-path equivalence is pinned by
    `test_moe_exit_act_concat_op_gpu.py`, which runs the registered op beside
    the unfused `kimi_situ_and_mul` + concat chain on the device.
    """
    comptime total_cols = latent_cols + half_cols
    comptime gate_up_cols = 2 * half_cols

    var latent_host = ctx.enqueue_create_host_buffer[.bfloat16](
        tokens * wide_cols
    )
    var gate_up_host = ctx.enqueue_create_host_buffer[.bfloat16](
        tokens * gate_up_cols
    )
    random(
        TileTensor(latent_host, row_major(tokens, wide_cols)), min=-1.0, max=1.0
    )
    random(
        TileTensor(gate_up_host, row_major(tokens, gate_up_cols)),
        min=-1.0,
        max=1.0,
    )
    # The same values as float32: a bf16 value is exact there, so both
    # launches see identical inputs and only the destination's rounding
    # differs.
    var latent_f32_host = ctx.enqueue_create_host_buffer[.float32](
        tokens * wide_cols
    )
    var gate_up_f32_host = ctx.enqueue_create_host_buffer[.float32](
        tokens * gate_up_cols
    )
    for i in range(tokens * wide_cols):
        latent_f32_host[i] = latent_host[i].cast[.float32]()
    for i in range(tokens * gate_up_cols):
        gate_up_f32_host[i] = gate_up_host[i].cast[.float32]()

    var latent_dev = ctx.enqueue_create_buffer[.bfloat16](tokens * wide_cols)
    var gate_up_dev = ctx.enqueue_create_buffer[.bfloat16](
        tokens * gate_up_cols
    )
    var latent_f32_dev = ctx.enqueue_create_buffer[.float32](tokens * wide_cols)
    var gate_up_f32_dev = ctx.enqueue_create_buffer[.float32](
        tokens * gate_up_cols
    )
    ctx.enqueue_copy(latent_dev, latent_host)
    ctx.enqueue_copy(gate_up_dev, gate_up_host)
    ctx.enqueue_copy(latent_f32_dev, latent_f32_host)
    ctx.enqueue_copy(gate_up_f32_dev, gate_up_f32_host)

    # The latent is the leading block of a wider row, so this view carries the
    # parent's row stride rather than a packed `latent_cols`.
    var latent = TileTensor(latent_dev, row_major(tokens, wide_cols)).tile[
        tokens, latent_cols
    ](0, 0)
    var gate_up = TileTensor(gate_up_dev, row_major(tokens, gate_up_cols))
    var latent_f32 = TileTensor(
        latent_f32_dev, row_major(tokens, wide_cols)
    ).tile[tokens, latent_cols](0, 0)
    var gate_up_f32 = TileTensor(
        gate_up_f32_dev, row_major(tokens, gate_up_cols)
    )

    @inline(.always)
    def latent_fn[
        width: Int
    ](coords: Coord) {var latent} -> SIMD[.bfloat16, width]:
        return rebind[SIMD[.bfloat16, width]](latent.load[width=width](coords))

    @inline(.always)
    def gate_up_fn[
        width: Int
    ](coords: Coord) {var gate_up} -> SIMD[.bfloat16, width]:
        return rebind[SIMD[.bfloat16, width]](gate_up.load[width=width](coords))

    @inline(.always)
    def latent_f32_fn[
        width: Int
    ](coords: Coord) {var latent_f32} -> SIMD[.float32, width]:
        return rebind[SIMD[.float32, width]](
            latent_f32.load[width=width](coords)
        )

    @inline(.always)
    def gate_up_f32_fn[
        width: Int
    ](coords: Coord) {var gate_up_f32} -> SIMD[.float32, width]:
        return rebind[SIMD[.float32, width]](
            gate_up_f32.load[width=width](coords)
        )

    var output_dev = ctx.enqueue_create_buffer[.bfloat16](tokens * total_cols)
    moe_exit_act_concat[dtype=.bfloat16, simd_width=simd_width](
        TileTensor(output_dev, row_major(tokens, total_cols)),
        latent_fn,
        gate_up_fn,
        latent_cols,
        half_cols,
        beta,
        linear_beta,
        ctx,
    )
    var wide_dev = ctx.enqueue_create_buffer[.float32](tokens * total_cols)
    moe_exit_act_concat[dtype=.float32, simd_width=simd_width](
        TileTensor(wide_dev, row_major(tokens, total_cols)),
        latent_f32_fn,
        gate_up_f32_fn,
        latent_cols,
        half_cols,
        beta,
        linear_beta,
        ctx,
    )

    var output_host = ctx.enqueue_create_host_buffer[.bfloat16](
        tokens * total_cols
    )
    var wide_host = ctx.enqueue_create_host_buffer[.float32](
        tokens * total_cols
    )
    ctx.enqueue_copy(output_host, output_dev)
    ctx.enqueue_copy(wide_host, wide_dev)
    ctx.synchronize()

    var out = TileTensor(output_host, row_major(tokens, total_cols))
    var wide = TileTensor(wide_host, row_major(tokens, total_cols))
    var latent_rows = TileTensor(latent_host, row_major(tokens, wide_cols))
    for t in range(tokens):
        # The latent half is a copy, so the bytes must match.
        for j in range(latent_cols):
            assert_equal(out[t, j], latent_rows[t, j])
        # The activated half is that same launch's float32 value, rounded
        # once: the two destinations of one kernel agree.
        for j in range(half_cols):
            assert_equal(
                bitcast[.uint16](out[t, latent_cols + j]),
                bitcast[.uint16](wide[t, latent_cols + j].cast[.bfloat16]()),
                String(
                    t"the activation at row {t} column {j} is not the device"
                    t" fp32 value rounded once"
                ),
            )


def _check_f32_destination[
    tokens: Int, latent_cols: Int, half_cols: Int, simd_width: Int
](
    wide_cols: Int,
    beta: Float32,
    linear_beta: Float32,
    ctx: DeviceContext,
) raises:
    """A float32 destination rounds to the caller's dtype, checked by value.

    Every served path writes the model dtype, so this shape exists only to
    cover the kernel's other destination. Its chain is compared as a value
    against the host's scalar math, which is exact enough for a tolerance but
    NOT a byte reference: the device's vector transcendentals differ from the
    scalar ones by a rounding step.
    """
    comptime total_cols = latent_cols + half_cols
    comptime gate_up_cols = 2 * half_cols

    var latent_host = ctx.enqueue_create_host_buffer[.float32](
        tokens * wide_cols
    )
    var gate_up_host = ctx.enqueue_create_host_buffer[.float32](
        tokens * gate_up_cols
    )
    random(
        TileTensor(latent_host, row_major(tokens, wide_cols)), min=-1.0, max=1.0
    )
    random(
        TileTensor(gate_up_host, row_major(tokens, gate_up_cols)),
        min=-1.0,
        max=1.0,
    )
    var latent_dev = ctx.enqueue_create_buffer[.float32](tokens * wide_cols)
    var gate_up_dev = ctx.enqueue_create_buffer[.float32](tokens * gate_up_cols)
    ctx.enqueue_copy(latent_dev, latent_host)
    ctx.enqueue_copy(gate_up_dev, gate_up_host)

    var latent = TileTensor(latent_dev, row_major(tokens, wide_cols)).tile[
        tokens, latent_cols
    ](0, 0)
    var gate_up = TileTensor(gate_up_dev, row_major(tokens, gate_up_cols))

    @inline(.always)
    def latent_fn[
        width: Int
    ](coords: Coord) {var latent} -> SIMD[.float32, width]:
        return rebind[SIMD[.float32, width]](latent.load[width=width](coords))

    @inline(.always)
    def gate_up_fn[
        width: Int
    ](coords: Coord) {var gate_up} -> SIMD[.float32, width]:
        return rebind[SIMD[.float32, width]](gate_up.load[width=width](coords))

    var output_dev = ctx.enqueue_create_buffer[.float32](tokens * total_cols)
    moe_exit_act_concat[dtype=.float32, simd_width=simd_width](
        TileTensor(output_dev, row_major(tokens, total_cols)),
        latent_fn,
        gate_up_fn,
        latent_cols,
        half_cols,
        beta,
        linear_beta,
        ctx,
    )
    var output_host = ctx.enqueue_create_host_buffer[.float32](
        tokens * total_cols
    )
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    var out = TileTensor(output_host, row_major(tokens, total_cols))
    var latent_rows = TileTensor(latent_host, row_major(tokens, wide_cols))
    var gate_up_rows = TileTensor(gate_up_host, row_major(tokens, gate_up_cols))
    for t in range(tokens):
        for j in range(latent_cols):
            assert_equal(out[t, j], latent_rows[t, j])
        for j in range(half_cols):
            var gate = gate_up_rows[t, j]
            var up = gate_up_rows[t, j + half_cols]
            var situ = beta * tanh(gate / beta)
            situ = situ * sigmoid(gate)
            if linear_beta != 0.0:
                up = linear_beta * tanh(up / linear_beta)
            assert_almost_equal(
                out[t, latent_cols + j], situ * up, atol=2e-2, rtol=2e-2
            )


def main() raises:
    with DeviceContext() as ctx:
        # K3 decode at bs=4: a 448-wide latent, a 768-wide shared half, the
        # latent leading 64 columns that belong to someone else.
        _check[32, 448, 768, 8](
            wide_cols=512,
            beta=1.5,
            linear_beta=1.5,
            ctx=ctx,
        )
        # `linear_beta` off: the up half passes through unchanged.
        _check[32, 448, 768, 8](
            wide_cols=512,
            beta=1.5,
            linear_beta=0.0,
            ctx=ctx,
        )
        # Widths that are not multiples of the 8-element vector: 20 and 13,
        # so the wrapper selects `simd_width = 1` and the kernel runs one
        # element per thread, with the latent still leading a wider row.
        _check[1, 20, 13, 1](
            wide_cols=27,
            beta=1.5,
            linear_beta=1.5,
            ctx=ctx,
        )
        # A non-bfloat16 destination, so the rounding is the caller's dtype.
        _check_f32_destination[8, 64, 32, 8](
            wide_cols=64,
            beta=1.5,
            linear_beta=1.5,
            ctx=ctx,
        )
