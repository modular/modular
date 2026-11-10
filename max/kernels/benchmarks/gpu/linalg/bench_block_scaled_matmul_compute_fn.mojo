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
"""Times SM100 MXFP8 `block_scaled_matmul` with raw compute closures.

Covers the dense MLP GEMMs of a prefill step: a swiglu closure that reads the
gate GEMM's output (the up projection), and a closure that adds two residual
tensors (the down projection). For each shape it reports the plain GEMM, the
GEMM dispatched with the closure, and the elementwise pass alone, and prints
a hash of the closure result so two builds can be compared bit for bit. It
also counts the elements that differ from `plain bf16 GEMM -> elementwise
pass`. For M > 512, that reference rounds the closure's input once more than
the fused epilogue does, so a nonzero count is expected on data that does not
saturate the closure. At those prefill sizes, raw closures receive the
float32 scratch directly. The graph-fusion callback casts that input to the
logical output dtype before evaluating its fused expression.

Run with `--shape=<index> --m=<rows>`; `--shape=-1` runs every shape. `--sf`
scales the matmul output, to keep it out of the closure's clamp range.
"""

from std.math import ceildiv, exp
from std.sys import size_of
from std.sys.info import simd_width_of
from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel
from internal_utils import arg_parse
from internal_utils._utils import (
    InitializationType,
    init_vector_launch,
    _init_block_scaled_scales_launch,
)
from layout import Coord, Idx, TileTensor, row_major
from linalg.block_scaled_quantization import block_scaled_matmul
from linalg.fp4_utils import (
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
)
from linalg.utils import _as_compute_fn

comptime SWIGLU = 0
comptime ADD2 = 1

comptime a_type = DType.float8_e4m3fn
comptime c_type = DType.bfloat16


def _run_shape[
    N: Int, K: Int, kind: Int
](ctx: DeviceContext[], m: Int, iters: Int, sf: Float32) raises:
    comptime scales_type = MXFP8_SF_DTYPE
    comptime SF = MXFP8_SF_VECTOR_SIZE

    var a_scales_shape = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        Idx[ceildiv(K, SF * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var b_scales_shape = row_major(
        Idx[ceildiv(N, SF_MN_GROUP_SIZE)],
        Idx[ceildiv(K, SF * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var a_dev = ctx.enqueue_create_buffer[a_type](m * K)
    var b_dev = ctx.enqueue_create_buffer[a_type](N * K)
    var a_scales_dev = ctx.enqueue_create_buffer[scales_type](
        a_scales_shape.product()
    )
    var b_scales_dev = ctx.enqueue_create_buffer[scales_type](
        b_scales_shape.product()
    )
    var c_plain_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var c_closure_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var x_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var y_dev = ctx.enqueue_create_buffer[c_type](m * N)

    init_vector_launch[a_type](
        a_dev, m * K, InitializationType.uniform_distribution, ctx
    )
    init_vector_launch[a_type](
        b_dev, N * K, InitializationType.uniform_distribution, ctx
    )
    init_vector_launch[c_type](
        x_dev, m * N, InitializationType.uniform_distribution, ctx
    )
    init_vector_launch[c_type](
        y_dev, m * N, InitializationType.uniform_distribution, ctx
    )
    _init_block_scaled_scales_launch[scales_type](
        a_scales_dev, a_scales_shape.product(), ctx
    )
    _init_block_scaled_scales_launch[scales_type](
        b_scales_dev, b_scales_shape.product(), ctx
    )

    var a = TileTensor(a_dev, row_major(m, Idx[K]))
    var b = TileTensor(b_dev, row_major(Idx[N], Idx[K]))
    var a_scales = TileTensor(a_scales_dev, a_scales_shape)
    var b_scales = TileTensor(b_scales_dev, b_scales_shape)
    var c_plain = TileTensor(c_plain_dev, row_major(m, Idx[N]))
    var c_closure = TileTensor(c_closure_dev, row_major(m, Idx[N]))
    var c_ref = TileTensor(c_ref_dev, row_major(m, Idx[N]))
    var x = TileTensor(x_dev, row_major(m, Idx[N]))
    var y = TileTensor(y_dev, row_major(m, Idx[N]))

    # `swiglu` is the swigluoai MLP: gate is `x`, the matmul output is up.
    def swiglu[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[dtype, width]) {var x} -> SIMD[dtype, width]:
        comptime assert idx.rank == 2, "the matmul epilogue contract is rank 2"
        var gate = min(x.load[width=width](idx).cast[.float32](), Float32(7.0))
        var up = val.cast[.float32]().clamp(-7.0, 7.0)
        var sig = 1.0 / (1.0 + exp(-1.702 * gate))
        return (gate * sig * (up + 1.0)).cast[dtype]()

    # `add2` is the shared-expert down closure: routed + shared + residual.
    def add2[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[dtype, width]) {var x, var y} -> SIMD[dtype, width]:
        comptime assert idx.rank == 2, "the matmul epilogue contract is rank 2"
        var sum = val.cast[.float32]()
        sum += x.load[width=width](idx).cast[.float32]()
        sum += y.load[width=width](idx).cast[.float32]()
        return sum.cast[dtype]()

    def plain(c: DeviceContext[]) raises {imm}:
        block_scaled_matmul[
            SF_VECTOR_SIZE=SF,
            transpose_b=True,
            pdl_level=PDLLevel.OFF,
        ](c_plain, a, b, a_scales, b_scales, sf, c)

    def closure(c: DeviceContext[]) raises {imm}:
        comptime if kind == SWIGLU:
            block_scaled_matmul[
                SF_VECTOR_SIZE=SF,
                transpose_b=True,
                has_compute_fn=True,
                pdl_level=PDLLevel.OFF,
            ](
                c_closure,
                a,
                b,
                a_scales,
                b_scales,
                sf,
                c,
                compute_fn=_as_compute_fn(swiglu),
            )
        else:
            block_scaled_matmul[
                SF_VECTOR_SIZE=SF,
                transpose_b=True,
                has_compute_fn=True,
                pdl_level=PDLLevel.OFF,
            ](
                c_closure,
                a,
                b,
                a_scales,
                b_scales,
                sf,
                c,
                compute_fn=_as_compute_fn(add2),
            )

    comptime simd_size = simd_width_of[c_type, target=ctx.target]()

    # Reads `c_plain` and writes `c_ref`, so repeated launches are idempotent.
    def elementwise_pass(c: DeviceContext[]) raises {imm}:
        def apply[
            width: Int, alignment: Int = 1
        ](idx: Coord) {var c_plain, var c_ref, var swiglu, var add2}:
            var v = c_plain.load[width=width](idx)
            comptime if kind == SWIGLU:
                c_ref.store[alignment=alignment * size_of[c_type]()](
                    idx, swiglu[c_type, width, alignment=alignment](idx, v)
                )
            else:
                c_ref.store[alignment=alignment * size_of[c_type]()](
                    idx, add2[c_type, width, alignment=alignment](idx, v)
                )

        elementwise[simd_size](apply, (m, N), c)

    plain(ctx)
    closure(ctx)
    elementwise_pass(ctx)
    ctx.synchronize()

    var plain_ns = ctx.execution_time(plain, iters)
    var closure_ns = ctx.execution_time(closure, iters)
    var ew_ns = ctx.execution_time(elementwise_pass, iters)

    var host_closure = ctx.enqueue_create_host_buffer[c_type](m * N)
    var host_ref = ctx.enqueue_create_host_buffer[c_type](m * N)
    var host_plain = ctx.enqueue_create_host_buffer[c_type](m * N)
    ctx.enqueue_copy(host_closure, c_closure_dev)
    ctx.enqueue_copy(host_ref, c_ref_dev)
    ctx.enqueue_copy(host_plain, c_plain_dev)
    ctx.synchronize()
    var mismatches = 0
    var hash = UInt64(14695981039346656037)
    var max_abs_diff = Float32(0)
    var sum_abs_out = Float64(0)
    var sum_abs_plain = Float64(0)
    for i in range(m * N):
        var got = host_closure[i]
        var want = host_ref[i]
        if got.to_bits() != want.to_bits():
            mismatches += 1
        var diff = abs(got.cast[.float32]() - want.cast[.float32]())
        max_abs_diff = max(max_abs_diff, diff)
        sum_abs_out += Float64(abs(want.cast[.float32]()))
        sum_abs_plain += Float64(abs(host_plain[i].cast[.float32]()))
        hash = (hash ^ UInt64(got.to_bits())) * UInt64(1099511628211)

    print(
        t"RESULT kind={'swiglu' if kind == SWIGLU else 'add2'} M={m} N={N}"
        t" K={K} plain_us={Float64(plain_ns) / Float64(iters) / 1000}"
        t" closure_us={Float64(closure_ns) / Float64(iters) / 1000}"
        t" elementwise_us={Float64(ew_ns) / Float64(iters) / 1000}"
        t" vs_bf16_pass_mismatches={mismatches}"
        t" vs_bf16_pass_max_abs_diff={max_abs_diff}"
        t" mean_abs_out={sum_abs_out / Float64(m * N)}"
        t" mean_abs_plain={sum_abs_plain / Float64(m * N)} hash={hash}"
    )


def main() raises:
    var shape = arg_parse("shape", -1)
    var m = arg_parse("m", 16384)
    var iters = arg_parse("iters", 20)
    var sf = Float32(arg_parse("sf", 1.0))
    with DeviceContext() as ctx:
        # Shared expert up (swiglu) and down (adds), then the dense MLP's.
        if shape == -1 or shape == 0:
            _run_shape[3072, 6144, SWIGLU](ctx, m, iters, sf)
        if shape == -1 or shape == 1:
            _run_shape[6144, 3072, ADD2](ctx, m, iters, sf)
        if shape == -1 or shape == 2:
            _run_shape[12288, 6144, SWIGLU](ctx, m, iters, sf)
        if shape == -1 or shape == 3:
            _run_shape[6144, 12288, ADD2](ctx, m, iters, sf)
