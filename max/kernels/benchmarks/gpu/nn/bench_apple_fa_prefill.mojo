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
"""TFLOPS benchmark for the Apple M5 MMA flash-attention PREFILL kernels.

Ported from `bench_mha_prefill_v2.mojo` (the AMD prefill bench) to the Apple
path. Apple GPU (Metal 4, `compute_capability == 5`) only. `main` runs:

  1. `fa_prefill_apple` vs the `mha_gpu_naive` fallback (`_bench_shape`, causal)
     at small seqs -- the design's "beat naive" bar (naive is O(seq^2), so it is
     capped short).
  2. `fa_prefill_apple` throughput (`_bench_prefill`) across the seq range for
     causal AND NullMask, b1, d=128.

FLOP count mirrors the AMD bench: `2 * B * H * seq * num_keys * depth`. For the
causal cases the tile-skip means the reported number is EFFECTIVE throughput
(~2x the actual MMA rate); NullMask processes the full square so its number is the
true MMA rate. The fa/fa comparison at the SAME FLOP count is apples-to-apples.
Operand setup mirrors `test/gpu/nn/test_apple_fa_prefill.mojo` (the known-correct
invocation). Q/K/V/O are oversized via `CacheBustingBuffer` and offset per
iteration to defeat L2 reuse, so the absolute GFLOPS are realistic (not
cache-inflated at small seq).

Run:
  mojo max/kernels/benchmarks/gpu/nn/bench_apple_fa_prefill.mojo
"""

from std.collections import OptionalReg
from std.sys import get_defined_bool, get_defined_int

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext
from std.math import sqrt

from internal_utils import CacheBustingBuffer
from internal_utils._utils import InitializationType
from layout import (
    UNKNOWN_VALUE,
    Idx,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)

from nn.attention.mha_mask import CausalMask, NullMask, MHAMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.apple.fa_prefill import fa_prefill_apple
from nn.attention.gpu.mha import mha_gpu_naive


def _mask_label[mask_t: MHAMask]() -> String:
    comptime if mask_t == CausalMask:
        return "causal"
    else:
        return "null"


# ===-------------------------------------------------------------------=== #
# Base grid: `fa_prefill_apple` vs `mha_gpu_naive` at one (causal) shape.
# ===-------------------------------------------------------------------=== #
def _bench_shape[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    kv_heads: Int,
    naive: Bool,
](mut m: Bench, batch: Int, seq: Int, ctx: DeviceContext) raises:
    comptime group = num_heads // kv_heads
    var scale = Float32(1) / sqrt(Float32(depth))
    var num_keys = seq

    var q_n = batch * num_heads * seq * depth
    var kv_n = batch * kv_heads * num_keys * depth
    var o_n = q_n

    # Cache busting: oversize the Q/K/V/O allocations and offset per iteration
    # to defeat L2 reuse (otherwise small-seq absolutes are cache-inflated).
    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](q_n, simd_size, ctx)
    var cb_k = CacheBustingBuffer[qkv_type](kv_n, simd_size, ctx)
    var cb_v = CacheBustingBuffer[qkv_type](kv_n, simd_size, ctx)
    # Output is BF16 (qkv_type) — the Apple kernel writes bf16, not fp32.
    var cb_o = CacheBustingBuffer[qkv_type](o_n, simd_size, ctx)

    cb_q.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_k.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_v.init_on_device(InitializationType.uniform_distribution, ctx)

    # `valid_length` is `[batch+1]` and is not read on the dense path, so it
    # stays a tiny fixed buffer (no cache-busting needed).
    var vl_d = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    var vl_ptr = vl_d.unsafe_ptr()

    @parameter
    @always_inline
    @__copy_capture(cb_q, cb_k, cb_v, cb_o, vl_ptr)
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def _launch(ctx: DeviceContext, iteration: Int) raises:
            # Inputs: immutable views into the per-iteration cache-bust window.
            var q_ptr = (
                cb_q.offset_ptr(iteration)
                .bitcast[Scalar[qkv_type]]()
                .as_immutable()
                .as_unsafe_any_origin()
            )
            var k_ptr = (
                cb_k.offset_ptr(iteration)
                .bitcast[Scalar[qkv_type]]()
                .as_immutable()
                .as_unsafe_any_origin()
            )
            var v_ptr = (
                cb_v.offset_ptr(iteration)
                .bitcast[Scalar[qkv_type]]()
                .as_immutable()
                .as_unsafe_any_origin()
            )
            # Output stays MUTABLE: the kernel needs a mutable output view, and
            # a copy-captured DeviceBuffer would yield an immutable one (real
            # bug hit earlier). Pointer carries mutability in the type.
            var o_ptr = cb_o.offset_ptr(iteration).bitcast[Scalar[qkv_type]]()
            var q_t = TileTensor(
                q_ptr, row_major(batch, seq, Idx[num_heads], Idx[depth])
            )
            var k_t = TileTensor(
                k_ptr, row_major(batch, num_keys, Idx[kv_heads], Idx[depth])
            )
            var v_t = TileTensor(
                v_ptr, row_major(batch, num_keys, Idx[kv_heads], Idx[depth])
            )
            var o_t = TileTensor(
                o_ptr, row_major(batch, seq, Idx[num_heads], Idx[depth])
            )
            var vl_t = TileTensor(vl_ptr, row_major(batch + 1))
            var k_op = LayoutTensorMHAOperand(k_t)
            var v_op = LayoutTensorMHAOperand(v_t)
            comptime SinkOpt = OptionalReg[
                LayoutTensor[
                    qkv_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
                ]
            ]
            var sink_opt = SinkOpt(None)

            comptime if naive:
                mha_gpu_naive[
                    ragged=False,
                    _use_valid_length=False,
                    _is_cache_length_accurate=True,
                ](
                    q_t.to_layout_tensor(),
                    k_op,
                    v_op,
                    CausalMask(),
                    o_t.to_layout_tensor(),
                    vl_t.to_layout_tensor(),
                    scale,
                    batch,
                    seq,
                    num_keys,
                    num_heads,
                    depth,
                    group,
                    ctx,
                    sink_opt,
                )
            else:
                fa_prefill_apple[
                    ragged=False,
                    _use_valid_length=False,
                    _is_cache_length_accurate=True,
                ](
                    q_t.to_layout_tensor(),
                    k_op,
                    v_op,
                    CausalMask(),
                    o_t.to_layout_tensor(),
                    vl_t.to_layout_tensor(),
                    scale,
                    batch,
                    seq,
                    num_keys,
                    num_heads,
                    depth,
                    group,
                    ctx,
                    sink_opt,
                )

        b.iter_custom[_launch](ctx)

    def compute_flops() {read} -> Int:
        return 2 * batch * num_heads * seq * num_keys * depth

    comptime label = "mha_gpu_naive" if naive else "fa_prefill_apple"
    m.bench_function[bench_func](
        BenchId(
            label,
            input_id=String(
                "b=",
                batch,
                "/seq=",
                seq,
                "/h=",
                num_heads,
                "/kv=",
                kv_heads,
                "/d=",
                depth,
            ),
        ),
        [ThroughputMeasure(BenchMetric.flops, compute_flops())],
    )
    ctx.synchronize()
    _ = cb_q
    _ = cb_k
    _ = cb_v
    _ = cb_o
    _ = vl_d^


# ===-------------------------------------------------------------------=== #
# Crossover sweep: no-SMEM baseline at a shape/mask.
# ===-------------------------------------------------------------------=== #
def _bench_prefill[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    kv_heads: Int,
    mask_t: MHAMask,
](mut m: Bench, mask: mask_t, batch: Int, seq: Int, ctx: DeviceContext) raises:
    comptime group = num_heads // kv_heads
    var scale = Float32(1) / sqrt(Float32(depth))
    var num_keys = seq

    var q_n = batch * num_heads * seq * depth
    var kv_n = batch * kv_heads * num_keys * depth

    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](q_n, simd_size, ctx)
    var cb_k = CacheBustingBuffer[qkv_type](kv_n, simd_size, ctx)
    var cb_v = CacheBustingBuffer[qkv_type](kv_n, simd_size, ctx)
    var cb_o = CacheBustingBuffer[qkv_type](q_n, simd_size, ctx)

    cb_q.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_k.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_v.init_on_device(InitializationType.uniform_distribution, ctx)

    var vl_d = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    var vl_ptr = vl_d.unsafe_ptr()

    @parameter
    @always_inline
    @__copy_capture(cb_q, cb_k, cb_v, cb_o, vl_ptr)
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def _launch(ctx: DeviceContext, iteration: Int) raises:
            var q_ptr = (
                cb_q.offset_ptr(iteration)
                .bitcast[Scalar[qkv_type]]()
                .as_immutable()
                .as_unsafe_any_origin()
            )
            var k_ptr = (
                cb_k.offset_ptr(iteration)
                .bitcast[Scalar[qkv_type]]()
                .as_immutable()
                .as_unsafe_any_origin()
            )
            var v_ptr = (
                cb_v.offset_ptr(iteration)
                .bitcast[Scalar[qkv_type]]()
                .as_immutable()
                .as_unsafe_any_origin()
            )
            var o_ptr = cb_o.offset_ptr(iteration).bitcast[Scalar[qkv_type]]()
            var q_t = TileTensor(
                q_ptr, row_major(batch, seq, Idx[num_heads], Idx[depth])
            )
            var k_t = TileTensor(
                k_ptr, row_major(batch, num_keys, Idx[kv_heads], Idx[depth])
            )
            var v_t = TileTensor(
                v_ptr, row_major(batch, num_keys, Idx[kv_heads], Idx[depth])
            )
            var o_t = TileTensor(
                o_ptr, row_major(batch, seq, Idx[num_heads], Idx[depth])
            )
            var vl_t = TileTensor(vl_ptr, row_major(batch + 1))
            var k_op = LayoutTensorMHAOperand(k_t)
            var v_op = LayoutTensorMHAOperand(v_t)
            comptime SinkOpt = OptionalReg[
                LayoutTensor[
                    qkv_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
                ]
            ]
            var sink_opt = SinkOpt(None)

            fa_prefill_apple[
                ragged=False,
                _use_valid_length=False,
                _is_cache_length_accurate=True,
            ](
                q_t.to_layout_tensor(),
                k_op,
                v_op,
                mask,
                o_t.to_layout_tensor(),
                vl_t.to_layout_tensor(),
                scale,
                batch,
                seq,
                num_keys,
                num_heads,
                depth,
                group,
                ctx,
                sink_opt,
            )

        b.iter_custom[_launch](ctx)

    def compute_flops() {read} -> Int:
        return 2 * batch * num_heads * seq * num_keys * depth

    m.bench_function[bench_func](
        BenchId(
            "fa_prefill_apple",
            input_id=String(_mask_label[mask_t](), "/b=", batch, "/seq=", seq),
        ),
        [ThroughputMeasure(BenchMetric.flops, compute_flops())],
    )
    ctx.synchronize()
    _ = cb_q
    _ = cb_k
    _ = cb_v
    _ = cb_o
    _ = vl_d^


def main() raises:
    print("== bench_apple_fa_prefill: fa_prefill_apple vs mha_gpu_naive")
    comptime qkv = DType.bfloat16
    # Shape knobs via compile-time defines (pass as `get_defined_int[seq]=8192`
    # etc. after `--`, or `-D seq=8192` to `mojo`). With `seq` unset the default
    # sweep runs; with `seq` set, one user-specified (batch, seq) shape runs.
    comptime d = get_defined_int["depth", 128]()
    comptime nh = get_defined_int["heads", 32]()
    comptime kvh = get_defined_int["kv_heads", 8]()
    comptime user_seq = get_defined_int["seq", 0]()
    comptime user_batch = get_defined_int["batch", 1]()
    comptime causal = get_defined_bool["causal", True]()
    var m = Bench()
    with DeviceContext() as ctx:
        comptime if user_seq > 0:
            # One user-specified shape (mask selected by `causal`, default True).
            comptime if causal:
                _bench_prefill[qkv, d, nh, kvh, CausalMask](
                    m, CausalMask(), user_batch, user_seq, ctx
                )
            else:
                _bench_prefill[qkv, d, nh, kvh, NullMask](
                    m, NullMask(), user_batch, user_seq, ctx
                )
        else:
            # Default sweep. fa_prefill_apple vs mha_gpu_naive (causal) at small
            # seqs (naive is O(seq^2), so cap it short) -- the speedup over the
            # fallback -- then fa_prefill_apple throughput across the seq range,
            # causal + NullMask (the honest full-work rate), b1.
            var small = [512, 1024, 2048]
            for i in range(len(small)):
                _bench_shape[qkv, d, nh, kvh, naive=False](m, 1, small[i], ctx)
                _bench_shape[qkv, d, nh, kvh, naive=True](m, 1, small[i], ctx)
            var seqs = [512, 1024, 2048, 4096, 8192, 16384]
            for i in range(len(seqs)):
                _bench_prefill[qkv, d, nh, kvh, CausalMask](
                    m, CausalMask(), 1, seqs[i], ctx
                )
                _bench_prefill[qkv, d, nh, kvh, NullMask](
                    m, NullMask(), 1, seqs[i], ctx
                )
    m.dump_report()
