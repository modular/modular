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
#
# Fuzz target: the varlen causal conv1d (`causal_conv1d_varlen_fwd_gpu`),
# channels last, as the Nemotron-H mixer runs it.
#
# The input is a column range of a wider row, like a slice of the fused
# in-projection output: an arbitrary element offset and row padding. Covers decode
# and ragged prefill (0-length sequences allowed), scattered `cache_indices`
# into a pool larger than the batch, and a random mix of sequences with and
# without an initial state. The `ref` oracle (--check 1) compares the output and
# the whole conv-state pool against the CPU reference, so slots outside
# `cache_indices` must come back untouched. Memory-safety oracles need no extra
# support.

from std.memory import alloc
from std.random import rand, random_ui64, seed
from std.sys.defines import get_defined_int

from layout import MixedLayout, TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.varlen_causal_conv1d import (
    causal_conv1d_varlen_fwd_cpu,
    causal_conv1d_varlen_fwd_gpu,
)

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime dtype = DType.bfloat16
comptime WIDTH = get_defined_int["conv_width", 4]()
# bf16 prefill: 64 threads times 4 channels, and tokens per tile.
comptime BLOCK_CHANNELS = 256
comptime TILE_SEQ = 64
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var batch: Int
    var dim: Int
    var max_len: Int  # 1 is decode, more is ragged prefill (0-length allowed)
    var len_seed: Int
    var x_off: Int  # first column of the conv input in its row, in elements
    var pad: Int  # elements after the conv input in its row
    var init_mode: Int  # 0 no tensor, 1 all sequences, 2 random subset
    var slot_seed: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "batch=",
            self.batch,
            " dim=",
            self.dim,
            " max_len=",
            self.max_len,
            " len_seed=",
            self.len_seed,
            " x_off=",
            self.x_off,
            " pad=",
            self.pad,
            " init_mode=",
            self.init_mode,
            " slot_seed=",
            self.slot_seed,
        )


def _conv_cpu[
    dtype: DType,
    //,
    silu_activation: Bool,
    use_residual: Bool = False,
    channels_last: Bool = False,
](
    x: TileTensor[mut=False, dtype, ...],
    weight: TileTensor[mut=False, dtype, ...],
    bias: TileTensor[mut=False, dtype, ...],
    query_start_loc: TileTensor[mut=False, .int32, ...],
    cache_indices: TileTensor[mut=False, .uint32, ...],
    has_initial_state: TileTensor[mut=False, .bool, ...],
    conv_states: TileTensor[mut=True, ...],
    output: TileTensor[mut=True, dtype, ...],
):
    """Runs the CPU conv with reader functions over these tensors.

    An empty `cache_indices` maps sequence `b` to slot `b`, as in the op.
    """

    def x_fn[
        width: Int, alignment: Int
    ](i: Int, j: Int) {var x} -> SIMD[dtype, width]:
        return x.load[width=width]((i, j))

    def slot_fn[
        width: Int, alignment: Int
    ](b: Int) {var cache_indices} -> SIMD[.uint32, width]:
        if Int(cache_indices.dim[0]()) == 0:
            return SIMD[.uint32, width](UInt32(b))
        return cache_indices.load[width=width]((b,))

    causal_conv1d_varlen_fwd_cpu[silu_activation, use_residual, channels_last](
        weight,
        bias,
        query_start_loc,
        has_initial_state,
        conv_states,
        output,
        x_fn,
        slot_fn,
    )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var decode = random_ui64(0, 1) == 0
        specs.append(
            CaseSpec(
                boundary_int(1, 64, 8),
                boundary_int(1, 1100, BLOCK_CHANNELS),
                1 if decode else boundary_int(2, 300, TILE_SEQ),
                Int(random_ui64(0, 1 << 30)),
                boundary_int(0, 40, 8),
                boundary_int(0, 40, 8),
                Int(random_ui64(0, 2)),
                Int(random_ui64(0, 1 << 30)),
            )
        )
    return specs^


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    var batch = spec.batch
    var dim = spec.dim
    var pool_slots = 2 * batch
    var state_len = WIDTH - 1

    seed(spec.len_seed)
    var lens = List[Int]()
    var total = 0
    for _ in range(batch):
        var n = 1 if spec.max_len == 1 else Int(
            random_ui64(0, UInt64(spec.max_len))
        )
        lens.append(n)
        total += n
    if total == 0:
        lens[0] = 1
        total = 1

    # Distinct random slots: a partial Fisher-Yates shuffle of the pool.
    seed(spec.slot_seed)
    var slots = List[Int]()
    for i in range(pool_slots):
        slots.append(i)
    for i in range(batch):
        var j = Int(random_ui64(UInt64(i), UInt64(pool_slots - 1)))
        var tmp = slots[i]
        slots[i] = slots[j]
        slots[j] = tmp

    var proj_row = spec.x_off + dim + spec.pad
    var proj_h = alloc[Scalar[dtype]](total * proj_row)
    var weight_h = alloc[Scalar[dtype]](dim * WIDTH)
    var bias_h = alloc[Scalar[dtype]](dim)
    var his_h = alloc[Scalar[.bool]](batch)
    var qsl_h = alloc[Int32](batch + 1)
    var slot_h = alloc[UInt32](batch)
    var pool_h = alloc[Scalar[dtype]](pool_slots * dim * state_len)
    var pool_ref_h = alloc[Scalar[dtype]](pool_slots * dim * state_len)
    var y_ref_h = alloc[Scalar[dtype]](total * dim)
    var y_gpu_h = alloc[Scalar[dtype]](total * dim)

    rand(proj_h, total * proj_row)
    rand(weight_h, dim * WIDTH)
    rand(bias_h, dim)
    rand(pool_h, pool_slots * dim * state_len)
    for i in range(pool_slots * dim * state_len):
        pool_ref_h.store(i, pool_h.load(i))
    var cum = 0
    qsl_h.store(0, Int32(0))
    for b in range(batch):
        cum += lens[b]
        qsl_h.store(b + 1, Int32(cum))
        slot_h.store(b, UInt32(slots[b]))
        var has_init = spec.init_mode == 1 or (
            spec.init_mode == 2 and random_ui64(0, 1) == 1
        )
        his_h.store(b, Scalar[.bool](has_init))

    # Mode 0 passes an empty has_initial_state tensor, which the kernels read
    # as "no sequence has an initial state".
    var his_len = 0 if spec.init_mode == 0 else batch

    var x_strides = (proj_row, 1)

    var proj_d = ctx.enqueue_create_buffer[dtype](total * proj_row)
    var weight_d = ctx.enqueue_create_buffer[dtype](dim * WIDTH)
    var bias_d = ctx.enqueue_create_buffer[dtype](dim)
    var his_d = ctx.enqueue_create_buffer[.bool](batch)
    var qsl_d = ctx.enqueue_create_buffer[.int32](batch + 1)
    var slot_d = ctx.enqueue_create_buffer[.uint32](batch)
    var pool_d = ctx.enqueue_create_buffer[dtype](pool_slots * dim * state_len)
    var y_d = ctx.enqueue_create_buffer[dtype](total * dim)
    ctx.enqueue_copy(proj_d, proj_h)
    ctx.enqueue_copy(weight_d, weight_h)
    ctx.enqueue_copy(bias_d, bias_h)
    ctx.enqueue_copy(his_d, his_h)
    ctx.enqueue_copy(qsl_d, qsl_h)
    ctx.enqueue_copy(slot_d, slot_h)
    ctx.enqueue_copy(pool_d, pool_h)

    var x_d = proj_d.create_sub_buffer[dtype](
        spec.x_off, total * proj_row - spec.x_off
    )
    var x_g = TileTensor(x_d, MixedLayout((total, dim), x_strides))
    var weight_g = TileTensor(weight_d, row_major(dim, WIDTH))
    var bias_g = TileTensor(bias_d, row_major(dim))
    var qsl_g = TileTensor(qsl_d, row_major(batch + 1))
    var slot_g = TileTensor(slot_d, row_major(batch))
    var his_g = TileTensor(his_d, row_major(his_len))
    var pool_g = TileTensor(pool_d, row_major(pool_slots, dim, state_len))
    var y_g = TileTensor(y_d, row_major(total, dim))

    def x_fn[
        width: Int, alignment: Int
    ](i: Int, j: Int) {var x_g} -> SIMD[dtype, width]:
        return x_g.load[width=width]((i, j))

    def slot_fn[
        width: Int, alignment: Int
    ](b: Int) {var slot_g} -> SIMD[.uint32, width]:
        return slot_g.load[width=width]((b,))

    var x_addr = Int(x_g.unsafe_ptr())
    var x_row_stride = Int(x_g.layout.stride[0]().value())
    causal_conv1d_varlen_fwd_gpu[
        WIDTH, silu_activation=True, channels_last=True
    ](
        weight_g,
        bias_g,
        qsl_g,
        his_g,
        pool_g,
        y_g,
        x_addr,
        x_row_stride,
        x_fn,
        slot_fn,
        ctx,
    )
    ctx.synchronize()

    if check:
        var x_t = TileTensor(
            proj_h + spec.x_off, MixedLayout((total, dim), x_strides)
        )
        var weight_t = TileTensor(weight_h, row_major(dim, WIDTH))
        var bias_t = TileTensor(bias_h, row_major(dim))
        var qsl_t = TileTensor(qsl_h, row_major(batch + 1))
        var slot_t = TileTensor(slot_h, row_major(batch))
        var his_t = TileTensor(his_h, row_major(his_len))
        var pool_t = TileTensor(
            pool_ref_h, row_major(pool_slots, dim, state_len)
        )
        var y_t = TileTensor(y_ref_h, row_major(total, dim))
        _conv_cpu[
            True,
            channels_last=True,
        ](
            x_t,
            weight_t,
            bias_t,
            qsl_t,
            slot_t,
            his_t,
            pool_t,
            y_t,
        )
        ctx.enqueue_copy(y_gpu_h, y_d)
        ctx.enqueue_copy(pool_h, pool_d)
        ctx.synchronize()

        var n_y = total * dim
        var n_pool = pool_slots * dim * state_len
        if not numeric_check(
            Span(unsafe_ptr=y_gpu_h, length=n_y),
            Span(unsafe_ptr=y_ref_h, length=n_y),
            rtol=1e-2,
        ):
            raise Error("causal_conv1d_varlen y mismatch")
        if not numeric_check(
            Span(unsafe_ptr=pool_h, length=n_pool),
            Span(unsafe_ptr=pool_ref_h, length=n_pool),
            rtol=1e-2,
        ):
            raise Error("causal_conv1d_varlen state pool mismatch")

    proj_h.free()
    weight_h.free()
    bias_h.free()
    his_h.free()
    qsl_h.free()
    slot_h.free()
    pool_h.free()
    pool_ref_h.free()
    y_ref_h.free()
    y_gpu_h.free()
    _ = proj_d
    _ = x_d
    _ = weight_d
    _ = bias_d
    _ = his_d
    _ = qsl_d
    _ = slot_d
    _ = pool_d
    _ = y_d


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "batch=",
                specs[i].batch,
                "dim=",
                specs[i].dim,
                "max_len=",
                specs[i].max_len,
                "len_seed=",
                specs[i].len_seed,
                "x_off=",
                specs[i].x_off,
                "pad=",
                specs[i].pad,
                "init_mode=",
                specs[i].init_mode,
                "slot_seed=",
                specs[i].slot_seed,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--batch", 4),
            flag_int(args, "--dim", 128),
            flag_int(args, "--max_len", 1),
            flag_int(args, "--len_seed", 1),
            flag_int(args, "--x_off", 0),
            flag_int(args, "--pad", 0),
            flag_int(args, "--init_mode", 1),
            flag_int(args, "--slot_seed", 1),
        )
        print("FUZZ_SINGLE", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_causal_conv1d_varlen seed=",
        the_seed,
        "budget=",
        the_budget,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
