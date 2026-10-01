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
# Fuzz target: the B200 Mamba-2 SSD scan
# (`mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split`).
#
# Covers decode (one token per sequence) and short ragged prefill, with the
# strided x/B/C/dt views the Nemotron-H mixer passes, scattered
# `cache_indices` into a pool larger than the batch, and a random mix of
# sequences with and without an initial state. The `ref` oracle (--check 1)
# compares y and the whole state pool against the CPU in-place reference, so
# slots outside `cache_indices` must come back untouched. Memory-safety oracles
# need no extra support. DSTATE is compile-time (`-D ssd_dstate=16|64|128|256`).

from std.math import ceildiv
from std.memory import alloc
from std.random import rand, random_ui64, seed
from std.sys.defines import get_defined_int

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.mamba2_ssd_scan import (
    Strides1D,
    Strides2D,
    Strides3D,
    Strides4D,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split,
)

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime dtype = DType.bfloat16
comptime DSTATE = get_defined_int["ssd_dstate", 128]()
comptime NGROUPS = 8
comptime DSTATE_SPLIT = 8
comptime BLOCK_THREADS = 128
comptime CH_PER_BLOCK = BLOCK_THREADS // DSTATE_SPLIT
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()

# Row padding is a multiple of 16 elements so the kernel's 32-byte B/C loads
# stay aligned, as they are for the model's conv output.
comptime PAD_UNIT = 16


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var batch: Int
    var nheads8: Int  # nheads = 8 * nheads8
    var head_dim: Int
    var max_len: Int  # 1 is decode, more is ragged prefill (0-length allowed)
    var len_seed: Int
    var pad_units: Int  # x/B/C/dt row padding, in units of PAD_UNIT elements
    var init_mode: Int  # 0 no tensor, 1 all sequences, 2 random subset
    var slot_seed: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "batch=",
            self.batch,
            " nheads8=",
            self.nheads8,
            " head_dim=",
            self.head_dim,
            " max_len=",
            self.max_len,
            " len_seed=",
            self.len_seed,
            " pad_units=",
            self.pad_units,
            " init_mode=",
            self.init_mode,
            " slot_seed=",
            self.slot_seed,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var decode = random_ui64(0, 1) == 0
        specs.append(
            CaseSpec(
                boundary_int(1, 64, 8),
                boundary_int(1, 8, 2),
                boundary_int(1, 80, 16),
                1 if decode else boundary_int(2, 24, 4),
                Int(random_ui64(0, 1 << 30)),
                Int(random_ui64(0, 4)),
                Int(random_ui64(0, 2)),
                Int(random_ui64(0, 1 << 30)),
            )
        )
    return specs^


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    var batch = spec.batch
    var nheads = 8 * spec.nheads8
    var head_dim = spec.head_dim
    var ratio = nheads // NGROUPS
    var pad = spec.pad_units * PAD_UNIT
    var pool_slots = 2 * batch
    var state_row = head_dim * DSTATE
    var head_row = nheads * state_row

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

    var x_row = nheads * head_dim + pad
    var bc_row = NGROUPS * DSTATE + pad
    var dt_row = nheads + pad

    var x_h = alloc[Scalar[dtype]](total * x_row)
    var dt_h = alloc[Scalar[dtype]](total * dt_row)
    var A_h = alloc[Scalar[dtype]](nheads)
    var B_h = alloc[Scalar[dtype]](total * bc_row)
    var C_h = alloc[Scalar[dtype]](total * bc_row)
    var D_h = alloc[Scalar[dtype]](nheads)
    var dt_bias_h = alloc[Scalar[dtype]](nheads)
    var his_h = alloc[Scalar[.bool]](batch)
    var qsl_h = alloc[Int32](batch + 1)
    var slot_h = alloc[UInt32](batch)
    var pool_h = alloc[Float32](pool_slots * head_row)
    var pool_ref_h = alloc[Float32](pool_slots * head_row)
    var y_ref_h = alloc[Scalar[dtype]](total * nheads * head_dim)
    var y_gpu_h = alloc[Scalar[dtype]](total * nheads * head_dim)

    rand(x_h, total * x_row)
    rand(dt_h, total * dt_row)
    rand(A_h, nheads)
    rand(B_h, total * bc_row)
    rand(C_h, total * bc_row)
    rand(D_h, nheads)
    rand(dt_bias_h, nheads)
    rand(pool_h, pool_slots * head_row)
    for i in range(nheads):
        A_h.store(i, Scalar[dtype](Float32(A_h.load(i)) * -1.0 - 0.1))
    for i in range(total * dt_row):
        dt_h.store(i, Scalar[dtype](Float32(dt_h.load(i)) - 0.5))
    for i in range(pool_slots * head_row):
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

    var x_strides: Strides3D = (x_row, head_dim, 1)
    var dt_strides: Strides2D = (dt_row, 1)
    var A_strides: Strides1D = (1,)
    var B_strides: Strides3D = (bc_row, DSTATE, 1)
    var C_strides: Strides3D = (bc_row, DSTATE, 1)
    var D_strides: Strides1D = (1,)
    var dt_bias_strides: Strides1D = (1,)
    var y_strides: Strides3D = (nheads * head_dim, head_dim, 1)
    var pool_strides: Strides4D = (head_row, state_row, DSTATE, 1)

    var x_d = ctx.enqueue_create_buffer[dtype](total * x_row)
    var dt_d = ctx.enqueue_create_buffer[dtype](total * dt_row)
    var A_d = ctx.enqueue_create_buffer[dtype](nheads)
    var B_d = ctx.enqueue_create_buffer[dtype](total * bc_row)
    var C_d = ctx.enqueue_create_buffer[dtype](total * bc_row)
    var D_d = ctx.enqueue_create_buffer[dtype](nheads)
    var dt_bias_d = ctx.enqueue_create_buffer[dtype](nheads)
    var his_d = ctx.enqueue_create_buffer[.bool](batch)
    var qsl_d = ctx.enqueue_create_buffer[.int32](batch + 1)
    var slot_d = ctx.enqueue_create_buffer[.uint32](batch)
    var pool_d = ctx.enqueue_create_buffer[.float32](pool_slots * head_row)
    var y_d = ctx.enqueue_create_buffer[dtype](total * nheads * head_dim)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(dt_d, dt_h)
    ctx.enqueue_copy(A_d, A_h)
    ctx.enqueue_copy(B_d, B_h)
    ctx.enqueue_copy(C_d, C_h)
    ctx.enqueue_copy(D_d, D_h)
    ctx.enqueue_copy(dt_bias_d, dt_bias_h)
    ctx.enqueue_copy(his_d, his_h)
    ctx.enqueue_copy(qsl_d, qsl_h)
    ctx.enqueue_copy(slot_d, slot_h)
    ctx.enqueue_copy(pool_d, pool_h)

    var x_g = TileTensor(x_d, row_major(total, nheads, head_dim))
    var dt_g = TileTensor(dt_d, row_major(total, nheads))
    var A_g = TileTensor(A_d, row_major(nheads))
    var B_g = TileTensor(B_d, row_major(total, NGROUPS, DSTATE))
    var C_g = TileTensor(C_d, row_major(total, NGROUPS, DSTATE))
    var D_g = TileTensor(D_d, row_major(nheads))
    var dt_bias_g = TileTensor(dt_bias_d, row_major(nheads))
    var y_g = TileTensor(y_d, row_major(total, nheads, head_dim))
    var pool_g = TileTensor(
        pool_d, row_major(pool_slots, nheads, head_dim, DSTATE)
    )
    var qsl_g = TileTensor(qsl_d, row_major(batch + 1))
    var his_g = TileTensor(his_d, row_major(his_len))
    var slot_g = TileTensor(slot_d, row_major(batch))

    comptime kernel = mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split[
        dtype,
        DSTATE,
        x_g.LayoutType,
        dt_g.LayoutType,
        A_g.LayoutType,
        B_g.LayoutType,
        C_g.LayoutType,
        D_g.LayoutType,
        dt_bias_g.LayoutType,
        y_g.LayoutType,
        pool_g.LayoutType,
        qsl_g.LayoutType,
        his_g.LayoutType,
        slot_g.LayoutType,
        x_g.Engine,
        DSTATE_SPLIT,
    ]
    var compiled = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled,
        Int32(nheads),
        Int32(head_dim),
        Int32(NGROUPS),
        Int32(ratio),
        Int32(batch),
        Int8(1),
        x_g,
        dt_g,
        A_g,
        B_g,
        C_g,
        D_g,
        dt_bias_g,
        y_g,
        pool_g,
        qsl_g,
        his_g,
        slot_g,
        x_strides,
        dt_strides,
        A_strides,
        B_strides,
        C_strides,
        D_strides,
        dt_bias_strides,
        y_strides,
        pool_strides,
        grid_dim=(ceildiv(head_dim, CH_PER_BLOCK), nheads, batch),
        block_dim=(DSTATE_SPLIT, CH_PER_BLOCK, 1),
    )
    ctx.synchronize()

    if check:
        var x_t = TileTensor(x_h, row_major(total, nheads, head_dim))
        var dt_t = TileTensor(dt_h, row_major(total, nheads))
        var A_t = TileTensor(A_h, row_major(nheads))
        var B_t = TileTensor(B_h, row_major(total, NGROUPS, DSTATE))
        var C_t = TileTensor(C_h, row_major(total, NGROUPS, DSTATE))
        var D_t = TileTensor(D_h, row_major(nheads))
        var dt_bias_t = TileTensor(dt_bias_h, row_major(nheads))
        var y_t = TileTensor(y_ref_h, row_major(total, nheads, head_dim))
        var pool_t = TileTensor(
            pool_ref_h, row_major(pool_slots, nheads, head_dim, DSTATE)
        )
        var qsl_t = TileTensor(qsl_h, row_major(batch + 1))
        var his_t = TileTensor(his_h, row_major(his_len))
        var slot_t = TileTensor(slot_h, row_major(batch))
        mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu[dtype, DSTATE](
            nheads,
            head_dim,
            NGROUPS,
            ratio,
            batch,
            Int8(1),
            x_t,
            dt_t,
            A_t,
            B_t,
            C_t,
            D_t,
            dt_bias_t,
            y_t,
            pool_t,
            qsl_t,
            his_t,
            slot_t,
            x_strides,
            dt_strides,
            A_strides,
            B_strides,
            C_strides,
            D_strides,
            dt_bias_strides,
            y_strides,
            pool_strides,
        )
        ctx.enqueue_copy(y_gpu_h, y_d)
        ctx.enqueue_copy(pool_h, pool_d)
        ctx.synchronize()

        var n_y = total * nheads * head_dim
        var n_pool = pool_slots * head_row
        if not numeric_check(
            Span(unsafe_ptr=y_gpu_h, length=n_y),
            Span(unsafe_ptr=y_ref_h, length=n_y),
        ):
            raise Error("mamba2_ssd_scan y mismatch")
        if not numeric_check(
            Span(unsafe_ptr=pool_h, length=n_pool),
            Span(unsafe_ptr=pool_ref_h, length=n_pool),
            atol=1e-4,
            rtol=1e-3,
        ):
            raise Error("mamba2_ssd_scan state pool mismatch")

    x_h.free()
    dt_h.free()
    A_h.free()
    B_h.free()
    C_h.free()
    D_h.free()
    dt_bias_h.free()
    his_h.free()
    qsl_h.free()
    slot_h.free()
    pool_h.free()
    pool_ref_h.free()
    y_ref_h.free()
    y_gpu_h.free()
    _ = x_d
    _ = dt_d
    _ = A_d
    _ = B_d
    _ = C_d
    _ = D_d
    _ = dt_bias_d
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
                "nheads8=",
                specs[i].nheads8,
                "head_dim=",
                specs[i].head_dim,
                "max_len=",
                specs[i].max_len,
                "len_seed=",
                specs[i].len_seed,
                "pad_units=",
                specs[i].pad_units,
                "init_mode=",
                specs[i].init_mode,
                "slot_seed=",
                specs[i].slot_seed,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--batch", 4),
            flag_int(args, "--nheads8", 1),
            flag_int(args, "--head_dim", 64),
            flag_int(args, "--max_len", 1),
            flag_int(args, "--len_seed", 1),
            flag_int(args, "--pad_units", 0),
            flag_int(args, "--init_mode", 1),
            flag_int(args, "--slot_seed", 1),
        )
        print("FUZZ_SINGLE", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_mamba2_ssd_scan seed=", the_seed, "budget=", the_budget, "==="
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
