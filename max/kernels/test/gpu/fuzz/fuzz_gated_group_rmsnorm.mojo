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
# Fuzz target: gated_group_rmsnorm_gpu (Mamba-2 gated group RMSNorm).
#
# `group_size` is compile-time (`-D ggr_group_size=512`, the Nemotron value;
# 520 and 960 cover the vector-loop tail, 100 covers the scalar fallback). The
# runtime axes are rows, group count, and the gate view: `gate_pad` widens the
# gate row stride past the logical width and `gate_off` shifts its base pointer
# into the row, so the sweep crosses the 16-byte alignment. A gate view that is
# not 16-byte aligned claims no alignment, as the graph compiler's fused input
# does. `y`, `weight` and `output` stay aligned.

from std.math import rsqrt
from std.random import random_ui64, seed
from std.sys import size_of
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.activations import silu
from state_space.gated_group_rmsnorm import (
    gated_group_rmsnorm_gpu,
)

from _fuzz import (
    VD_NORMAL,
    boundary_int,
    collect_args,
    fill_all_equal,
    fill_normal,
    fill_sparse,
    fill_uniform,
    flag,
    flag_int,
    numeric_check,
)

comptime dtype = DType.bfloat16
comptime group_size = get_defined_int["ggr_group_size", 512]()
comptime EPS = Float32(1e-5)
comptime TILE = 8  # elements per 16-byte vector: the alignment pivot.
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var rows: Int
    var num_groups: Int
    var gate_pad: Int  # gate row stride minus the logical width
    var gate_off: Int  # gate base offset in elements, <= gate_pad
    var data_seed: Int
    var dist: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "rows=",
            self.rows,
            " num_groups=",
            self.num_groups,
            " gate_pad=",
            self.gate_pad,
            " gate_off=",
            self.gate_off,
            " data_seed=",
            self.data_seed,
            " dist=",
            self.dist,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var pad = boundary_int(0, 6400, 64)
        specs.append(
            CaseSpec(
                boundary_int(1, 130, 4),
                boundary_int(1, 16, 8),
                pad,
                boundary_int(0, min(pad, 40), TILE),
                boundary_int(0, 1_000_000, 1000),
                Int(random_ui64(0, 3)),
            )
        )
    return specs^


def fill_stable(span: Span[mut=True, Scalar[dtype], _], dist: Int):
    """Finite, moderate values; the sparse case produces all-zero groups."""
    if dist == 0:
        fill_normal(span, mean=0.0, std=1.0)
    elif dist == 1:
        fill_uniform(span, lo=-2.0, hi=2.0)
    elif dist == 2:
        fill_sparse(span, density=0.05, lo=-2.0, hi=2.0)
    else:
        fill_all_equal(span, value=0.75)


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    seed(spec.data_seed)
    var rows = spec.rows
    var num_groups = spec.num_groups
    var inter = num_groups * group_size
    var gstride = inter + spec.gate_pad
    var total = rows * inter

    var y_h = ctx.enqueue_create_host_buffer[dtype](total)
    var gate_h = ctx.enqueue_create_host_buffer[dtype](rows * gstride)
    var w_h = ctx.enqueue_create_host_buffer[.float32](inter)
    var out_h = ctx.enqueue_create_host_buffer[dtype](total)
    fill_stable(y_h.as_span(), spec.dist)
    fill_stable(gate_h.as_span(), spec.dist)
    fill_uniform(w_h.as_span(), lo=0.1, hi=2.0)

    var y_d = ctx.enqueue_create_buffer[dtype](total)
    var gate_d = ctx.enqueue_create_buffer[dtype](rows * gstride)
    var w_d = ctx.enqueue_create_buffer[.float32](inter)
    var out_d = ctx.enqueue_create_buffer[dtype](total)
    ctx.enqueue_copy(y_d, y_h)
    ctx.enqueue_copy(gate_d, gate_h)
    ctx.enqueue_copy(w_d, w_h)

    var gate_view = gate_d.create_sub_buffer[dtype](
        spec.gate_off, rows * gstride - spec.gate_off
    )
    var y_t = TileTensor(y_d, row_major(rows, inter))
    var gate_t = TileTensor(gate_view, row_major(rows, gstride))
    var w_t = TileTensor(w_d, row_major(inter))
    var out_t = TileTensor(out_d, row_major(rows, inter))

    def gate_aligned[
        width: Int, alignment: Int
    ](n: Int, col: Int) {var gate_t} -> SIMD[dtype, width]:
        return gate_t.load[width=width, alignment=alignment * size_of[dtype]()](
            (n, col)
        )

    # A view that cannot prove 16-byte alignment claims none, as the graph
    # compiler's fused input does.
    def gate_unaligned[
        width: Int, alignment: Int
    ](n: Int, col: Int) {var gate_t} -> SIMD[dtype, width]:
        return gate_t.load[width=width, alignment=size_of[dtype]()]((n, col))

    var gate_aligned_view = (
        spec.gate_off % TILE == 0 and (gstride * size_of[dtype]()) % 16 == 0
    )
    if gate_aligned_view:
        gated_group_rmsnorm_gpu[dtype, dtype, group_size](
            out_t, y_t, gate_aligned, w_t, rows, num_groups, EPS, ctx
        )
    else:
        gated_group_rmsnorm_gpu[dtype, dtype, group_size](
            out_t, y_t, gate_unaligned, w_t, rows, num_groups, EPS, ctx
        )
    ctx.synchronize()

    if check:
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        var expected = ctx.enqueue_create_host_buffer[dtype](total)
        for n in range(rows):
            for g in range(num_groups):
                var base = g * group_size
                var m2 = Float64(0)
                for j in range(group_size):
                    var yv = y_h[n * inter + base + j].cast[.float32]()
                    var gv = gate_h[
                        n * gstride + spec.gate_off + base + j
                    ].cast[.float32]()
                    var gated = Float64(yv * silu(gv))
                    m2 += gated * gated
                var nf = Float32(rsqrt(m2 / Float64(group_size) + Float64(EPS)))
                for j in range(group_size):
                    var idx = n * inter + base + j
                    var yv = y_h[idx].cast[.float32]()
                    var gv = gate_h[
                        n * gstride + spec.gate_off + base + j
                    ].cast[.float32]()
                    var gated = yv * silu(gv)
                    var t_in = (gated * nf).cast[dtype]().cast[.float32]()
                    expected[idx] = (w_h[base + j] * t_in).cast[dtype]()
        # Two bf16 roundings (`t_in` and the output) allow a 2-ulp flip when
        # the fp32 reduction order moves `nf` by an ulp.
        if not numeric_check(
            out_h.as_span(), expected.as_span(), atol=1e-3, rtol=2e-2
        ):
            raise Error("gated_group_rmsnorm: mismatch vs reference")

    _ = y_d
    _ = gate_d
    _ = w_d
    _ = out_d


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
                "rows=",
                specs[i].rows,
                "num_groups=",
                specs[i].num_groups,
                "gate_pad=",
                specs[i].gate_pad,
                "gate_off=",
                specs[i].gate_off,
                "data_seed=",
                specs[i].data_seed,
                "dist=",
                specs[i].dist,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--rows", 8),
            flag_int(args, "--num_groups", 8),
            flag_int(args, "--gate_pad", 0),
            flag_int(args, "--gate_off", 0),
            flag_int(args, "--data_seed", 1),
            flag_int(args, "--dist", VD_NORMAL),
        )
        print("FUZZ_SINGLE", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_gated_group_rmsnorm seed=",
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
