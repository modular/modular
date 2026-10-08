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

from std.math import max, min
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.argsort import argsort

from _fuzz import (
    VD_LARGE,
    boundary_int,
    collect_args,
    fill_by_dist,
    flag,
    flag_int,
)

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var n: Int
    var ascending: Int
    var dist: Int
    var data_seed: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "n=",
            self.n,
            " ascending=",
            self.ascending,
            " dist=",
            self.dist,
            " data_seed=",
            self.data_seed,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        specs.append(
            CaseSpec(
                boundary_int(1, 8192, 256),
                Int(random_ui64(0, 1)),
                Int(random_ui64(0, 4)),
                Int(random_ui64(0, 2147483647)),
            )
        )
    return specs^


def _fill_input[
    dtype: DType
](span: Span[mut=True, Scalar[dtype], _], dist: Int):
    comptime if dtype == DType.int64:
        if dist == VD_LARGE:
            # Nearby keys above 2**53 expose lossy float64 index oracles.
            for i in range(len(span)):
                var value = (Int64(1) << 60) + Int64(random_ui64(0, 65535))
                span[i] = (-value if random_ui64(0, 1) == 0 else value).cast[
                    dtype
                ]()
            return
        var values = List[Float32](length=len(span), fill=0.0)
        fill_by_dist(Span(values), dist)
        for i in range(len(span)):
            span[i] = (values[i] * 65536.0).cast[dtype]()
    else:
        fill_by_dist(span, dist)


def _run[
    dtype: DType, ascending: Bool
](ctx: DeviceContext, n: Int, dist: Int, check: Bool) raises:
    var input_h = ctx.enqueue_create_host_buffer[dtype](n)
    _fill_input(input_h.as_span(), dist)
    var input_d = ctx.enqueue_create_buffer[dtype](n)
    var indices_d = ctx.enqueue_create_buffer[.int64](n)
    var indices_h = ctx.enqueue_create_host_buffer[.int64](n)
    for i in range(n):
        indices_h[i] = -1
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(indices_d, indices_h)

    var input = TileTensor(input_d, row_major(n))
    var indices = TileTensor(indices_d, row_major(n))
    argsort[ascending=ascending, target="gpu"](indices, input, ctx)
    ctx.enqueue_copy(indices_h, indices_d)
    ctx.synchronize()

    var seen = List[Bool](length=n, fill=False)
    for i in range(n):
        var index = Int(indices_h[i])
        if index < 0 or index >= n:
            print(
                "FUZZ_NUMERIC_FAIL reason=index_range position=",
                i,
                " index=",
                index,
                " n=",
                n,
            )
            raise Error("argsort returned an invalid index")
        if seen[index]:
            print("FUZZ_NUMERIC_FAIL reason=duplicate_index index=", index)
            raise Error("argsort did not return a permutation")
        seen[index] = True

    if check:
        var reference = List[Scalar[dtype]](length=n, fill=0)
        for i in range(n):
            reference[i] = input_h[i]

        def compare(lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Bool:
            comptime if ascending:
                return lhs < rhs
            else:
                return lhs > rhs

        sort(reference, compare)
        # Selections are exact. Float64 conversion would hide int64 low bits.
        for i in range(n):
            var value = input_h[Int(indices_h[i])]
            if value != reference[i]:
                print(
                    "FUZZ_NUMERIC_FAIL reason=sorted_value position=",
                    i,
                    " actual=",
                    value,
                    " expected=",
                    reference[i],
                )
                raise Error("argsort values differ from the CPU reference")

        var input_after = ctx.enqueue_create_host_buffer[dtype](n)
        ctx.enqueue_copy(input_after, input_d)
        ctx.synchronize()
        for i in range(n):
            if input_after[i] != input_h[i]:
                print("FUZZ_NUMERIC_FAIL reason=input_mutation position=", i)
                raise Error("argsort modified its input")

    _ = input_d
    _ = indices_d
    _ = input
    _ = indices


def _run_in_order[
    dtype: DType
](ctx: DeviceContext, n: Int, dist: Int, ascending: Bool, check: Bool) raises:
    if ascending:
        _run[dtype, True](ctx, n, dist, check)
    else:
        _run[dtype, False](ctx, n, dist, check)


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    # Shrinking to zero must stay in the kernel's nonempty, finite-key domain.
    var n = max(1, spec.n)
    var dist = max(0, min(4, spec.dist))
    seed(spec.data_seed)
    comptime if get_defined_int["argsort_i64", 0]() == 1:
        _run_in_order[DType.int64](ctx, n, dist, spec.ascending != 0, check)
    else:
        _run_in_order[DType.float32](ctx, n, dist, spec.ascending != 0, check)


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
                "n=",
                specs[i].n,
                "ascending=",
                specs[i].ascending,
                "dist=",
                specs[i].dist,
                "data_seed=",
                specs[i].data_seed,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--n", 512),
            flag_int(args, "--ascending", 1),
            flag_int(args, "--dist", 0),
            flag_int(args, "--data_seed", the_seed),
        )
        print("FUZZ_SINGLE", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
