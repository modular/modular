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
# Fuzz target: layer_norm (`layer_norm_gpu`) (see gpu-kernels-fuzzing-design.md).
#
# Fully runtime-shapeable: fuzzes (rows, cols). Memory-safety oracle by default;
# with --check, an FP64 CPU reference (out = (x-mean)*rsqrt(var+eps)*gamma+beta,
# mean/var over the inner axis) drives the numerical (`ref`) oracle.

from std.math import sqrt
from std.random import rand, seed
from std.sys.defines import get_defined_int

from std.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.normalization import *
from std.utils.index import Index, IndexList

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime ln_type = DType.float32
comptime ln_rank = 2
comptime TILE = 128  # warp-tiling vs block dispatch pivot on cols.
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var rows: Int
    var cols: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("rows=", self.rows, " cols=", self.cols)


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        specs.append(
            CaseSpec(boundary_int(1, 256, 8), boundary_int(1, 8192, TILE))
        )
    return specs^


def _layer_norm_ref(
    src: Span[Scalar[ln_type], _],
    gamma: Span[Scalar[ln_type], _],
    beta: Span[Scalar[ln_type], _],
    dst: Span[mut=True, Scalar[ln_type], _],
    rows: Int,
    cols: Int,
    eps: Float64,
):
    """FP64 CPU layer_norm: out = (x-mean)*rsqrt(var+eps)*gamma + beta."""
    for r in range(rows):
        var base = r * cols
        var mean = Float64(0)
        for c in range(cols):
            mean += src[base + c].cast[DType.float64]()
        mean /= Float64(cols)
        var var_ = Float64(0)
        for c in range(cols):
            var d = src[base + c].cast[DType.float64]() - mean
            var_ += d * d
        var_ /= Float64(cols)
        var norm = 1.0 / sqrt(var_ + eps)
        for c in range(cols):
            var x = src[base + c].cast[DType.float64]()
            var g = gamma[c].cast[DType.float64]()
            var b = beta[c].cast[DType.float64]()
            dst[base + c] = (((x - mean) * norm) * g + b).cast[ln_type]()


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    var rows = spec.rows
    var cols = spec.cols
    var shape = Index(rows, cols)

    var data_h = ctx.enqueue_create_host_buffer[ln_type](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[ln_type](cols)
    var beta_h = ctx.enqueue_create_host_buffer[ln_type](cols)
    rand(data_h.as_span())
    rand(gamma_h.as_span())
    rand(beta_h.as_span())

    var data_d = ctx.enqueue_create_buffer[ln_type](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[ln_type](cols)
    var beta_d = ctx.enqueue_create_buffer[ln_type](cols)
    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    var param_shape = Index(cols)
    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_d, row_major(Coord(param_shape)))
    var epsilon = Scalar[ln_type](1e-5)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    def input_fn[
        width: Int, _rank: Int, alignment: Int
    ](coords: IndexList[_rank]) -> SIMD[ln_type, width]:
        var idx = data_buf.layout(Coord(coords))
        return data_buf.raw_load[width=width, alignment=alignment](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    def gamma_fn[
        width: Int, rank: Int, alignment: Int
    ](coords: IndexList[rank]) -> SIMD[ln_type, width]:
        var idx = gamma.layout(coords[0])
        return gamma.raw_load[width=width, alignment=alignment](idx[0])

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    def output_fn[
        width: SIMDSize, rank_: Int, alignment: Int
    ](coords: IndexList[rank_], val: SIMD[ln_type, width]):
        var idx = data_buf.layout(Coord(coords))
        data_buf.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[ln_type, width]](val)
        )

    layer_norm_gpu[input_fn, gamma_fn, output_fn](shape, beta, epsilon, ctx=ctx)
    ctx.synchronize()

    if check:
        var out_h = ctx.enqueue_create_host_buffer[ln_type](rows * cols)
        ctx.enqueue_copy(out_h, data_d)
        ctx.synchronize()
        var ref_h = ctx.enqueue_create_host_buffer[ln_type](rows * cols)
        _layer_norm_ref(
            data_h.as_span(),
            gamma_h.as_span(),
            beta_h.as_span(),
            ref_h.as_span(),
            rows,
            cols,
            epsilon.cast[DType.float64](),
        )
        if not numeric_check(out_h.as_span(), ref_h.as_span()):
            raise Error("layer_norm numeric mismatch")

    _ = data_d
    _ = gamma_d
    _ = beta_d
    _ = data_buf


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
                "cols=",
                specs[i].cols,
            )
        return

    if mode == "single":
        var rows = flag_int(args, "--rows", 8)
        var cols = flag_int(args, "--cols", 128)
        print("FUZZ_SINGLE rows=", rows, "cols=", cols)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(rows, cols), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print("=== fuzz_layer_norm seed=", the_seed, "budget=", the_budget, "===")
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
