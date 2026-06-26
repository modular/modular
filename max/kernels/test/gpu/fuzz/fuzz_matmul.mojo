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
# Fuzz target: matmul / GEMM (`_matmul_gpu`) (see gpu-kernels-fuzzing-design.md).
#
# On SM100 the tuned tensor-core kernel reads N and K from the tensors' STATIC
# shape, so they must be compile-time (set via `-D N=.. -D K=..`); M is the
# runtime fuzz axis. bf16 + transpose_b is the tuned path (N, K multiples of 8
# keep TMA alignment). Memory-safety oracle (memcheck / redzone).

from std.math import ceildiv
from std.random import rand, seed
from std.sys.defines import get_defined_int

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime dtype = DType.bfloat16
comptime N = get_defined_int[
    "N", 2048
]()  # COMPTIME (tuned SM100 reads static N)
comptime K = get_defined_int["K", 2048]()  # COMPTIME
comptime TILE = 128  # matmul block-M tile -- the interesting modulus for M.
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


def naive_matmul_ref_kernel(
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """C[m,n] = sum_k A[m,k]*B[n,k] with fp32 accumulation (transpose_b).

    A higher-precision reference than same-precision cuBLAS: it accumulates in
    fp32, exposing shared-rounding/reduction bugs the bf16 tensor-core path and
    a same-precision vendor reference would both hide.
    """
    var col = global_idx.x
    var row = global_idx.y
    if row < m and col < n:
        var acc = Float32(0)
        for k_i in range(k):
            acc += (
                a[row * k + k_i].cast[DType.float32]()
                * b[col * k + k_i].cast[DType.float32]()
            )
        c[row * n + col] = acc.cast[dtype]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var m: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("m=", self.m, " N=", N, " K=", K)


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        specs.append(CaseSpec(boundary_int(1, 4096, TILE)))
    return specs^


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    var m = spec.m
    var a_size = m * K
    var b_size = N * K  # transpose_b => B is [N, K]
    var c_size = m * N

    var a_host = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host = ctx.enqueue_create_host_buffer[dtype](b_size)
    rand(a_host.as_span())
    rand(b_host.as_span())

    var a_dev = ctx.enqueue_create_buffer[dtype](a_size)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_size)
    var c_dev = ctx.enqueue_create_buffer[dtype](c_size)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    # M runtime (bare Int); N, K comptime (Idx[...]) so the tuned kernel engages.
    var a = TileTensor(a_dev, row_major(Coord(m, Idx[K])))
    var b = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c = TileTensor(c_dev, row_major(Coord(m, Idx[N])))

    _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, ctx)
    ctx.synchronize()

    if check:
        # Numerical oracle: compare against an fp32-accum naive reference.
        var c_ref_dev = ctx.enqueue_create_buffer[dtype](c_size)
        comptime BX = 16
        comptime BY = 16
        ctx.enqueue_function[naive_matmul_ref_kernel](
            c_ref_dev,
            a_dev,
            b_dev,
            m,
            N,
            K,
            grid_dim=(ceildiv(N, BX), ceildiv(m, BY)),
            block_dim=(BX, BY),
        )
        ctx.synchronize()
        var c_h = ctx.enqueue_create_host_buffer[dtype](c_size)
        var c_ref_h = ctx.enqueue_create_host_buffer[dtype](c_size)
        ctx.enqueue_copy(c_h, c_dev)
        ctx.enqueue_copy(c_ref_h, c_ref_dev)
        ctx.synchronize()
        if not numeric_check(
            c_h.as_span(), c_ref_h.as_span(), atol=2.0, rtol=5e-2
        ):
            raise Error("matmul vs fp32-accum naive mismatch")
        _ = c_ref_dev

    _ = a_dev
    _ = b_dev
    _ = c_dev


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
            print("FUZZ_SPEC idx=", i, "m=", specs[i].m)
        return

    if mode == "single":
        var m = flag_int(args, "--m", 128)
        print("FUZZ_SINGLE m=", m, "N=", N, "K=", K)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(m), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_matmul seed=",
        the_seed,
        "budget=",
        the_budget,
        "N=",
        N,
        "K=",
        K,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
