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
# mojo build --debug-level=full --mcmodel=medium --large-data-threshold=1048576
# to build this file if running into linking issues with large PTX kernels.

from std.random import random_float64, random_si64
from std.math import abs, ceildiv

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu, matmul_kernel_naive
from std.utils import IndexList


@inline(.always)
def epilogue_test_fn[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](
    idx: IndexList[2],
    dim_space: IndexList[2],
    val: SIMD[dtype, width],
) -> SIMD[
    dtype, width
]:
    var bias = SIMD[dtype, width](0)

    comptime for i in range(width):
        bias[i] = (
            0.5
            + Float64(idx[0] + idx[1] + i)
            / Float64(dim_space[0] + dim_space[1])
        ).cast[dtype]()

    return val + bias


def test[
    in_type: DType,
    out_type: DType,
    transpose_b: Bool,
    M: Optional[Int],
    N: Optional[Int],
    K: Optional[Int],
](ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    comptime assert Bool(N) and Bool(
        K
    ), "This test currently requires static N and K."

    print(t"{m} x {n} x {k} transpose_b {transpose_b}")

    var a_size = m * k
    var b_size = k * n
    var c_size = m * n

    # Host buffers
    var a_host = ctx.enqueue_create_host_buffer[in_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[in_type](b_size)
    var c_host = ctx.enqueue_create_host_buffer[out_type](c_size)
    var c_host_ref = ctx.enqueue_create_host_buffer[out_type](c_size)

    comptime rand_min = -100
    comptime rand_max = 100

    for i in range(a_size):
        a_host[i] = random_si64(rand_min, rand_max).cast[in_type]()

    for i in range(b_size):
        b_host[i] = random_si64(rand_min, rand_max).cast[in_type]()

    for i in range(c_size):
        c_host[i] = 0
        c_host_ref[i] = 0

    # Device buffers
    var a_dev = ctx.enqueue_create_buffer[in_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[in_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[out_type](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[out_type](c_size)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)
    ctx.enqueue_copy(c_ref_dev, c_host_ref)

    # TileTensors for _matmul_gpu
    comptime static_b_dim0 = N if transpose_b else K
    comptime static_b_dim1 = K if transpose_b else N
    var a_tensor = TileTensor(
        a_dev,
        row_major(m, Idx[K.value()]),
    )
    var b_tensor = TileTensor(
        b_dev,
        row_major(Idx[static_b_dim0.value()], Idx[static_b_dim1.value()]),
    )
    var c_tensor = TileTensor(
        c_dev,
        row_major(m, Idx[N.value()]),
    )

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
        c_tensor,
        a_tensor.as_imm(),
        b_tensor.as_imm(),
        ctx,
    )

    ctx.enqueue_copy(c_host, c_dev)

    var c_tt = TileTensor(
        c_ref_dev,
        row_major(m, n),
    )
    var a_tt = TileTensor[mut=False](a_dev, row_major(m, k))
    var b_shape_0 = n if transpose_b else k
    var b_shape_1 = k if transpose_b else n
    var b_tt = TileTensor[mut=False](b_dev, row_major(b_shape_0, b_shape_1))

    comptime BLOCK_DIM = 16
    comptime naive_kernel = matmul_kernel_naive[
        out_type,
        in_type,
        in_type,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
        transpose_b=transpose_b,
        c_engine=type_of(c_tt).Engine,
        a_engine=type_of(a_tt).Engine,
        b_engine=type_of(b_tt).Engine,
    ]
    ctx.enqueue_function[naive_kernel](
        c_tt,
        a_tt,
        b_tt,
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )
    ctx.synchronize()

    ctx.enqueue_copy(c_host_ref, c_ref_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(c_size):
        if c_host[i] != c_host_ref[i]:
            errors += 1

    print("errors", errors)
    if errors > 0:
        raise "GEMV failed: exact mismatch"


def test_transpose_b_all_rows[
    in_type: DType, out_type: DType, N: Int, K: Int
](ctx: DeviceContext, m: Int) raises:
    """Transpose_b matmul through the full dispatch, every output row checked
    against an fp64 host reference.

    The output is prefilled with a sentinel so a row the dispatch fails to
    write fails deterministically instead of reading stale allocator memory.
    N == 1 with M > 1 selects GEMV_KERNEL, whose transpose_b launch used to
    swap operands and extents as if this were a GEVM (M == 1), leaving rows
    1..M-1 unwritten.
    """
    print(t"transpose_b all-rows: {m} x {N} x {K} {in_type} -> {out_type}")

    var a_size = m * K
    var b_size = N * K
    var c_size = m * N

    var a_host = ctx.enqueue_create_host_buffer[in_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[in_type](b_size)
    var c_host = ctx.enqueue_create_host_buffer[out_type](c_size)

    for i in range(a_size):
        a_host[i] = random_float64(min=-1.0, max=1.0).cast[in_type]()

    for i in range(b_size):
        b_host[i] = random_float64(min=-1.0, max=1.0).cast[in_type]()

    for i in range(c_size):
        c_host[i] = Float32(1e6).cast[out_type]()

    var a_dev = ctx.enqueue_create_buffer[in_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[in_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[out_type](c_size)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)

    var a_tensor = TileTensor(a_dev, row_major(m, Idx[K]))
    var b_tensor = TileTensor(b_dev, row_major(Idx[N], Idx[K]))
    var c_tensor = TileTensor(c_dev, row_major(m, Idx[N]))

    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        c_tensor, a_tensor.as_imm(), b_tensor.as_imm(), ctx
    )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    for r in range(m):
        for j in range(N):
            var expected = Float64(0)
            for kk in range(K):
                expected += (
                    a_host[r * K + kk].cast[.float64]()
                    * b_host[j * K + kk].cast[.float64]()
                )
            # bf16 outputs round to ~2^-8 relative and the fp32 accumulator
            # orders the K reduction differently from this sequential sum,
            # so allow 1% relative; the 1e6 sentinel is far outside that.
            var got = c_host[r * N + j].cast[.float64]()
            if not (abs(got - expected) <= 1e-2 + 1e-2 * abs(expected)):
                raise Error(
                    t"row {r} col {j}: got {got}, expected {expected} "
                    t"(m={m}, n={N}, k={K})"
                )


def test_transpose_b_n1_value_fns[
    in_type: DType, out_type: DType, N: Int, K: Int
](ctx: DeviceContext, m: Int) raises:
    """N == 1 transpose_b through the epilogue and compute-fn matmul entries.

    Both GEMV_KERNEL value-fn launches must deliver one rank-2 (row, col)
    coordinate per output element -- the row covering every output row, not
    just row 0 -- with that element's dot product as the value. The epilogue
    records the coordinates it receives; the compute fn folds them into the
    stored value so a wrong coordinate fails the value check.
    """
    print(t"transpose_b n1 value fns: {m} x {N} x {K} {in_type} -> {out_type}")

    var a_size = m * K
    var b_size = N * K
    var c_size = m * N

    var a_host = ctx.enqueue_create_host_buffer[in_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[in_type](b_size)
    for i in range(a_size):
        a_host[i] = random_float64(min=-1.0, max=1.0).cast[in_type]()

    for i in range(b_size):
        b_host[i] = random_float64(min=-1.0, max=1.0).cast[in_type]()

    var a_dev = ctx.enqueue_create_buffer[in_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[in_type](b_size)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    @inline(.always)
    def dot_ref(r: Int, j: Int) {imm a_host, imm b_host} -> Float64:
        var acc = Float64(0)
        for kk in range(K):
            acc += (
                a_host[r * K + kk].cast[.float64]()
                * b_host[j * K + kk].cast[.float64]()
            )
        return acc

    var a = TileTensor(a_dev, row_major(m, Idx[K])).as_imm()
    var b = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()

    # Epilogue run: the closure owns every store, so `c` is a shape-only tile
    # over a null pointer (no path may read or write it). The closure records
    # both the value and the coordinate it is handed.
    var out_dev = ctx.enqueue_create_buffer[out_type](c_size)
    var rows_dev = ctx.enqueue_create_buffer[.int32](c_size)
    var cols_dev = ctx.enqueue_create_buffer[.int32](c_size)
    var out_host = ctx.enqueue_create_host_buffer[out_type](c_size)
    var rows_host = ctx.enqueue_create_host_buffer[.int32](c_size)
    var cols_host = ctx.enqueue_create_host_buffer[.int32](c_size)
    for i in range(c_size):
        out_host[i] = Float32(1e6).cast[out_type]()
        rows_host[i] = -1
        cols_host[i] = -1
    ctx.enqueue_copy(out_dev, out_host)
    ctx.enqueue_copy(rows_dev, rows_host)
    ctx.enqueue_copy(cols_dev, cols_host)

    var out = TileTensor(out_dev.unsafe_ptr(), row_major(m, Idx[N]))
    var rows = TileTensor(rows_dev.unsafe_ptr(), row_major(m, Idx[N]))
    var cols = TileTensor(cols_dev.unsafe_ptr(), row_major(m, Idx[N]))

    @inline(.always)
    def record_epilogue[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var out, var rows, var cols}:
        comptime assert idx.rank == 2, "the matmul epilogue contract is rank 2"
        out.store(idx, val.cast[out_type]())
        rows.store(idx, SIMD[.int32, 1](Int32(idx[0])))
        cols.store(idx, SIMD[.int32, 1](Int32(idx[1])))

    var null_c_ptr: Optional[
        MutPointer[Scalar[out_type], MutUntrackedOrigin]
    ] = None
    var c_shape_only = TileTensor(
        null_c_ptr._unsafe_nullable(), row_major(m, Idx[N])
    )

    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        c_shape_only, a, b, record_epilogue, ctx
    )

    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(rows_host, rows_dev)
    ctx.enqueue_copy(cols_host, cols_dev)
    ctx.synchronize()

    for r in range(m):
        for j in range(N):
            var expected = dot_ref(r, j)
            var flat = r * N + j
            var got = out_host[flat].cast[.float64]()
            if not (abs(got - expected) <= 1e-2 + 1e-2 * abs(expected)):
                raise Error(
                    t"epilogue row {r} col {j}: got {got}, expected "
                    t"{expected} (m={m}, n={N}, k={K})"
                )
            if Int(rows_host[flat]) != r or Int(cols_host[flat]) != j:
                raise Error(
                    t"epilogue row {r} col {j}: got coord "
                    t"({rows_host[flat]}, {cols_host[flat]}), expected "
                    t"({r}, {j})"
                )

    # Compute-fn run: the kernel stores, so give it a real output buffer.
    var c_dev = ctx.enqueue_create_buffer[out_type](c_size)
    var c_host = ctx.enqueue_create_host_buffer[out_type](c_size)
    for i in range(c_size):
        c_host[i] = Float32(1e6).cast[out_type]()
    ctx.enqueue_copy(c_dev, c_host)

    @inline(.always)
    def coord_encode_fn[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
        comptime assert idx.rank == 2, "the matmul epilogue contract is rank 2"
        return val + SIMD[_dtype, width](Int(idx[0]) * 1000 + Int(idx[1]))

    var c = TileTensor(c_dev, row_major(m, Idx[N]))
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        c, a, b, coord_encode_fn, ctx
    )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    for r in range(m):
        for j in range(N):
            var marked = dot_ref(r, j) + Float64(1000 * r + j)
            var got = c_host[r * N + j].cast[.float64]()
            if not (abs(got - marked) <= 1e-2 + 1e-2 * abs(marked)):
                raise Error(
                    t"compute fn row {r} col {j}: got {got}, expected "
                    t"{marked} (m={m}, n={N}, k={K})"
                )


def test_gemv_split_k[in_type: DType](ctx: DeviceContext) raises:
    """Test GEMV_SPLIT_K path: M=1, transpose_b=True, K % simd_width == 0."""
    print("=== Testing GEMV_SPLIT_K with", in_type, "===")

    comptime out_type = DType.float32 if in_type == DType.bfloat16 else DType.bfloat16

    # Basic GEMV_SPLIT_K
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(4096),
        K=Int(4096),
    ](ctx, 1, 4096, 4096)

    # N % TILE_N != 0 (bounds checking)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(75837),
        K=Int(5120),
    ](ctx, 1, 75837, 5120)

    # Large K (tests 512-thread / 256-thread dispatch + unroll)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(16384),
        K=Int(16384),
    ](ctx, 1, 16384, 16384)

    # Small K (tests 64-thread dispatch)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(16384),
        K=Int(1024),
    ](ctx, 1, 16384, 1024)

    # Mid K (tests 128-thread dispatch)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(16384),
        K=Int(2048),
    ](ctx, 1, 16384, 2048)


def main() raises:
    with DeviceContext() as ctx:
        # GEMV_SPLIT_K tests for bf16 and fp8
        test_gemv_split_k[.bfloat16](ctx)
        test_gemv_split_k[.float8_e4m3fn](ctx)

        # GEMV_KERNEL with N == 1 and transpose_b: M == 1 keeps the swapped
        # GEVM launch, M > 1 must write every row (fp32 at both K points).
        test_transpose_b_all_rows[.float32, .float32, 1, 32](ctx, 1)
        test_transpose_b_all_rows[.float32, .float32, 1, 32](ctx, 2)
        test_transpose_b_all_rows[.float32, .float32, 1, 32](ctx, 6)
        test_transpose_b_all_rows[.float32, .float32, 1, 32](ctx, 18)
        test_transpose_b_all_rows[.float32, .float32, 1, 32](ctx, 129)
        test_transpose_b_all_rows[.float32, .float32, 1, 7168](ctx, 1)
        test_transpose_b_all_rows[.float32, .float32, 1, 7168](ctx, 2)
        test_transpose_b_all_rows[.float32, .float32, 1, 7168](ctx, 6)
        test_transpose_b_all_rows[.float32, .float32, 1, 7168](ctx, 18)
        test_transpose_b_all_rows[.float32, .float32, 1, 7168](ctx, 129)

        # bf16 with K % simd_width != 0: the other N == 1 arm that selects
        # GEMV_KERNEL.
        test_transpose_b_all_rows[.bfloat16, .float32, 1, 4095](ctx, 2)
        test_transpose_b_all_rows[.bfloat16, .float32, 1, 4095](ctx, 18)

        # N == 2 control: the GEMV path is skipped for a wider N, so this
        # checks the harness against a non-GEMV route.
        test_transpose_b_all_rows[.float32, .float32, 2, 32](ctx, 6)

        # The same N == 1 M > 1 shapes through the epilogue and compute-fn
        # entries: the value fns must see one rank-2 coordinate per output
        # element. The bf16 case also reaches the GEMV arm on B200, where
        # the SM100 fp32 dispatch intercepts the f32 shape.
        test_transpose_b_n1_value_fns[.float32, .float32, 1, 32](ctx, 6)
        test_transpose_b_n1_value_fns[.bfloat16, .float32, 1, 4095](ctx, 6)

        # BF16-only tests for other GEMV paths (FP8 not supported here)

        # GEMV_KERNEL_VECTOR: M = 1, N wider than the split-K grid limit and
        # K shallow enough that the kernel packs several rows per warp.
        test[
            in_type=.bfloat16,
            out_type=.bfloat16,
            transpose_b=True,
            M=None,
            N=Int(262144),
            K=Int(256),
        ](ctx, 1, 262144, 256)

        # GEMV_KERNEL_VECTOR: N = 1, K % simd_width == 0, transpose_b = False
        test[
            in_type=.bfloat16,
            out_type=.float32,
            transpose_b=False,
            M=None,
            N=Int(1),
            K=Int(4096),
        ](ctx, 4096, 1, 4096)

        # GEMV_KERNEL_VECTOR: N = 1, K % simd_width == 0, transpose_b = True
        test[
            in_type=.bfloat16,
            out_type=.bfloat16,
            transpose_b=True,
            M=None,
            N=Int(1),
            K=Int(13824),
        ](ctx, 5120, 1, 13824)

        # GEMV_KERNEL: M = 1, K % simd_width !=0, transpose_b = True
        test[
            in_type=.bfloat16,
            out_type=.float32,
            transpose_b=True,
            M=None,
            N=Int(4096),
            K=Int(4095),
        ](ctx, 1, 4096, 4095)

        # GEMV_KERNEL: N = 1, K % simd_width !=0, transpose_b = False
        test[
            in_type=.bfloat16,
            out_type=.float32,
            transpose_b=False,
            M=None,
            N=Int(1),
            K=Int(4095),
        ](ctx, 4096, 1, 4095)

        # matmul_naive: M = 1, K % WARP_SIZE != 0, transpose_b = False
        test[
            in_type=.bfloat16,
            out_type=.float32,
            transpose_b=False,
            M=None,
            N=Int(4096),
            K=Int(4095),
        ](ctx, 1, 4096, 4095)
