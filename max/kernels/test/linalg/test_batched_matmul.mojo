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

from layout import Coord, TileTensor, row_major
from linalg.bmm import batched_matmul
from std.memory.alloc import Alignment, Layout, alloc
from std.testing import assert_equal


comptime dtype = DType.bfloat16
comptime alignment = Alignment.of_bytes[64]()


def check_bf16_matmul[
    transpose_b: Bool, nonuniform: Bool = False
](batch: Int, m: Int, n: Int, k: Int) raises:
    var a_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](length=batch * m * k)
    ).into_managed()
    var b_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](length=batch * k * n)
    ).into_managed()
    var c_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](length=batch * m * n)
    ).into_managed()
    var a = TileTensor(a_alloc.unsafe_ptr(), row_major(batch, m, k))
    var b = TileTensor(
        b_alloc.unsafe_ptr(),
        row_major(batch, n, k) if transpose_b else row_major(batch, k, n),
    )
    var c = TileTensor(c_alloc.unsafe_ptr(), row_major(batch, m, n))
    comptime if nonuniform:
        # Signed multiples of 1/16 distinguish rows/columns and are exact in BF16.
        for i in range(batch * m * k):
            a_alloc.unsafe_ptr()[unsafe_offset=i] = (
                Float32((i * 11 + 17) % 29 - 14) / 16
            ).cast[dtype]()
        for i in range(batch * k * n):
            b_alloc.unsafe_ptr()[unsafe_offset=i] = (
                Float32((i * 13 + 23) % 31 - 15) / 16
            ).cast[dtype]()
    else:
        _ = a.fill(1)
        _ = b.fill(1)
    _ = c.fill(-1)
    batched_matmul[transpose_b=transpose_b, target="cpu"](
        c, a.as_imm(), b.as_imm()
    )

    for i in range(batch * m * n):
        var reference = Float32(k)
        comptime if nonuniform:
            var batch_idx = i // (m * n)
            var row = (i // n) % m
            var col = i % n
            reference = 0
            for p in range(k):
                var av = a_alloc.unsafe_ptr()[
                    unsafe_offset=(batch_idx * m + row) * k + p
                ].cast[.float32]()
                var b_offset = (batch_idx * k + p) * n + col
                comptime if transpose_b:
                    b_offset = (batch_idx * n + col) * k + p
                var bv = b_alloc.unsafe_ptr()[unsafe_offset=b_offset].cast[
                    .float32
                ]()
                reference += av * bv
            # At K=513, partial sums are multiples of 1/256 with numerator
            # bounded by 513*14*15 < 2**24: FP32 accumulation is exact.
        assert_equal(
            c_alloc.unsafe_ptr()[unsafe_offset=i], reference.cast[dtype]()
        )


def callback_values[
    t: DType, width: SIMDLength
](idx: Coord, val: SIMD[t, width]) -> SIMD[t, width]:
    comptime assert t == dtype
    comptime assert idx.rank == 4
    var result = SIMD[t, width](0)
    comptime for lane in range(width):
        var encoded = (
            Int(idx[0].value()) * 96
            + Int(idx[1].value()) * 24
            + Int(idx[2].value()) * 8
            + Int(idx[3].value())
            + lane
        )
        # Dot 257 must round to BF16 256 before the callback subtracts 256.
        result[lane] = val[lane] - Scalar[t](256) + Scalar[t](encoded)
    return result


def check_callbacks[
    transpose_b: Bool, use_compute: Bool
](b0: Int, b1: Int) raises:
    var batch = b0 * b1
    var m = 65
    var n = 67
    var k = 257
    var a_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](length=batch * m * k)
    ).into_managed()
    var b_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](length=batch * k * n)
    ).into_managed()
    var c_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](length=batch * m * n)
    ).into_managed()
    var external_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](length=batch * m * n)
    ).into_managed()
    var a = TileTensor(a_alloc.unsafe_ptr(), row_major(b0, b1, m, k))
    var b = TileTensor(
        b_alloc.unsafe_ptr(),
        row_major(b0, b1, n, k) if transpose_b else row_major(b0, b1, k, n),
    )
    var c = TileTensor(c_alloc.unsafe_ptr(), row_major(b0, b1, m, n))
    var external = TileTensor(
        external_alloc.unsafe_ptr(), row_major(b0, b1, m, n)
    )
    _ = a.fill(1)
    _ = b.fill(1)
    _ = c.fill(-1)
    _ = external.fill(-1)

    @inline(.always)
    def epilogue[
        t: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[t, width]) {var external}:
        external.store[alignment=alignment](
            idx, callback_values(idx, val).cast[dtype]()
        )

    @inline(.always)
    def compute[
        t: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[t, width]) -> SIMD[t, width]:
        return callback_values(idx, val)

    comptime if use_compute:
        batched_matmul[transpose_b=transpose_b, target="cpu"](
            c, a.as_imm(), b.as_imm(), compute
        )
    else:
        batched_matmul[transpose_b=transpose_b, target="cpu"](
            c, a.as_imm(), b.as_imm(), epilogue
        )

    for i in range(batch * m * n):
        var col = i % n
        var row = (i // n) % m
        var batch1 = (i // (m * n)) % b1
        var batch0 = i // (b1 * m * n)
        var expected = Scalar[dtype](batch0 * 96 + batch1 * 24 + row * 8 + col)
        comptime if use_compute:
            assert_equal(c_alloc.unsafe_ptr()[unsafe_offset=i], expected)
        else:
            assert_equal(external_alloc.unsafe_ptr()[unsafe_offset=i], expected)
            assert_equal(
                c_alloc.unsafe_ptr()[unsafe_offset=i], Scalar[dtype](-1)
            )


def main() raises:
    # Long reductions cross K cache tiles on AVX2. A BF16 running sum of ones
    # stops at 256; the expected 8192 is exactly representable. Odd M/N test tails.
    check_bf16_matmul[transpose_b=False](1, 3, 5, 8192)
    check_bf16_matmul[transpose_b=True](1, 3, 5, 8192)
    # Runtime batch=1 also covers the BMM path used by eager rank-2 matmul.
    # On a 20-worker AVX2 CPU, batch=1 partitions M and batch=6 partitions N.
    # Assertions are independent of task count; N spans full SIMD vectors + tails.
    for batch in range(1, 7, 5):
        check_bf16_matmul[transpose_b=False, nonuniform=True](
            batch, 65, 67, 513
        )
        check_bf16_matmul[transpose_b=True, nonuniform=True](batch, 65, 67, 513)
        var b0 = 1 if batch == 1 else 2
        var b1 = 1 if batch == 1 else 3
        check_callbacks[transpose_b=False, use_compute=False](b0, b1)
        check_callbacks[transpose_b=True, use_compute=True](b0, b1)
