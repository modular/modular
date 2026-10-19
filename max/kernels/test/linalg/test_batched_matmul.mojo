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

from layout import Coord, TensorLayout, TileTensor, row_major
from linalg.bmm import batched_matmul
from std.memory.alloc import Alignment, Layout, alloc
from std.testing import assert_equal


def callback_values[
    dtype: DType, width: SIMDLength
](idx: Coord, val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime assert dtype == DType.bfloat16
    comptime assert idx.rank == 4
    # The compact 2x3x3x5 fixture encodes coordinates below 256, where all
    # integers are exactly representable in BF16.
    var result = SIMD[dtype, width](0)
    comptime for lane in range(width):
        var encoded = (
            Int(idx[0].value()) * 96
            + Int(idx[1].value()) * 24
            + Int(idx[2].value()) * 8
            + Int(idx[3].value())
            + lane
        )
        result[lane] = val[lane] - Scalar[dtype](256) + Scalar[dtype](encoded)
    return result


def check_bf16_accumulation[
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    CLayout: TensorLayout,
    //,
    transpose_b: Bool,
    # 0: plain output; 1: external epilogue output; 2: compute callback.
    callback_mode: Int = 0,
    nonuniform: Bool = False,
](
    a_layout: ALayout, b_layout: BLayout, c_layout: CLayout, expected: Int = 512
) raises:
    # The default fixture sums 512 ones, exactly representable in fp32/bf16.
    # A running bf16 sum stops increasing at 256 (issue #7176).
    comptime dtype = DType.bfloat16
    comptime assert not nonuniform or callback_mode == 0
    comptime alignment = Alignment.of_bytes[64]()
    var a_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](count=a_layout.product())
    ).into_managed()
    var b_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](count=b_layout.product())
    ).into_managed()
    var c_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](count=c_layout.product())
    ).into_managed()
    var external_alloc = alloc(
        Layout[Scalar[dtype], alignment=alignment](count=c_layout.product())
    ).into_managed()
    var a = TileTensor(a_alloc.unsafe_ptr(), a_layout)
    var b = TileTensor(b_alloc.unsafe_ptr(), b_layout)
    var c = TileTensor(c_alloc.unsafe_ptr(), c_layout)
    var external = TileTensor(external_alloc.unsafe_ptr(), c_layout)
    comptime if nonuniform:
        # Signed multiples of 1/16 are exactly representable in BF16. Unlike
        # ones, this grid distinguishes rows/columns and catches bad A/B slices.
        for i in range(a_layout.product()):
            a_alloc.unsafe_ptr()[unsafe_offset=i] = (
                Float32((i * 11 + 17) % 29 - 14) / 16
            ).cast[dtype]()
        for i in range(b_layout.product()):
            b_alloc.unsafe_ptr()[unsafe_offset=i] = (
                Float32((i * 13 + 23) % 31 - 15) / 16
            ).cast[dtype]()
    else:
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

    comptime if callback_mode == 1:
        batched_matmul[transpose_b=transpose_b, target="cpu"](
            c, a.as_imm(), b.as_imm(), epilogue
        )
    elif callback_mode == 2:
        batched_matmul[transpose_b=transpose_b, target="cpu"](
            c, a.as_imm(), b.as_imm(), compute
        )
    else:
        batched_matmul[transpose_b=transpose_b, target="cpu"](
            c, a.as_imm(), b.as_imm()
        )

    for i in range(c_layout.product()):
        comptime if nonuniform:
            var m = Int(c.dim[c.rank - 2]())
            var n = Int(c.dim[c.rank - 1]())
            var k = Int(a.dim[a.rank - 1]())
            var batch = i // (m * n)
            var row = (i // n) % m
            var col = i % n
            var reference = Float32(0)
            for p in range(k):
                var av = a_alloc.unsafe_ptr()[
                    unsafe_offset=(batch * m + row) * k + p
                ].cast[.float32]()
                var b_offset = (
                    (batch * n + col) * k
                    + p if transpose_b else (batch * k + p) * n
                    + col
                )
                var bv = b_alloc.unsafe_ptr()[unsafe_offset=b_offset].cast[
                    .float32
                ]()
                reference += av * bv
            # At K=513, partial sums are multiples of 1/256 with numerator
            # bounded by 513*14*15 < 2**24: FP32 accumulation is exact.
            assert_equal(
                c_alloc.unsafe_ptr()[unsafe_offset=i], reference.cast[dtype]()
            )
        elif callback_mode == 0:
            assert_equal(
                c_alloc.unsafe_ptr()[unsafe_offset=i], Scalar[dtype](expected)
            )
        else:
            var n = Int(c.dim[3]())
            var m = Int(c.dim[2]())
            var b1 = Int(c.dim[1]())
            var col = i % n
            var row = (i // n) % m
            var batch1 = (i // (m * n)) % b1
            var batch0 = i // (b1 * m * n)
            var encoded = Scalar[dtype](
                batch0 * 96 + batch1 * 24 + row * 8 + col
            )
            comptime if callback_mode == 1:
                assert_equal(
                    external_alloc.unsafe_ptr()[unsafe_offset=i], encoded
                )
                assert_equal(
                    c_alloc.unsafe_ptr()[unsafe_offset=i], Scalar[dtype](-1)
                )
            else:
                assert_equal(c_alloc.unsafe_ptr()[unsafe_offset=i], encoded)


def main() raises:
    # Runtime dimensions exercise the same CPU path as the symbolic rank-3
    # graph used by eager matmul, including a rank-2 operand's batch of one.
    # Odd M/N also exercise partial output tiles.
    for batch in range(1, 3):
        check_bf16_accumulation[transpose_b=False](
            row_major(batch, 3, 512),
            row_major(batch, 512, 5),
            row_major(batch, 3, 5),
        )
        check_bf16_accumulation[transpose_b=True](
            row_major(batch, 3, 512),
            row_major(batch, 5, 512),
            row_major(batch, 3, 5),
        )

    check_bf16_accumulation[transpose_b=False](
        row_major(2, 3, 3, 512),
        row_major(2, 3, 512, 5),
        row_major(2, 3, 3, 5),
    )
    check_bf16_accumulation[transpose_b=True](
        row_major(2, 3, 3, 512),
        row_major(2, 3, 5, 512),
        row_major(2, 3, 3, 5),
    )
    # Large K crosses cache tiles on AVX2; the total is exactly representable.
    check_bf16_accumulation[transpose_b=False](
        row_major(1, 3, 8192),
        row_major(1, 8192, 5),
        row_major(1, 3, 5),
        8192,
    )
    check_bf16_accumulation[transpose_b=True](
        row_major(1, 3, 8192),
        row_major(1, 5, 8192),
        row_major(1, 3, 5),
        8192,
    )
    # Dot 257 casts to BF16 256 before the callback subtracts 256. Both callback
    # kinds must receive BF16 values and original rank-4 batch coordinates.
    check_bf16_accumulation[transpose_b=False, callback_mode=1](
        row_major(2, 3, 3, 257),
        row_major(2, 3, 257, 5),
        row_major(2, 3, 3, 5),
    )
    check_bf16_accumulation[transpose_b=True, callback_mode=2](
        row_major(2, 3, 3, 257),
        row_major(2, 3, 5, 257),
        row_major(2, 3, 3, 5),
    )

    # On a 20-worker AVX2 CPU, batch=1 partitions M and batch=6 partitions N.
    # Keep the result checks independent of the machine's task count. Odd
    # dimensions cover tails and noncontiguous B column slices; N also spans
    # full SIMD vectors in both callback variants.
    for batch in range(1, 7, 5):
        check_bf16_accumulation[transpose_b=False, nonuniform=True](
            row_major(batch, 65, 513),
            row_major(batch, 513, 67),
            row_major(batch, 65, 67),
        )
        check_bf16_accumulation[transpose_b=True, nonuniform=True](
            row_major(batch, 65, 513),
            row_major(batch, 67, 513),
            row_major(batch, 65, 67),
        )
        var b0 = 1 if batch == 1 else 2
        var b1 = 1 if batch == 1 else 3
        check_bf16_accumulation[transpose_b=False, callback_mode=1](
            row_major(b0, b1, 65, 257),
            row_major(b0, b1, 257, 67),
            row_major(b0, b1, 65, 67),
        )
        check_bf16_accumulation[transpose_b=True, callback_mode=2](
            row_major(b0, b1, 65, 257),
            row_major(b0, b1, 67, 257),
            row_major(b0, b1, 65, 67),
        )
