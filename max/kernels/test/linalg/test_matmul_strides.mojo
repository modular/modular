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

from layout import Coord, Idx, MixedLayout, TileTensor, row_major
from linalg.matmul.cpu import matmul
from linalg.matmul.cpu.apple_accelerate import use_apple_accelerate_lib
from linalg.matmul.cpu.default import Inner_matmul_default
from linalg.utils import GemmShape, InnerKernelID, select_inner_kernel
from std.memory.alloc import Alignment, Layout, alloc
from std.testing import assert_equal


def check_strides[
    dtype: DType, transpose_b: Bool = False
](
    m: Int,
    n: Int,
    k: Int,
    pad_a: Int = 0,
    pad_b: Int = 0,
    pad_c: Int = 0,
    num_threads: Int = 1,
) raises:
    var as_ = k + pad_a
    var bs = (k if transpose_b else n) + pad_b
    var cs = n + pad_c
    comptime alignment = Alignment.of_bytes[64]()
    var aa = alloc(
        Layout[Scalar[dtype], alignment=alignment](length=m * as_)
    ).into_managed()
    var ba = alloc(
        Layout[Scalar[dtype], alignment=alignment](
            length=(n if transpose_b else k) * bs
        )
    ).into_managed()
    var ca = alloc(
        Layout[Scalar[dtype], alignment=alignment](length=m * cs)
    ).into_managed()
    var ap = aa.unsafe_ptr()
    var bp = ba.unsafe_ptr()
    var cp = ca.unsafe_ptr()
    for i in range(m * as_):
        ap[unsafe_offset=i] = 31
    for i in range((n if transpose_b else k) * bs):
        bp[unsafe_offset=i] = 37
    for i in range(m * cs):
        cp[unsafe_offset=i] = -99
    for i in range(m):
        for p in range(k):
            ap[unsafe_offset=i * as_ + p] = Scalar[dtype](
                (3 * i + 2 * p) % 7 - 3
            )
    for p in range(k):
        for j in range(n):
            var offset = j * bs + p if transpose_b else p * bs + j
            bp[unsafe_offset=offset] = Scalar[dtype]((2 * p + 3 * j) % 7 - 3)
    var a = TileTensor(ap, MixedLayout(Coord(m, k), Coord(as_, Idx[1])))
    var b = TileTensor(
        bp,
        MixedLayout(
            Coord(n, k) if transpose_b else Coord(k, n), Coord(bs, Idx[1])
        ),
    )
    var c = TileTensor(cp, MixedLayout(Coord(m, n), Coord(cs, Idx[1])))
    matmul[transpose_b=transpose_b](c, a.as_imm(), b.as_imm(), 0, num_threads)
    for i in range(m):
        for j in range(cs):
            var expected = Float32(-99)
            if j < n:
                expected = 0
                for p in range(k):
                    var offset = j * bs + p if transpose_b else p * bs + j
                    expected += (
                        ap[unsafe_offset=i * as_ + p].cast[.float32]()
                        * bp[unsafe_offset=offset].cast[.float32]()
                    )
            assert_equal(cp[unsafe_offset=i * cs + j], expected.cast[dtype]())


def check_split_k() raises:
    # Explicit K tiles and M/N offsets exercise C reload without scheduler heuristics.
    var a: List[Float32] = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    var c = List[Float32](capacity=32)
    for _ in range(32):
        c.append(-99)
    var packed: List[Float32] = [0, 0, 0, 0]
    var av = TileTensor(a.unsafe_ptr(), row_major(Coord(4, 4)))
    var cv = TileTensor(
        c.unsafe_ptr(), MixedLayout(Coord(4, 5), Coord(8, Idx[1]))
    )
    var bv = TileTensor(packed.unsafe_ptr(), row_major(Coord(1, 2, 2)))
    for start_k in range(0, 4, 2):
        for p in range(2):
            for j in range(2):
                packed[p * 2 + j] = Float32(start_k + p + j + 1)
        Inner_matmul_default().__inner_matmul__[2, 2, 1](
            cv,
            av.as_imm(),
            bv.as_imm(),
            GemmShape(1, 1, start_k),
            GemmShape(3, 3, 4),
            (2, 2),
            False,
        )
    for i in range(32):
        var row = i // 8
        var col = i % 8
        var expected = Float32(-99)
        if 1 <= row < 3 and 1 <= col < 3:
            expected = Float32((row + 1) * (10 + 4 * (col - 1)))
        assert_equal(c[i], expected)


def main() raises:
    comptime if select_inner_kernel[
        .float32, .float32, .float32
    ]() == InnerKernelID.DEFAULT and not use_apple_accelerate_lib[
        .float32, .float32, .float32
    ]():
        check_strides[.float32](5, 7, 9)
        check_strides[.float32](5, 7, 9, pad_a=3)
        check_strides[.float32](5, 7, 9, pad_c=4)
        check_strides[.float32](5, 7, 9, pad_a=3, pad_c=4)
        check_strides[.float32](5, 7, 9, pad_b=5)
        check_strides[.float32, True](5, 7, 9, 3, 5, 4)
        check_strides[.float32](65, 67, 257, pad_a=3, pad_c=4, num_threads=4)
        check_strides[.bfloat16](5, 7, 9, pad_a=3, pad_c=4)
        check_strides[.float16, True](5, 7, 9, pad_a=3, pad_c=4)
        check_split_k()
