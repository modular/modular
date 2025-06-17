# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from gpu.host._compile import _compile_code_asm, get_gpu_target
from nn.mha_mask import (
    AndMask,
    CausalMask,
    NullMask,
    SlidingWindowCausalMask,
    TileMaskStatus,
)
from testing import assert_equal, assert_true

from utils.index import Index, IndexList
from utils.numerics import min_or_neg_inf


def test_causal_mask():
    alias type = DType.int32

    print("test_causal_mask")
    var mask = CausalMask()

    # Check mask value.
    # TODO(KERN-782): should be -inf but softmax saturates with NaNs.
    var mask_val = -10000
    var masked_vec = mask.mask(Index(0, 0, 4, 3), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, mask_val, mask_val))

    masked_vec = mask.mask(Index(0, 0, 4, 0), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 2, 3))

    masked_vec = mask.mask(Index(0, 0, 1, 6), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](mask_val))

    # Check tile status.
    assert_true(
        mask.status(Index(4, 4), Index(4, 4)) == TileMaskStatus.PARTIAL_MASK
    )
    assert_true(
        mask.status(Index(0, 2), Index(2, 2)) == TileMaskStatus.FULL_MASK
    )
    assert_true(mask.status(Index(2, 0), Index(2, 2)) == TileMaskStatus.NO_MASK)
    assert_true(mask.status(Index(2, 1), Index(2, 2)) == TileMaskStatus.NO_MASK)
    assert_true(
        mask.status(Index(1, 5), Index(2, 2)) == TileMaskStatus.FULL_MASK
    )
    assert_true(
        mask.status(Index(64, 0), Index(64, 128)) == TileMaskStatus.PARTIAL_MASK
    )
    assert_true(
        mask.status(Index(64, 128), Index(64, 128)) == TileMaskStatus.FULL_MASK
    )
    assert_true(
        mask.status(Index(64, 256), Index(64, 128)) == TileMaskStatus.FULL_MASK
    )
    assert_true(
        mask.status(Index(64, 384), Index(64, 128)) == TileMaskStatus.FULL_MASK
    )


def test_causal_mask_asm():
    """Verify mask comparison is not in 64 bits."""

    print("== test_causal_mask_asm")

    fn kernel(q_idx: UInt32, k_idx: UInt32) -> Float32:
        var mask = CausalMask()
        var vec = mask.mask(
            IndexList[4, element_type = DType.uint32](
                0, 0, Int(q_idx), Int(k_idx)
            ),
            SIMD[DType.float32, 4](0),
        )
        if (
            mask.status(
                Index[dtype = DType.uint32](q_idx, k_idx),
                Index[dtype = DType.uint32](4, 5),
            )
            == TileMaskStatus.PARTIAL_MASK
        ):
            return vec[3]

        return vec[2]

    var asm = _compile_code_asm[kernel, target = get_gpu_target()]()
    assert_true("setp.lt.u64" not in asm)
    assert_true("setp.lt.s64" not in asm)


def test_and_mask():
    alias type = DType.int32

    print("test_and_mask")
    # Or-ing a causal mask with a null mask should result in a causal mask.
    var mask = AndMask[CausalMask(), NullMask()]()

    var masked_vec = mask.mask(Index(0, 0, 4, 3), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 0, 0))

    masked_vec = mask.mask(Index(0, 0, 4, 0), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 2, 3))

    masked_vec = mask.mask(Index(0, 0, 1, 6), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0))

    # Check tile status.
    assert_true(mask.status(Index(4, 4), Index(4, 4)) == TileMaskStatus.NO_MASK)
    assert_true(mask.status(Index(0, 2), Index(2, 2)) == TileMaskStatus.NO_MASK)
    assert_true(mask.status(Index(2, 0), Index(2, 2)) == TileMaskStatus.NO_MASK)

    var mask2 = AndMask[CausalMask(), CausalMask()]()
    assert_true(
        mask2.status(Index(4, 4), Index(4, 4)) == TileMaskStatus.PARTIAL_MASK
    )
    assert_true(
        mask2.status(Index(64, 384), Index(64, 128))
        == TileMaskStatus.FULL_MASK,
        msg=String(
            "lhs = ",
            mask2.status(Index(0, 0), Index(0, 0)),
            " rhs = ",
            TileMaskStatus.FULL_MASK,
        ),
    )


def test_sliding_window_causal_mask():
    print("test_sliding_window_causal_mask")

    alias mask = SlidingWindowCausalMask[3]()

    @always_inline
    def check_status(
        offset: IndexList[2, **_],
        size: __type_of(offset),
        expected: TileMaskStatus,
    ):
        var status = mask.status(offset, size)
        assert_equal(
            status,
            expected,
            msg=String(
                "  ",
                offset,
                ", ",
                size,
                " > ",
                status,
                " (expected: ",
                expected,
                ")",
            ),
        )

        # K > 0 1 2 3 4 5 6 7 8
        # Q v x-----------------x
        # 0 | 1 0 0 0 0 0 0 0 0
        # 1 | 1 1 0 0 0 0 0 0 0
        # 2 | 1 1 1 0 0 0 0 0 0
        # 3 | 0 1 1 1 0 0 0 0 0
        # 4 | 0 0 1 1 1 0 0 0 0
        # 5 | 0 0 0 1 1 1 0 0 0
        # 6 | 0 0 0 0 1 1 1 0 0
        # 7 | 0 0 0 0 0 1 1 1 0
        # 8 | 0 0 0 0 0 0 1 1 1

    check_status(Index(0, 0), Index(4, 4), TileMaskStatus.PARTIAL_MASK)
    check_status(Index(4, 0), Index(4, 4), TileMaskStatus.PARTIAL_MASK)

    check_status(Index(2, 1), Index(2, 2), TileMaskStatus.NO_MASK)
    check_status(Index(3, 1), Index(1, 3), TileMaskStatus.NO_MASK)
    check_status(Index(3, 3), Index(3, 1), TileMaskStatus.NO_MASK)

    check_status(Index(0, 4), Index(4, 4), TileMaskStatus.FULL_MASK)
    check_status(Index(4, 0), Index(4, 2), TileMaskStatus.FULL_MASK)
    check_status(Index(1, 4), Index(3, 2), TileMaskStatus.FULL_MASK)


def test_sliding_window_causal_mask_asm():
    """Verify mask comparison is not in 64 bits."""

    print("== test_sliding_window_causal_mask_asm")

    fn kernel(q_idx: UInt32, k_idx: UInt32) -> Float32:
        var mask = SlidingWindowCausalMask[8]()
        var vec = mask.mask(
            IndexList[4, element_type = DType.uint32](
                0, 0, Int(q_idx), Int(k_idx)
            ),
            SIMD[DType.float32, 4](0),
        )
        if (
            mask.status(
                Index[dtype = DType.uint32](q_idx, k_idx),
                Index[dtype = DType.uint32](64, 32),
            )
            == TileMaskStatus.PARTIAL_MASK
        ):
            return vec[3]

        return vec[2]

    var asm = _compile_code_asm[kernel, target = get_gpu_target()]()
    print(asm)
    assert_true("setp.lt.u64" not in asm)
    assert_true("setp.lt.s64" not in asm)


def main():
    test_causal_mask()
    test_causal_mask_asm()
    test_and_mask()
    test_sliding_window_causal_mask()
    test_sliding_window_causal_mask_asm()
