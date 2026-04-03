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

from layout import Layout, LayoutTensor
from nn.attention.mha_mask import (
    MASK_VALUE,
    CausalPaddingMask,
    TileMaskStatus,
)
from std.testing import assert_equal

from std.utils.index import Index


def test_causal_padding_mask_status() raises:
    """Test tile status classification for CausalPaddingMask."""
    var storage = InlineArray[Scalar[DType.uint32], 2](uninitialized=True)
    storage[0] = 6
    storage[1] = 4
    var valid_lengths = LayoutTensor[DType.uint32, Layout.row_major(2)](storage)
    var mask = CausalPaddingMask(valid_lengths)

    # Causal FULL_MASK: max_q < min_k (tile entirely above diagonal).
    assert_equal(
        mask.status(Index(0, 4), Index(4, 4)), TileMaskStatus.FULL_MASK
    )

    # Causal would be NO_MASK but padding is unknown, so PARTIAL_MASK.
    assert_equal(
        mask.status(Index(4, 0), Index(4, 4)), TileMaskStatus.PARTIAL_MASK
    )

    # Causal alone is PARTIAL_MASK, padding unknown -> PARTIAL_MASK.
    assert_equal(
        mask.status(Index(2, 2), Index(4, 4)), TileMaskStatus.PARTIAL_MASK
    )

    # Another causal FULL_MASK case: q range [0,2) vs k range [4,8).
    assert_equal(
        mask.status(Index(0, 4), Index(2, 4)), TileMaskStatus.FULL_MASK
    )


def test_causal_padding_mask_apply() raises:
    """Test per-element masking for CausalPaddingMask."""
    var storage = InlineArray[Scalar[DType.uint32], 1](uninitialized=True)
    storage[0] = 3  # valid_length = 3 for seq 0
    var valid_lengths = LayoutTensor[DType.uint32, Layout.row_major(1)](storage)
    var mask = CausalPaddingMask(valid_lengths)

    var score_vec = SIMD[DType.float32, 4](0.0)
    score_vec[0] = 1.0
    score_vec[1] = 2.0
    score_vec[2] = 3.0
    score_vec[3] = 4.0

    comptime SIMD_T = SIMD[DType.float32, 4]
    var inf_vec = SIMD_T(MASK_VALUE)

    # q=4, k=0..3: causal allows all (q >= k), padding allows k<3.
    # k=0: visible, k=1: visible, k=2: visible, k=3: padding-masked.
    assert_equal(
        mask.mask(Index(0, 0, 4, 0), score_vec),
        SIMD_T(1.0, 2.0, 3.0, MASK_VALUE),
    )

    # q=1, k=0..3: causal allows k=0,1; padding allows k<3.
    # k=0: visible, k=1: visible, k=2: causal-masked, k=3: both masked.
    assert_equal(
        mask.mask(Index(0, 0, 1, 0), score_vec),
        SIMD_T(1.0, 2.0, MASK_VALUE, MASK_VALUE),
    )

    # q=2, k=4..7: all masked (both causal q<k and padding k>=3).
    assert_equal(mask.mask(Index(0, 0, 2, 4), score_vec), inf_vec)

    # q=0, k=0..3: causal allows only k=0; padding allows k<3.
    # k=0: visible, k=1..3: causal-masked.
    assert_equal(
        mask.mask(Index(0, 0, 0, 0), score_vec),
        SIMD_T(1.0, MASK_VALUE, MASK_VALUE, MASK_VALUE),
    )

    # q=2, k=0..3: causal allows k=0,1,2; padding allows k<3.
    # All three conditions agree: k=0,1,2 visible, k=3 padding-masked.
    assert_equal(
        mask.mask(Index(0, 0, 2, 0), score_vec),
        SIMD_T(1.0, 2.0, 3.0, MASK_VALUE),
    )


def test_causal_padding_mask_multi_seq() raises:
    """Test masking with multiple sequences having different valid lengths."""
    var storage = InlineArray[Scalar[DType.uint32], 2](uninitialized=True)
    storage[0] = 2  # seq 0: valid_length = 2
    storage[1] = 5  # seq 1: valid_length = 5
    var valid_lengths = LayoutTensor[DType.uint32, Layout.row_major(2)](storage)
    var mask = CausalPaddingMask(valid_lengths)

    comptime SIMD_T = SIMD[DType.float32, 4]
    var score_vec = SIMD_T(1.0, 2.0, 3.0, 4.0)

    # Seq 0 (valid_length=2), q=3, k=0..3:
    # causal allows k=0,1,2,3; padding allows k<2.
    # k=0: visible, k=1: visible, k=2,3: padding-masked.
    assert_equal(
        mask.mask(Index(0, 0, 3, 0), score_vec),
        SIMD_T(1.0, 2.0, MASK_VALUE, MASK_VALUE),
    )

    # Seq 1 (valid_length=5), q=3, k=0..3:
    # causal allows k=0,1,2,3; padding allows all (k<5).
    # All visible.
    assert_equal(
        mask.mask(Index(1, 0, 3, 0), score_vec),
        SIMD_T(1.0, 2.0, 3.0, 4.0),
    )


def main() raises:
    test_causal_padding_mask_status()
    test_causal_padding_mask_apply()
    test_causal_padding_mask_multi_seq()
