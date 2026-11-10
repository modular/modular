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

from std.testing import TestSuite, assert_false, assert_true

from extensibility import TensorTuple
from nn.attention.mha_mask import (
    MASK_VALUE,
    CausalMask,
    ChunkedCausalMask,
    ChunkedMask,
    MHAMask,
    NullMask,
    SlidingWindowCausalMask,
    SlidingWindowNonCausalMask,
)


def _visible[M: MHAMask](mask: M, q: Int, k: Int) -> Bool:
    var score = mask.mask((Int(0), Int(0), q, k), SIMD[DType.float32, 1](1.0))
    return score[0] != Float32(MASK_VALUE)


def test_null() raises:
    var mask = NullMask.make(TensorTuple())
    assert_true(_visible(mask, 0, 7))


def test_causal() raises:
    var mask = CausalMask.make(TensorTuple())
    assert_true(_visible(mask, 3, 3))
    assert_false(_visible(mask, 3, 4))


def test_sliding_window_causal() raises:
    var mask = SlidingWindowCausalMask[4].make(TensorTuple())
    assert_true(_visible(mask, 7, 4))
    assert_false(_visible(mask, 7, 3))
    assert_false(_visible(mask, 7, 8))


def test_sliding_window_noncausal() raises:
    var mask = SlidingWindowNonCausalMask[4].make(TensorTuple())
    assert_true(_visible(mask, 7, 9))
    assert_false(_visible(mask, 7, 3))


def test_chunked() raises:
    var mask = ChunkedMask[4].make(TensorTuple())
    assert_true(_visible(mask, 5, 7))
    assert_false(_visible(mask, 5, 3))


def test_chunked_causal() raises:
    comptime Mask = type_of(ChunkedCausalMask[4]())
    var mask = Mask.make(TensorTuple())
    assert_true(_visible(mask, 6, 5))
    assert_false(_visible(mask, 6, 7))
    assert_false(_visible(mask, 6, 3))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
