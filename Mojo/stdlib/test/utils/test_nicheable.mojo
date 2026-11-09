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

from std.memory import MaybeUninit
from std.sys import size_of
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from std.utils._nicheable import (
    NicheIndex,
    UnsafeNicheable,
    _NonNegativeScalarNiche,
)

comptime DTYPES = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.int,
    DType.int128,
]


struct Length[dtype: DType](TrivialRegisterPassable, UnsafeNicheable) where (
    dtype.is_integral() and dtype.is_signed()
):
    var value: Scalar[Self.dtype]

    def __init__(out self, value: Scalar[Self.dtype]):
        self.value = value

    comptime Niche = _NonNegativeScalarNiche[Self, "value", dtype=Self.dtype]

    @staticmethod
    def niche_count() -> Int:
        return Self.Niche.niche_count()

    @staticmethod
    def write_niche[index: Int](memory: MutPointer[MaybeUninit[Self], _]):
        Self.Niche.write_niche[index](memory)

    @staticmethod
    def classify_niche(memory: ImmPointer[MaybeUninit[Self], _]) -> NicheIndex:
        return Self.Niche.classify_niche(memory)


def test_niche_count() raises:
    assert_equal(Length[.int8].niche_count(), Int(Int8.MAX))
    assert_equal(Length[.int16].niche_count(), Int(Int16.MAX))
    assert_equal(Length[.int32].niche_count(), Int(Int32.MAX))
    assert_equal(Length[.int64].niche_count(), Int.MAX)
    assert_equal(Length[.int].niche_count(), Int.MAX)
    assert_equal(Length[.int128].niche_count(), Int.MAX)


def test_niche_round_trip() raises:
    comptime for dtype in DTYPES:
        comptime assert dtype.is_integral() and dtype.is_signed()
        comptime L = Length[dtype]
        comptime last = L.niche_count() - 1
        var storage = MaybeUninit[L]()

        L.write_niche[0](Pointer(to=storage))
        assert_true(L.classify_niche(Pointer(to=storage)) == NicheIndex(0))
        L.write_niche[1](Pointer(to=storage))
        assert_true(L.classify_niche(Pointer(to=storage)) == NicheIndex(1))
        L.write_niche[last](Pointer(to=storage))
        assert_true(L.classify_niche(Pointer(to=storage)) == NicheIndex(last))

        storage.write(L(0))
        assert_true(
            L.classify_niche(Pointer(to=storage)) == NicheIndex.NotANiche
        )
        storage.write(L(Scalar[dtype].MAX))
        assert_true(
            L.classify_niche(Pointer(to=storage)) == NicheIndex.NotANiche
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
