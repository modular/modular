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
    UnsafeCustomNicheStorage,
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


def test_tuple_niches() raises:
    comptime L8 = Length[.int8]
    comptime L16 = Length[.int16]

    # A one-element tuple has its element's niches.
    comptime One = Tuple[L16]
    assert_equal(One.niche_count(), L16.niche_count())
    var one = MaybeUninit[One]()
    One.write_niche[2](Pointer(to=one))
    assert_true(One.classify_niche(Pointer(to=one)) == NicheIndex(2))
    one.write(One(L16(7)))
    assert_true(One.classify_niche(Pointer(to=one)) == NicheIndex.NotANiche)

    # A tuple has the niches of its element with the most niches, and leaves
    # the other elements as they are.
    comptime Most = Tuple[L8, L16]
    assert_equal(Most.niche_count(), L16.niche_count())
    var most = MaybeUninit[Most]()
    most.write(Most(L8(3), L16(4)))
    assert_true(Most.classify_niche(Pointer(to=most)) == NicheIndex.NotANiche)
    Most.write_niche[5](Pointer(to=most))
    assert_true(Most.classify_niche(Pointer(to=most)) == NicheIndex(5))
    assert_equal(most.unsafe_ptr()[][0].value, 3)
    assert_equal(most.unsafe_ptr()[][1].value, ~5)

    # On a tie, the first element's niches.
    comptime Tie = Tuple[L16, L16]
    var tie = MaybeUninit[Tie]()
    tie.write(Tie(L16(1), L16(2)))
    Tie.write_niche[0](Pointer(to=tie))
    assert_equal(tie.unsafe_ptr()[][0].value, ~0)
    assert_equal(tie.unsafe_ptr()[][1].value, 2)

    assert_false(conforms_to(Tuple[Int, Int], UnsafeNicheable))
    assert_false(conforms_to(Tuple[], UnsafeNicheable))


def test_tuple_custom_niche_storage() raises:
    comptime P = Pointer[Int, ImmStaticOrigin]

    # Only a one-element tuple has its element's layout.
    assert_true(conforms_to(Tuple[P], UnsafeCustomNicheStorage))
    assert_false(conforms_to(Tuple[P, P], UnsafeCustomNicheStorage))
    assert_false(conforms_to(Tuple[Length[.int8]], UnsafeCustomNicheStorage))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
