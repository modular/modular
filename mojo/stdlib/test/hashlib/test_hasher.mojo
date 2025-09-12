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

from hashlib.hasher import Hasher
from hashlib._ahash import AHasher
from pathlib import Path

from testing import assert_equal


struct DummyHasher(Hasher):
    var _dummy_value: UInt64

    fn __init__(out self):
        self._dummy_value = 0

    fn _update_with_bytes(
        mut self,
        data: UnsafePointer[
            UInt8, address_space = AddressSpace.GENERIC, mut=False, **_
        ],
        length: Int,
    ):
        for i in range(length):
            self._dummy_value += data[i].cast[DType.uint64]()

    fn _update_with_simd(mut self, value: SIMD[_, _]):
        self._dummy_value += value.cast[DType.uint64]().reduce_add()

    fn update[T: Hashable](mut self, value: T):
        value.__hash__(self)

    fn finish(var self) -> UInt64:
        return self._dummy_value


@fieldwise_init
struct SomeHashableStruct(Hashable, ImplicitlyCopyable, Movable):
    var _value: Int64

    fn __hash__[H: Hasher](self, mut hasher: H):
        hasher._update_with_simd(self._value)


def test_hasher():
    var hasher = DummyHasher()
    var hashable = SomeHashableStruct(42)
    hasher.update(hashable)
    assert_equal(hasher^.finish(), 42)


def test_hash_with_hasher():
    var hashable = SomeHashableStruct(10)
    assert_equal(hash[HasherType=DummyHasher](hashable), 10)


@fieldwise_init
struct ComplexHashableStruct(Hashable):
    var _value1: SomeHashableStruct
    var _value2: SomeHashableStruct

    fn __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(self._value1)
        hasher.update(self._value2)


def test_complex_hasher():
    var hasher = DummyHasher()
    var hashable = ComplexHashableStruct(
        SomeHashableStruct(42), SomeHashableStruct(10)
    )
    hasher.update(hashable)
    assert_equal(hasher^.finish(), 52)


def test_complex_hash_with_hasher():
    var hashable = ComplexHashableStruct(
        SomeHashableStruct(42), SomeHashableStruct(10)
    )
    assert_equal(hash[HasherType=DummyHasher](hashable), 52)


@fieldwise_init
struct ComplexHashableStructWithList(Hashable):
    var _value1: SomeHashableStruct
    var _value2: SomeHashableStruct
    var _value3: List[UInt8]

    fn __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(self._value1)
        hasher.update(self._value2)
        # This is okay because self is passed as read-only so the pointer will
        # be valid until at least the end of the function
        hasher._update_with_bytes(
            data=self._value3.unsafe_ptr(),
            length=len(self._value3),
        )


@fieldwise_init
struct ComplexHashableStructWithListAndWideSIMD(Hashable):
    var _value1: SomeHashableStruct
    var _value2: SomeHashableStruct
    var _value3: List[UInt8]
    var _value4: SIMD[DType.uint32, 4]

    fn __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(self._value1)
        hasher.update(self._value2)
        # This is okay because self is passed as read-only so the pointer will
        # be valid until at least the end of the function
        hasher._update_with_bytes(
            data=self._value3.unsafe_ptr(),
            length=len(self._value3),
        )
        hasher.update(self._value4)


def test_update_with_bytes():
    var hasher = DummyHasher()
    var hashable = ComplexHashableStructWithList(
        SomeHashableStruct(42), SomeHashableStruct(10), List[UInt8](1, 2, 3)
    )
    hasher.update(hashable)
    assert_equal(hasher^.finish(), 58)


alias _hash_with_hasher = hash[
    _, HasherType = AHasher[SIMD[DType.uint64, 4](0, 0, 0, 0)]
]


def test_with_ahasher():
    var hashable1 = ComplexHashableStructWithList(
        SomeHashableStruct(42), SomeHashableStruct(10), List[UInt8](1, 2, 3)
    )
    var hash_value = _hash_with_hasher(hashable1)
    assert_equal(hash_value, 7948090191592501094)
    var hashable2 = ComplexHashableStructWithListAndWideSIMD(
        SomeHashableStruct(42),
        SomeHashableStruct(10),
        List[UInt8](1, 2, 3),
        SIMD[DType.uint32, 4](1, 2, 3, 4),
    )
    hash_value = _hash_with_hasher(hashable2)
    assert_equal(hash_value, 1754891767834419861)


def test_hash_hashable_with_hasher_types():
    assert_equal(_hash_with_hasher(DType.uint64), 6529703120343940753)
    assert_equal(_hash_with_hasher(StaticString("")), 11583516797109448887)
    assert_equal(_hash_with_hasher(String()), 11583516797109448887)
    assert_equal(_hash_with_hasher(StringSlice("")), 11583516797109448887)
    assert_equal(_hash_with_hasher(Int(-123)), 4720193641311814362)
    assert_equal(_hash_with_hasher(UInt(123)), 4498397628805512285)
    assert_equal(
        _hash_with_hasher(SIMD[DType.float16, 4](0.1, -0.1, 12, 0)),
        9316495345323385448,
    )
    assert_equal(_hash_with_hasher(Path("/tmp")), 16491058316913697698)


def main():
    test_hasher()
    test_hash_with_hasher()
    test_complex_hasher()
    test_complex_hash_with_hasher()
    test_update_with_bytes()
    test_with_ahasher()
    test_hash_hashable_with_hasher_types()
