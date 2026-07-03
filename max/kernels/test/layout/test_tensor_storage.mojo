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
"""Tests for the `PointerStorage` implementation of the `TensorStorage` trait.

Exercises every operation `PointerStorage` provides against a host
`InlineArray` buffer: trait conformance, scalar and vectorized `load`/`store`
round-trips (with and without an element offset), `offset`, `distance`, and
`unsafe_cast` reinterpretation. `load` and `distance` require immutable
(`mut=False`) handles, while `store` requires a mutable one, so the tests use
`as_immutable()` where a read-only handle is needed.
"""

from std.sys import align_of
from std.testing import assert_equal, TestSuite

from layout.tensor_storage import PointerStorage

# Natural element alignments for the dtypes exercised below. `PointerStorage`
# load/store require an explicit alignment (unlike `UnsafePointer`, which
# defaults to the element alignment).
comptime ALIGN_F32 = align_of[DType.float32]()
comptime ALIGN_I32 = align_of[DType.int32]()
comptime ALIGN_U32 = align_of[DType.uint32]()


def test_load_store_scalar() raises:
    var buf = InlineArray[Float32, 4](fill=0.0)
    var storage = buf.unsafe_ptr()

    PointerStorage[element_width=1].store[alignment=ALIGN_F32](
        storage, Float32(3.5)
    )

    assert_equal(
        PointerStorage[element_width=1].load[width=1, alignment=ALIGN_F32](
            storage
        ),
        Float32(3.5),
    )


def test_load_store_simd() raises:
    var buf = InlineArray[Float32, 8](fill=0.0)
    var storage = buf.unsafe_ptr()

    var value = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    PointerStorage[element_width=1].store[alignment=ALIGN_F32](storage, value)

    assert_equal(
        PointerStorage[element_width=1].load[width=4, alignment=ALIGN_F32](
            storage
        ),
        value,
    )


def test_load_store_non_float_dtype() raises:
    var buf = InlineArray[Int32, 4](fill=0)
    var storage = buf.unsafe_ptr()

    var value = SIMD[DType.int32, 4](-1, 2, -3, 4)
    PointerStorage[element_width=1].store[alignment=ALIGN_I32](storage, value)

    assert_equal(
        PointerStorage[element_width=1].load[width=4, alignment=ALIGN_I32](
            storage
        ),
        value,
    )


def test_offset() raises:
    var buf = InlineArray[Float32, 4](fill=0.0)
    var storage = buf.unsafe_ptr()

    # Clear the buffer, then write `9.0` two elements in via an offset handle.
    PointerStorage[element_width=1].store[alignment=ALIGN_F32](
        storage, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    )
    var offset_storage = PointerStorage[element_width=1].offset(storage, 2)
    PointerStorage[element_width=1].store[alignment=ALIGN_F32](
        offset_storage, Float32(9.0)
    )

    # The write landed at element 2 of the real buffer.
    assert_equal(buf[2], Float32(9.0))
    # ...and the offset handle reads the same value back.
    assert_equal(
        PointerStorage[element_width=1].load[width=1, alignment=ALIGN_F32](
            offset_storage
        ),
        Float32(9.0),
    )


def test_load_store_offset_overload() raises:
    var buf = InlineArray[Float32, 4](fill=1.0)
    var storage = buf.unsafe_ptr()

    # Zero the buffer, then write at element 3 via the offset-taking store.
    PointerStorage[element_width=1].store[alignment=ALIGN_F32](
        storage, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    )
    PointerStorage[element_width=1].store[alignment=ALIGN_F32](
        storage, 3, Float32(7.0)
    )

    # The write landed at element 3, leaving the rest zeroed.
    assert_equal(buf[3], Float32(7.0))
    assert_equal(buf[0], Float32(0.0))
    # The offset-taking load reads the same element back.
    assert_equal(
        PointerStorage[element_width=1].load[width=1, alignment=ALIGN_F32](
            storage, 3
        ),
        Float32(7.0),
    )


def test_distance() raises:
    var buf = InlineArray[Float32, 8](fill=0.0)
    var storage = buf.unsafe_ptr()
    var offset_storage = PointerStorage[element_width=1].offset(storage, 3)

    assert_equal(
        PointerStorage[element_width=1].distance(offset_storage, storage), 3
    )
    assert_equal(
        PointerStorage[element_width=1].distance(storage, offset_storage), -3
    )
    assert_equal(PointerStorage[element_width=1].distance(storage, storage), 0)


def test_distance_offset_round_trip() raises:
    var buf = InlineArray[Float32, 16](fill=0.0)
    var storage = buf.unsafe_ptr()

    for n in range(16):
        var advanced = PointerStorage[element_width=1].offset(storage, n)
        assert_equal(
            PointerStorage[element_width=1].distance(advanced, storage), n
        )


def test_unsafe_cast() raises:
    var buf = InlineArray[Float32, 2](fill=0.0)
    var storage = buf.unsafe_ptr()
    PointerStorage[element_width=1].store[alignment=ALIGN_F32](
        storage, Float32(1.5)
    )

    # Reinterpret the float32 storage as uint32. No element conversion happens.
    var as_u32 = PointerStorage[element_width=1].unsafe_cast[
        DType.uint32, origin_of(storage), AddressSpace.GENERIC
    ](storage)

    # An independent pointer bitcast lands at the same address (distance 0)
    # and observes the same raw bits.
    var expected = storage.bitcast[Scalar[DType.uint32]]()
    assert_equal(PointerStorage[element_width=1].distance(as_u32, expected), 0)
    assert_equal(
        PointerStorage[element_width=1].load[width=1, alignment=ALIGN_U32](
            as_u32
        ),
        expected.load[width=1, alignment=ALIGN_U32](),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
