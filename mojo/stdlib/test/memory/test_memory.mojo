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

from sys import simd_width_of, size_of

from memory import (
    AddressSpace,
    memcmp,
    memcpy,
    memset,
    memset_zero,
)
from testing import (
    assert_almost_equal,
    assert_equal,
    assert_not_equal,
    assert_true,
)

from utils.numerics import nan

alias void = __mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`
alias int8_pop = __mlir_type.`!pop.scalar<si8>`


@fieldwise_init
@register_passable("trivial")
struct Pair:
    var lo: Int
    var hi: Int


def test_memcpy():
    var pair1 = Pair(1, 2)
    var pair2 = Pair(0, 0)

    var src = UnsafePointer(to=pair1)
    var dest = UnsafePointer(to=pair2)

    # UnsafePointer test
    pair2.lo = 0
    pair2.hi = 0
    memcpy(dest, src, 1)

    assert_equal(pair2.lo, 1)
    assert_equal(pair2.hi, 2)

    @parameter
    def _test_memcpy_buf[size: Int]():
        var buf = UnsafePointer[UInt8]().alloc(size * 2)
        memset_zero(buf + size, size)
        var src = UnsafePointer[UInt8]().alloc(size * 2)
        var dst = UnsafePointer[UInt8]().alloc(size * 2)
        for i in range(size * 2):
            buf[i] = src[i] = 2
            dst[i] = 0

        memcpy(dst, src, size)
        var err = memcmp(dst, buf, size)

        assert_equal(err, 0)
        buf.free()
        src.free()
        dst.free()

    _test_memcpy_buf[1]()
    _test_memcpy_buf[4]()
    _test_memcpy_buf[7]()
    _test_memcpy_buf[11]()
    _test_memcpy_buf[8]()
    _test_memcpy_buf[12]()
    _test_memcpy_buf[16]()
    _test_memcpy_buf[19]()
    _ = pair1
    _ = pair2


def test_memcpy_dtype():
    var a = UnsafePointer[Int32].alloc(4)
    var b = UnsafePointer[Int32].alloc(4)
    for i in range(4):
        a[i] = i
        b[i] = -1

    assert_equal(b[0], -1)
    assert_equal(b[1], -1)
    assert_equal(b[2], -1)
    assert_equal(b[3], -1)

    memcpy(b, a, 4)

    assert_equal(b[0], 0)
    assert_equal(b[1], 1)
    assert_equal(b[2], 2)
    assert_equal(b[3], 3)

    a.free()
    b.free()


def test_memcmp():
    var pair1 = Pair(1, 2)
    var pair2 = Pair(1, 2)

    var ptr1 = UnsafePointer(to=pair1)
    var ptr2 = UnsafePointer(to=pair2)

    var errors = memcmp(ptr1, ptr2, 1)

    assert_equal(errors, 0)
    _ = pair1
    _ = pair2


@fieldwise_init
@register_passable("trivial")
struct SixByteStruct:
    var a: Int16
    var b: Int16
    var c: Int16


def test_memcmp_non_multiple_of_int32():
    var triple1 = SixByteStruct(0, 0, 0)
    var triple2 = SixByteStruct(0, 0, 1)

    constrained[size_of[SixByteStruct]() == 6]()

    var ptr1 = UnsafePointer(to=triple1)
    var ptr2 = UnsafePointer(to=triple2)
    var errors = memcmp(ptr1, ptr2, 1)
    assert_equal(errors, -1)

    _ = triple1
    _ = triple2


def test_memcmp_overflow():
    p1 = UnsafePointer[Byte].alloc(1)
    p2 = UnsafePointer[Byte].alloc(1)
    p1.store(-120)
    p2.store(120)

    c = memcmp(p1, p2, 1)
    assert_equal(c, 1)

    c = memcmp(p2, p1, 1)
    assert_equal(c, -1)


def test_memcmp_simd():
    var length = simd_width_of[DType.int8]() + 10

    var p1 = UnsafePointer[Int8].alloc(length)
    var p2 = UnsafePointer[Int8].alloc(length)
    memset_zero(p1, length)
    memset_zero(p2, length)
    p1.store(120)
    p1.store(1, 100)
    p2.store(120)
    p2.store(1, 90)

    var c = memcmp(p1, p2, length)
    assert_equal(c, 1, "[120, 100, 0, ...] is bigger than [120, 90, 0, ...]")

    c = memcmp(p2, p1, length)
    assert_equal(c, -1, "[120, 90, 0, ...] is smaller than [120, 100, 0, ...]")

    memset_zero(p1, length)
    memset_zero(p2, length)

    p1.store(length - 2, 120)
    p1.store(length - 1, 100)
    p2.store(length - 2, 120)
    p2.store(length - 1, 90)

    c = memcmp(p1, p2, length)
    assert_equal(c, 1, "[..., 0, 120, 100] is bigger than [..., 0, 120, 90]")

    c = memcmp(p2, p1, length)
    assert_equal(c, -1, "[..., 0, 120, 90] is smaller than [..., 120, 100]")


def test_memcmp_extensive[
    dtype: DType, extermes: StaticString = ""
](count: Int):
    var ptr1 = UnsafePointer[Scalar[dtype]].alloc(count)
    var ptr2 = UnsafePointer[Scalar[dtype]].alloc(count)

    var dptr1 = UnsafePointer[Scalar[dtype]].alloc(count)
    var dptr2 = UnsafePointer[Scalar[dtype]].alloc(count)

    for i in range(count):
        ptr1[i] = i
        dptr1[i] = i

        @parameter
        if extermes == "":
            ptr2[i] = i + 1
            dptr2[i] = i + 1
        elif extermes == "nan":
            ptr2[i] = nan[dtype]()
            dptr2[i] = nan[dtype]()
        elif extermes == "inf":
            ptr2[i] = Scalar[dtype].MAX
            dptr2[i] = Scalar[dtype].MAX

    assert_equal(
        memcmp(ptr1, ptr1, count),
        0,
        String("for dtype=", dtype, ";count=", count),
    )
    assert_equal(
        memcmp(ptr1, ptr2, count),
        -1,
        String("for dtype=", dtype, ";count=", count),
    )
    assert_equal(
        memcmp(ptr2, ptr1, count),
        1,
        String("for dtype=", dtype, ";count=", count),
    )

    assert_equal(
        memcmp(dptr1, dptr1, count),
        0,
        String("for dtype=", dtype, ";extremes=", extermes, ";count=", count),
    )
    assert_equal(
        memcmp(dptr1, dptr2, count),
        -1,
        String("for dtype=", dtype, ";extremes=", extermes, ";count=", count),
    )
    assert_equal(
        memcmp(dptr2, dptr1, count),
        1,
        String("for dtype=", dtype, ";extremes=", extermes, ";count=", count),
    )

    ptr1.free()
    ptr2.free()
    dptr1.free()
    dptr2.free()


def test_memcmp_extensive():
    test_memcmp_extensive[DType.int8](1)
    test_memcmp_extensive[DType.int8](3)

    test_memcmp_extensive[DType.index](3)
    test_memcmp_extensive[DType.index](simd_width_of[Int]())
    test_memcmp_extensive[DType.index](4 * simd_width_of[DType.index]())
    test_memcmp_extensive[DType.index](4 * simd_width_of[DType.index]() + 1)
    test_memcmp_extensive[DType.index](4 * simd_width_of[DType.index]() - 1)

    test_memcmp_extensive[DType.float32](3)
    test_memcmp_extensive[DType.float32](simd_width_of[DType.float32]())
    test_memcmp_extensive[DType.float32](4 * simd_width_of[DType.float32]())
    test_memcmp_extensive[DType.float32](4 * simd_width_of[DType.float32]() + 1)
    test_memcmp_extensive[DType.float32](4 * simd_width_of[DType.float32]() - 1)

    test_memcmp_extensive[DType.float32, "nan"](3)
    test_memcmp_extensive[DType.float32, "nan"](99)
    test_memcmp_extensive[DType.float32, "nan"](254)

    test_memcmp_extensive[DType.float32, "inf"](3)
    test_memcmp_extensive[DType.float32, "inf"](99)
    test_memcmp_extensive[DType.float32, "inf"](254)


def test_memcmp_simd_boundary():
    """Test edge cases in SIMD memcmp implementation that could expose bugs."""
    alias simd_width = simd_width_of[DType.int8]()

    # Test 1: Difference exactly at SIMD boundary
    alias size = simd_width + 1
    var ptr1 = UnsafePointer[Int8].alloc(size)
    var ptr2 = UnsafePointer[Int8].alloc(size)

    # Fill with identical data
    for i in range(size):
        ptr1[i] = 42
        ptr2[i] = 42

    # Make difference at SIMD boundary
    ptr2[simd_width] = 43

    var result = memcmp(ptr1, ptr2, size)
    assert_equal(result, -1, "Should detect difference at SIMD boundary")

    # Test opposite direction
    ptr1[simd_width] = 44
    ptr2[simd_width] = 42
    result = memcmp(ptr1, ptr2, size)
    assert_equal(
        result, 1, "Should detect difference at SIMD boundary (reverse)"
    )

    ptr1.free()
    ptr2.free()


def test_memcmp_simd_overlap():
    """Test overlapping region handling in SIMD memcmp."""
    alias simd_width = simd_width_of[DType.int8]()

    # Test sizes that trigger overlapping tail reads
    var test_sizes = List[Int](
        simd_width + 1, simd_width + 2, simd_width * 2 - 1, simd_width * 2 + 1
    )

    for i in range(len(test_sizes)):
        var size = test_sizes[i]
        var ptr1 = UnsafePointer[Int8].alloc(size)
        var ptr2 = UnsafePointer[Int8].alloc(size)

        # Fill with identical data
        for j in range(size):
            ptr1[j] = 42
            ptr2[j] = 42

        # Should be equal
        var result = memcmp(ptr1, ptr2, size)
        assert_equal(result, 0, "Overlapping regions should be equal")

        # Make difference in overlap region
        ptr2[size - 1] = ptr2[size - 1] + 1
        result = memcmp(ptr1, ptr2, size)
        assert_equal(result, -1, "Should detect difference in overlap region")

        ptr1.free()
        ptr2.free()


def test_memcmp_simd_index_finding():
    """Test index finding logic in SIMD memcmp."""
    var simd_width = simd_width_of[DType.int8]()

    # Test difference at each possible SIMD lane position
    for lane in range(simd_width):
        var ptr1 = UnsafePointer[Int8].alloc(simd_width)
        var ptr2 = UnsafePointer[Int8].alloc(simd_width)

        # Fill with identical data
        for i in range(simd_width):
            ptr1[i] = 100
            ptr2[i] = 100

        # Create difference at specific lane
        ptr2[lane] = 101

        var result = memcmp(ptr1, ptr2, simd_width)
        assert_equal(
            result, -1, "Should detect difference at lane " + String(lane)
        )

        # Test opposite direction
        ptr1[lane] = 102
        ptr2[lane] = 100
        result = memcmp(ptr1, ptr2, simd_width)
        assert_equal(
            result,
            1,
            "Should detect difference at lane " + String(lane) + " (reverse)",
        )

        ptr1.free()
        ptr2.free()


def test_memcmp_simd_signed_overflow():
    """Test signed byte overflow cases in SIMD memcmp."""
    var ptr1 = UnsafePointer[Int8].alloc(4)
    var ptr2 = UnsafePointer[Int8].alloc(4)

    # Test extreme signed values
    ptr1[0] = -128  # Most negative
    ptr1[1] = -1
    ptr1[2] = 0
    ptr1[3] = 127  # Most positive

    ptr2[0] = -128
    ptr2[1] = -1
    ptr2[2] = 0
    ptr2[3] = 127

    var result = memcmp(ptr1, ptr2, 4)
    assert_equal(result, 0, "Identical extreme values should be equal")

    # Test signed comparison edge cases
    ptr1[0] = -1  # 0xFF as unsigned
    ptr2[0] = 1  # 0x01 as unsigned

    result = memcmp(ptr1, ptr2, 4)
    assert_equal(
        result, 1, "0xFF should be greater than 0x01 in unsigned comparison"
    )

    ptr1.free()
    ptr2.free()


def test_memcmp_simd_alignment():
    """Test alignment-related bugs in SIMD memcmp."""
    var size = 64
    var large_ptr1 = UnsafePointer[Int8].alloc(size)
    var large_ptr2 = UnsafePointer[Int8].alloc(size)

    # Fill with pattern
    for i in range(size):
        large_ptr1[i] = Int8(i % 256)
        large_ptr2[i] = Int8(i % 256)

    # Test various unaligned starting positions
    for offset in range(1, 8):
        var ptr1 = large_ptr1 + offset
        var ptr2 = large_ptr2 + offset
        var test_size = size - offset - 8

        var result = memcmp(ptr1, ptr2, test_size)
        assert_equal(
            result,
            0,
            "Unaligned comparison should work at offset " + String(offset),
        )

        # Create difference and test
        ptr2[test_size - 1] = ptr2[test_size - 1] + 1
        result = memcmp(ptr1, ptr2, test_size)
        assert_equal(
            result, -1, "Should detect difference with unaligned access"
        )

        # Restore for next iteration
        ptr2[test_size - 1] = ptr2[test_size - 1] - 1

    large_ptr1.free()
    large_ptr2.free()


def test_memcmp_simd_width_edge_cases():
    """Test edge cases around different SIMD widths."""
    var simd_width = simd_width_of[DType.int8]()

    # Test sizes that might cause issues with SIMD width calculations
    var critical_sizes = List[Int](
        simd_width - 1,
        simd_width,
        simd_width + 1,
        simd_width * 2 - 1,
        simd_width * 2,
        simd_width * 2 + 1,
        simd_width * 3 - 1,
        simd_width * 3,
        simd_width * 3 + 1,
    )

    for i in range(len(critical_sizes)):
        var size = critical_sizes[i]
        var ptr1 = UnsafePointer[Int8].alloc(size)
        var ptr2 = UnsafePointer[Int8].alloc(size)

        # Fill with identical sequential data
        for j in range(size):
            ptr1[j] = Int8(j % 256)
            ptr2[j] = Int8(j % 256)

        var result = memcmp(ptr1, ptr2, size)
        assert_equal(
            result,
            0,
            "Sequential data should be equal for size " + String(size),
        )

        # Test difference at end
        if size > 0:
            ptr2[size - 1] = ptr2[size - 1] + 1
            result = memcmp(ptr1, ptr2, size)
            assert_equal(
                result,
                -1,
                "Should detect end difference for size " + String(size),
            )

        ptr1.free()
        ptr2.free()


def test_memcmp_simd_zero_bytes():
    """Test handling of zero bytes in SIMD memcmp."""
    alias size = simd_width_of[DType.int8]() * 2
    var ptr1 = UnsafePointer[Int8].alloc(size)
    var ptr2 = UnsafePointer[Int8].alloc(size)

    # Fill with zeros
    memset_zero(ptr1, size)
    memset_zero(ptr2, size)

    var result = memcmp(ptr1, ptr2, size)
    assert_equal(result, 0, "Zero-filled buffers should be equal")

    # Test zero vs non-zero at different positions
    var test_positions = List[Int](0, 1, size // 2, size - 1)

    for i in range(len(test_positions)):
        var pos = test_positions[i]

        # Reset to zeros
        memset_zero(ptr1, size)
        memset_zero(ptr2, size)

        # Create difference at position
        ptr2[pos] = 1
        result = memcmp(ptr1, ptr2, size)
        assert_equal(
            result,
            -1,
            "Should detect zero vs non-zero at position " + String(pos),
        )

        # Test opposite
        ptr1[pos] = 2
        ptr2[pos] = 0
        result = memcmp(ptr1, ptr2, size)
        assert_equal(
            result,
            1,
            "Should detect non-zero vs zero at position " + String(pos),
        )

    ptr1.free()
    ptr2.free()


def test_memset():
    var pair = Pair(1, 2)

    var ptr = UnsafePointer(to=pair)
    memset_zero(ptr, 1)

    assert_equal(pair.lo, 0)
    assert_equal(pair.hi, 0)

    pair.lo = 1
    pair.hi = 2
    memset_zero(ptr, 1)

    assert_equal(pair.lo, 0)
    assert_equal(pair.hi, 0)

    var buf0 = UnsafePointer[Int32].alloc(2)
    memset(buf0, 1, 2)
    assert_equal(buf0.load(0), 16843009)
    memset(buf0, -1, 2)
    assert_equal(buf0.load(0), -1)
    buf0.free()

    var buf1 = UnsafePointer[Int8].alloc(2)
    memset(buf1, 5, 2)
    assert_equal(buf1.load(0), 5)
    buf1.free()

    var buf3 = UnsafePointer[Int32].alloc(2)
    memset(buf3, 1, 2)
    memset_zero[count=2](buf3)
    assert_equal(buf3.load(0), 0)
    assert_equal(buf3.load(1), 0)
    buf3.free()

    _ = pair


def test_pointer_string():
    var nullptr = UnsafePointer[Int]()
    assert_equal(String(nullptr), "0x0")

    var ptr = UnsafePointer[Int].alloc(1)
    assert_true(String(ptr).startswith("0x"))
    assert_not_equal(String(ptr), "0x0")
    ptr.free()


def test_dtypepointer_string():
    var nullptr = UnsafePointer[Float32]()
    assert_equal(String(nullptr), "0x0")

    var ptr = UnsafePointer[Float32].alloc(1)
    assert_true(String(ptr).startswith("0x"))
    assert_not_equal(String(ptr), "0x0")
    ptr.free()


def test_pointer_explicit_copy():
    var ptr = UnsafePointer[Int].alloc(1)
    ptr[] = 42
    var copy = UnsafePointer(other=ptr)
    assert_equal(copy[], 42)
    ptr.free()


def test_pointer_refitem():
    var ptr = UnsafePointer[Int].alloc(1)
    ptr[] = 42
    assert_equal(ptr[], 42)
    ptr.free()


def test_pointer_refitem_string():
    alias payload = "$Modular!Mojo!HelloWorld^"
    var ptr = UnsafePointer[String].alloc(1)
    __get_address_as_uninit_lvalue(ptr.address) = String()
    ptr[] = payload
    assert_equal(ptr[], payload)
    ptr.free()


def test_pointer_refitem_pair():
    var ptr = UnsafePointer[Pair].alloc(1)
    ptr[].lo = 42
    ptr[].hi = 24
    #   NOTE: We want to write the below but we can't implement a generic assert_equal yet.
    #   assert_equal(ptr[], Pair(42, 24))
    assert_equal(ptr[].lo, 42)
    assert_equal(ptr[].hi, 24)
    ptr.free()


def test_address_space_str():
    assert_equal(String(AddressSpace.GENERIC), "AddressSpace.GENERIC")
    assert_equal(String(AddressSpace(17)), "AddressSpace(17)")


def test_dtypepointer_gather():
    var ptr = UnsafePointer[Float32].alloc(4)
    ptr.store(0, SIMD[ptr.type.dtype, 4](0.0, 1.0, 2.0, 3.0))

    @parameter
    def _test_gather[
        width: Int
    ](offset: SIMD[_, width], desired: SIMD[ptr.type.dtype, width]):
        var actual = ptr.gather(offset)
        assert_almost_equal(
            actual, desired, msg="_test_gather", atol=0.0, rtol=0.0
        )

    @parameter
    def _test_masked_gather[
        width: Int
    ](
        offset: SIMD[_, width],
        mask: SIMD[DType.bool, width],
        default: SIMD[ptr.type.dtype, width],
        desired: SIMD[ptr.type.dtype, width],
    ):
        var actual = ptr.gather(offset, mask, default)
        assert_almost_equal(
            actual, desired, msg="_test_masked_gather", atol=0.0, rtol=0.0
        )

    var offset = SIMD[DType.int64, 8](3, 0, 2, 1, 2, 0, 3, 1)
    var desired = SIMD[ptr.type.dtype, 8](
        3.0, 0.0, 2.0, 1.0, 2.0, 0.0, 3.0, 1.0
    )

    _test_gather[1](UInt16(2), 2.0)
    _test_gather(offset.cast[DType.uint32]().slice[2](), desired.slice[2]())
    _test_gather(offset.cast[DType.uint64]().slice[4](), desired.slice[4]())

    var mask = offset.ge(0) & offset.lt(3)
    var default = SIMD[ptr.type.dtype, 8](-1.0)
    desired = SIMD[ptr.type.dtype, 8](-1.0, 0.0, 2.0, 1.0, 2.0, 0.0, -1.0, 1.0)

    _test_masked_gather[1](Int16(2), Scalar[DType.bool](False), -1.0, -1.0)
    _test_masked_gather[1](Int32(2), Scalar[DType.bool](True), -1.0, 2.0)
    _test_masked_gather(offset, mask, default, desired)

    ptr.free()


def test_dtypepointer_scatter():
    var ptr = UnsafePointer[Float32].alloc(4)
    ptr.store(0, SIMD[ptr.type.dtype, 4](0.0))

    @parameter
    def _test_scatter[
        width: Int
    ](
        offset: SIMD[_, width],
        val: SIMD[ptr.type.dtype, width],
        desired: SIMD[ptr.type.dtype, 4],
    ):
        ptr.scatter(offset, val)
        var actual = ptr.load[width=4](0)
        assert_almost_equal(
            actual, desired, msg="_test_scatter", atol=0.0, rtol=0.0
        )

    @parameter
    def _test_masked_scatter[
        width: Int
    ](
        offset: SIMD[_, width],
        val: SIMD[ptr.type.dtype, width],
        mask: SIMD[DType.bool, width],
        desired: SIMD[ptr.type.dtype, 4],
    ):
        ptr.scatter(offset, val, mask)
        var actual = ptr.load[width=4](0)
        assert_almost_equal(
            actual, desired, msg="_test_masked_scatter", atol=0.0, rtol=0.0
        )

    _test_scatter[1](
        UInt16(2), 2.0, SIMD[ptr.type.dtype, 4](0.0, 0.0, 2.0, 0.0)
    )
    _test_scatter(  # Test with repeated offsets
        SIMD[DType.uint32, 4](1, 1, 1, 1),
        SIMD[ptr.type.dtype, 4](-1.0, 2.0, -2.0, 1.0),
        SIMD[ptr.type.dtype, 4](0.0, 1.0, 2.0, 0.0),
    )
    _test_scatter(
        SIMD[DType.uint64, 4](3, 2, 1, 0),
        SIMD[ptr.type.dtype, 4](0.0, 1.0, 2.0, 3.0),
        SIMD[ptr.type.dtype, 4](3.0, 2.0, 1.0, 0.0),
    )

    ptr.store(0, SIMD[ptr.type.dtype, 4](0.0))

    _test_masked_scatter[1](
        Int16(2),
        2.0,
        Scalar[DType.bool](False),
        SIMD[ptr.type.dtype, 4](0.0, 0.0, 0.0, 0.0),
    )
    _test_masked_scatter[1](
        Int32(2),
        2.0,
        Scalar[DType.bool](True),
        SIMD[ptr.type.dtype, 4](0.0, 0.0, 2.0, 0.0),
    )
    _test_masked_scatter(  # Test with repeated offsets
        SIMD[DType.int64, 4](1, 1, 1, 1),
        SIMD[ptr.type.dtype, 4](-1.0, 2.0, -2.0, 1.0),
        SIMD[DType.bool, 4](True, True, True, False),
        SIMD[ptr.type.dtype, 4](0.0, -2.0, 2.0, 0.0),
    )
    _test_masked_scatter(
        SIMD[DType.index, 4](3, 2, 1, 0),
        SIMD[ptr.type.dtype, 4](0.0, 1.0, 2.0, 3.0),
        SIMD[DType.bool, 4](True, False, True, True),
        SIMD[ptr.type.dtype, 4](3.0, 2.0, 2.0, 0.0),
    )

    ptr.free()


def test_indexing():
    var ptr = UnsafePointer[Float32].alloc(4)
    for i in range(4):
        ptr[i] = i

    assert_equal(ptr[Int(2)], 2)
    assert_equal(ptr[1], 1)


def main():
    test_memcpy()
    test_memcpy_dtype()
    test_memcmp()
    test_memcmp_non_multiple_of_int32()
    test_memcmp_overflow()
    test_memcmp_simd()
    test_memcmp_extensive()
    test_memcmp_simd_boundary()
    test_memcmp_simd_overlap()
    test_memcmp_simd_index_finding()
    test_memcmp_simd_signed_overflow()
    test_memcmp_simd_alignment()
    test_memcmp_simd_width_edge_cases()
    test_memcmp_simd_zero_bytes()
    test_memset()

    test_pointer_explicit_copy()
    test_dtypepointer_string()
    test_pointer_refitem()
    test_pointer_refitem_string()
    test_pointer_refitem_pair()
    test_pointer_string()

    test_address_space_str()

    test_dtypepointer_gather()
    test_dtypepointer_scatter()
    test_indexing()
