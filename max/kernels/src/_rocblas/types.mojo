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

from memory import LegacyOpaquePointer as OpaquePointer
from os import abort


@fieldwise_init
@register_passable("trivial")
struct Handle(Defaultable):
    var _value: OpaquePointer

    fn __init__(out self):
        self._value = OpaquePointer()

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value


@fieldwise_init
@register_passable("trivial")
struct Operation:
    var _value: Int32

    comptime NONE = Self(111)
    comptime TRANSPOSE = Self(112)
    comptime CONJUGATE_TRANSPOSE = Self(113)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
@register_passable("trivial")
struct Fill:
    var _value: Int32

    comptime UPPER = Self(121)
    comptime LOWER = Self(122)
    comptime FULL = Self(123)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
@register_passable("trivial")
struct Diagonal:
    var _value: Int32

    comptime NON_UNIT = Self(131)
    comptime DIAGONAL_UNIT = Self(132)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
@register_passable("trivial")
struct Side:
    var _value: Int32

    comptime LEFT = Self(141)
    comptime RIGHT = Self(142)
    comptime BOTH = Self(143)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
@register_passable("trivial")
struct DataType:
    var _value: Int32

    comptime F16_R = Self(150)
    comptime F32_R = Self(151)
    comptime F64_R = Self(152)
    comptime F16_C = Self(153)
    comptime F32_C = Self(154)
    comptime F64_C = Self(155)

    comptime I8_R = Self(160)
    comptime U8_R = Self(161)
    comptime I32_R = Self(162)
    comptime U32_R = Self(163)

    comptime I8_C = Self(164)
    comptime U8_C = Self(165)
    comptime I32_C = Self(166)
    comptime U32_C = Self(167)

    comptime BF16_R = Self(168)
    comptime BF16_C = Self(169)
    comptime F8_R = Self(170)
    comptime BF8_R = Self(171)

    comptime INVALID = Self(255)

    fn __init__(out self, value: Int):
        self._value = value

    fn __init__(out self, dtype: DType) raises:
        if dtype is DType.float16:
            self = Self.F16_R
        elif dtype is DType.bfloat16:
            self = Self.BF16_R
        elif dtype is DType.float32:
            self = Self.F32_R
        elif dtype is DType.float64:
            self = Self.F64_R
        else:
            raise Error(
                "the dtype '", dtype, "' is not currently handled by rocBLAS"
            )

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return Int(self._value)


@register_passable("trivial")
struct ComputeType:
    var _value: Int32

    comptime F32 = Self(300)
    comptime F8_F8_F32 = Self(301)
    comptime F8_BF8_F32 = Self(302)
    comptime BF8_F8_F32 = Self(303)
    comptime BF8_BF8_F32 = Self(304)
    comptime INVALID = Self(455)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
@register_passable("trivial")
struct Status(Equatable, Writable):
    var _value: Int32

    comptime SUCCESS = Self(0)
    comptime INVALID_HANDLE = Self(1)
    comptime NOT_IMPLEMENTED = Self(2)
    comptime INVALID_POINTER = Self(3)
    comptime INVALID_SIZE = Self(4)
    comptime MEMORY_ERROR = Self(5)
    comptime INTERNAL_ERROR = Self(6)
    comptime PERF_DEGRADED = Self(7)
    comptime SIZE_QUERY_MISMATCH = Self(8)
    comptime SIZE_INCREASED = Self(9)
    comptime SIZE_UNCHANGED = Self(10)
    comptime INVALID_VALUE = Self(11)
    comptime CONTINUE = Self(12)
    comptime CHECK_NUMERICS_FAIL = Self(13)
    comptime EXCLUDED_FROM_BUILD = Self(14)
    comptime ARCH_MISMATCH = Self(15)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __int__(self) -> Int:
        return Int(self._value)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        if self == Self.SUCCESS:
            return writer.write("SUCCESS")
        if self == Self.INVALID_HANDLE:
            return writer.write("INVALID_HANDLE")
        if self == Self.NOT_IMPLEMENTED:
            return writer.write("NOT_IMPLEMENTED")
        if self == Self.INVALID_POINTER:
            return writer.write("INVALID_POINTER")
        if self == Self.INVALID_SIZE:
            return writer.write("INVALID_SIZE")
        if self == Self.MEMORY_ERROR:
            return writer.write("MEMORY_ERROR")
        if self == Self.INTERNAL_ERROR:
            return writer.write("INTERNAL_ERROR")
        if self == Self.PERF_DEGRADED:
            return writer.write("PERF_DEGRADED")
        if self == Self.SIZE_QUERY_MISMATCH:
            return writer.write("SIZE_QUERY_MISMATCH")
        if self == Self.SIZE_INCREASED:
            return writer.write("SIZE_INCREASED")
        if self == Self.SIZE_UNCHANGED:
            return writer.write("SIZE_UNCHANGED")
        if self == Self.INVALID_VALUE:
            return writer.write("INVALID_VALUE")
        if self == Self.CONTINUE:
            return writer.write("CONTINUE")
        if self == Self.CHECK_NUMERICS_FAIL:
            return writer.write("CHECK_NUMERICS_FAIL")
        if self == Self.EXCLUDED_FROM_BUILD:
            return writer.write("EXCLUDED_FROM_BUILD")
        if self == Self.ARCH_MISMATCH:
            return writer.write("ARCH_MISMATCH")

        abort("unreachable: invalid Status entry")


@fieldwise_init
@register_passable("trivial")
struct PointerMode:
    var _value: Int32

    comptime HOST = Self(0)
    comptime DEVICE = Self(1)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
@register_passable("trivial")
struct MallocBase:
    var _value: Int32


@fieldwise_init
@register_passable("trivial")
struct Algorithm:
    var _value: Int32

    comptime STANDARD = Self(0)
    comptime SOLUTION_INDEX = Self(1)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
@register_passable("trivial")
struct GEAMExOp:
    var _value: Int32

    comptime MIN_PLUS = Self(0)
    comptime PLUS_MIN = Self(1)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return Int(self._value)
