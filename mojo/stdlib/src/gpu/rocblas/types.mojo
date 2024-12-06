# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct Handle:
    var _value: UnsafePointer[NoneType]


@value
@register_passable("trivial")
struct Operation:
    var _value: Int32

    alias NONE = Self(111)
    alias TRANSPOSE = Self(112)
    alias CONJUGATE_TRANSPOSE = Self(113)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct Fill:
    var _value: Int32

    alias UPPER = Self(121)
    alias LOWER = Self(122)
    alias FULL = Self(123)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct Diagonal:
    var _value: Int32

    alias NON_UNIT = Self(131)
    alias DIAGONAL_UNIT = Self(132)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct Side:
    var _value: Int32

    alias LEFT = Self(141)
    alias RIGHT = Self(142)
    alias BOTH = Self(143)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct DataType:
    var _value: Int32

    alias F16_R = Self(150)
    alias F32_R = Self(151)
    alias F64_R = Self(152)
    alias F16_C = Self(153)
    alias F32_C = Self(154)
    alias F64_C = Self(155)

    alias I8_R = Self(160)
    alias U8_R = Self(161)
    alias I32_R = Self(162)
    alias U32_R = Self(163)

    alias I8_C = Self(164)
    alias U8_C = Self(165)
    alias I32_C = Self(166)
    alias U32_C = Self(167)

    alias BF16_R = Self(168)
    alias BF16_C = Self(169)
    alias F8_R = Self(170)
    alias BF8_R = Self(171)

    alias INVAID = Self(255)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct ComputeType:
    var _value: Int32

    alias F32 = Self(300)
    alias F8_F8_F32 = Self(301)
    alias F8_BF8_F32 = Self(302)
    alias BF8_F8_F32 = Self(303)
    alias BF8_BF8_F32 = Self(304)
    alias INVAID = Self(455)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct Status:
    var _value: Int32

    alias SUCCESS = Self(0)
    alias INVALID_HANDLE = Self(1)
    alias NOT_IMPLEMENTED = Self(2)
    alias INVALID_POINTER = Self(3)
    alias INVALID_SIZE = Self(4)
    alias MEMORY_ERROR = Self(5)
    alias INTERNAL_ERROR = Self(6)
    alias PERF_DEGRADED = Self(7)
    alias SIZE_QUERY_MISMATCH = Self(8)
    alias SIZE_INCREASED = Self(9)
    alias SIZE_UNCHANGED = Self(10)
    alias INVALID_VALUE = Self(11)
    alias CONTINUE = Self(12)
    alias CHECK_NUMERICS_FAIL = Self(13)
    alias EXCLUDED_FROM_BUILD = Self(14)
    alias ARCH_MISMATCH = Self(15)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct PointerMode:
    var _value: Int32

    alias HOST = Self(0)
    alias DEVICE = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct MallocBase:
    var _value: Int32


@value
@register_passable("trivial")
struct Algorithm:
    var _value: Int32

    alias STANDARD = Self(0)
    alias SOLUTION_INDEX = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct GEAMExOp:
    var _value: Int32

    alias MIN_PLUS = Self(0)
    alias PLUS_MIN = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __int__(self) -> Int:
        return int(self._value)
