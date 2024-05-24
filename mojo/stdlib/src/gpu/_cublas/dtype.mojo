# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct Property:
    var _value: Int8
    alias MAJOR_VERSION = Property(0)
    alias MINOR_VERSION = Property(1)
    alias PATCH_LEVEL = Property(2)

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        if self == Self.MAJOR_VERSION:
            return "MAJOR_VERSION"
        if self == Self.MINOR_VERSION:
            return "MINOR_VERSION"
        if self == Self.PATCH_LEVEL:
            return "PATCH_LEVEL"
        return abort[String]("invalid Property entry")

    fn __int__(self) -> Int:
        return int(self._value)


@value
@register_passable("trivial")
struct DataType:
    var _value: Int8
    alias R_16F = DataType(2)
    alias C_16F = DataType(6)
    alias R_16BF = DataType(14)
    alias C_16BF = DataType(15)
    alias R_32F = DataType(0)
    alias C_32F = DataType(4)
    alias R_64F = DataType(1)
    alias C_64F = DataType(5)
    alias R_8I = DataType(3)
    alias C_8I = DataType(7)
    alias R_8U = DataType(8)
    alias C_8U = DataType(9)
    alias R_32I = DataType(10)
    alias C_32I = DataType(11)
    alias R_8F_E4M3 = DataType(28)
    alias R_8F_E5M2 = DataType(29)

    # Undocumented tyeps.
    # alias R_4I = DataType(8)
    # alias C_4I = DataType(9)
    # alias R_4U = DataType(10)
    # alias C_4U = DataType(11)
    # alias R_16I = DataType(16)
    # alias C_16I = DataType(17)
    # alias R_16U = DataType(18)
    # alias C_16U = DataType(19)
    # alias R_32U = DataType(22)
    # alias C_32U = DataType(23)
    # alias R_64I = DataType(24)
    # alias C_64I = DataType(25)
    # alias R_64U = DataType(26)
    # alias C_64U = DataType(27)

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        if self == Self.R_16F:
            return "R_16F"
        if self == Self.C_16F:
            return "C_16F"
        if self == Self.R_16BF:
            return "R_16BF"
        if self == Self.C_16BF:
            return "C_16BF"
        if self == Self.R_32F:
            return "R_32F"
        if self == Self.C_32F:
            return "C_32F"
        if self == Self.R_64F:
            return "R_64F"
        if self == Self.C_64F:
            return "C_64F"
        if self == Self.R_8I:
            return "R_8I"
        if self == Self.C_8I:
            return "C_8I"
        if self == Self.R_8U:
            return "R_8U"
        if self == Self.C_8U:
            return "C_8U"
        if self == Self.R_32I:
            return "R_32I"
        if self == Self.C_32I:
            return "C_32I"

        # Undocumented types.
        # if self == Self.R_4I:
        #     return "R_4I"
        # if self == Self.C_4I:
        #     return "C_4I"
        # if self == Self.R_4U:
        #     return "R_4U"
        # if self == Self.C_4U:
        #     return "C_4U"
        # if self == Self.R_16I:
        #     return "R_16I"
        # if self == Self.C_16I:
        #     return "C_16I"
        # if self == Self.R_16U:
        #     return "R_16U"
        # if self == Self.C_16U:
        #     return "C_16U"
        # if self == Self.R_32U:
        #     return "R_32U"
        # if self == Self.C_32U:
        #     return "C_32U"
        # if self == Self.R_64I:
        #     return "R_64I"
        # if self == Self.C_64I:
        #     return "C_64I"
        # if self == Self.R_64U:
        #     return "R_64U"
        # if self == Self.C_64U:
        #     return "C_64U"

        return abort[String]("invalid DataType entry")

    fn __int__(self) -> Int:
        return int(self._value)
