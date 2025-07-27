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

from os import abort


@register_passable("trivial")
struct Property:
    var _value: Int32
    alias MAJOR_VERSION = Self(0)
    alias MINOR_VERSION = Self(1)
    alias PATCH_LEVEL = Self(2)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        if self == Self.MAJOR_VERSION:
            return "MAJOR_VERSION"
        if self == Self.MINOR_VERSION:
            return "MINOR_VERSION"
        if self == Self.PATCH_LEVEL:
            return "PATCH_LEVEL"
        return abort[String]("invalid Property entry")

    fn __int__(self) -> Int:
        return Int(self._value)


@register_passable("trivial")
struct DataType:
    var _value: Int32
    alias R_16F = Self(2)
    alias C_16F = Self(6)
    alias R_16BF = Self(14)
    alias C_16BF = Self(15)
    alias R_32F = Self(0)
    alias C_32F = Self(4)
    alias R_64F = Self(1)
    alias C_64F = Self(5)
    alias R_8I = Self(3)
    alias C_8I = Self(7)
    alias R_8U = Self(8)
    alias C_8U = Self(9)
    alias R_32I = Self(10)
    alias C_32I = Self(11)
    alias R_8F_E4M3 = Self(28)
    alias R_8F_E5M2 = Self(29)
    alias R_4I = Self(16)
    alias C_4I = Self(17)
    alias R_4U = Self(18)
    alias C_4U = Self(19)
    alias R_16I = Self(20)
    alias C_16I = Self(21)
    alias R_16U = Self(22)
    alias C_16U = Self(23)
    alias R_32U = Self(12)
    alias C_32U = Self(13)
    alias R_64I = Self(24)
    alias C_64I = Self(25)
    alias R_64U = Self(26)
    alias C_64U = Self(27)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
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
        if self == Self.R_4I:
            return "R_4I"
        if self == Self.C_4I:
            return "C_4I"
        if self == Self.R_4U:
            return "R_4U"
        if self == Self.C_4U:
            return "C_4U"
        if self == Self.R_16I:
            return "R_16I"
        if self == Self.C_16I:
            return "C_16I"
        if self == Self.R_16U:
            return "R_16U"
        if self == Self.C_16U:
            return "C_16U"
        if self == Self.R_32U:
            return "R_32U"
        if self == Self.C_32U:
            return "C_32U"
        if self == Self.R_64I:
            return "R_64I"
        if self == Self.C_64I:
            return "C_64I"
        if self == Self.R_64U:
            return "R_64U"
        if self == Self.C_64U:
            return "C_64U"
        if self == Self.R_8F_E4M3:
            return "R_8F_E4M3"
        if self == Self.R_8F_E5M2:
            return "R_8F_E5M2"

        return abort[String]("invalid DataType entry")

    fn __int__(self) -> Int:
        return Int(self._value)
