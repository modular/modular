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


@value
@register_passable("trivial")
struct LibraryProperty:
    var _value: Int32
    alias MAJOR_VERSION = Self(0)
    alias MINOR_VERSION = Self(1)
    alias PATCH_LEVEL = Self(2)

    @implicit
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
        return abort[String]("invalid LibraryProperty entry")

    fn __int__(self) -> Int:
        return Int(self._value)


@value
@register_passable("trivial")
struct Status(Stringable, Writable):
    var _value: Int8
    alias CUFFT_INVALID_PLAN = Self(1)
    alias CUFFT_SUCCESS = Self(0)
    alias CUFFT_ALLOC_FAILED = Self(2)
    alias CUFFT_INVALID_TYPE = Self(3)
    alias CUFFT_INVALID_VALUE = Self(4)
    alias CUFFT_INTERNAL_ERROR = Self(5)
    alias CUFFT_EXEC_FAILED = Self(6)
    alias CUFFT_SETUP_FAILED = Self(7)
    alias CUFFT_INVALID_SIZE = Self(8)
    alias CUFFT_UNALIGNED_DATA = Self(9)
    alias CUFFT_INCOMPLETE_PARAMETER_LIST = Self(10)
    alias CUFFT_INVALID_DEVICE = Self(11)
    alias CUFFT_PARSE_ERROR = Self(12)
    alias CUFFT_NO_WORKSPACE = Self(13)
    alias CUFFT_NOT_IMPLEMENTED = Self(14)
    alias CUFFT_LICENSE_ERROR = Self(15)
    alias CUFFT_NOT_SUPPORTED = Self(16)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUFFT_SUCCESS:
            return writer.write("CUFFT_SUCCESS")
        if self is Self.CUFFT_INVALID_PLAN:
            return writer.write("CUFFT_INVALID_PLAN")
        if self is Self.CUFFT_ALLOC_FAILED:
            return writer.write("CUFFT_ALLOC_FAILED")
        if self is Self.CUFFT_INVALID_TYPE:
            return writer.write("CUFFT_INVALID_TYPE")
        if self is Self.CUFFT_INVALID_VALUE:
            return writer.write("CUFFT_INVALID_VALUE")
        if self is Self.CUFFT_INTERNAL_ERROR:
            return writer.write("CUFFT_INTERNAL_ERROR")
        if self is Self.CUFFT_EXEC_FAILED:
            return writer.write("CUFFT_EXEC_FAILED")
        if self is Self.CUFFT_SETUP_FAILED:
            return writer.write("CUFFT_SETUP_FAILED")
        if self is Self.CUFFT_INVALID_SIZE:
            return writer.write("CUFFT_INVALID_SIZE")
        if self is Self.CUFFT_UNALIGNED_DATA:
            return writer.write("CUFFT_UNALIGNED_DATA")
        if self is Self.CUFFT_INCOMPLETE_PARAMETER_LIST:
            return writer.write("CUFFT_INCOMPLETE_PARAMETER_LIST")
        if self is Self.CUFFT_INVALID_DEVICE:
            return writer.write("CUFFT_INVALID_DEVICE")
        if self is Self.CUFFT_PARSE_ERROR:
            return writer.write("CUFFT_PARSE_ERROR")
        if self is Self.CUFFT_NO_WORKSPACE:
            return writer.write("CUFFT_NO_WORKSPACE")
        if self is Self.CUFFT_NOT_IMPLEMENTED:
            return writer.write("CUFFT_NOT_IMPLEMENTED")
        if self is Self.CUFFT_LICENSE_ERROR:
            return writer.write("CUFFT_LICENSE_ERROR")
        if self is Self.CUFFT_NOT_SUPPORTED:
            return writer.write("CUFFT_NOT_SUPPORTED")
        abort("invalid cufftResult_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return "cufftResult_t(" + String(self) + ")"

    fn __int__(self) -> Int:
        return Int(self._value)


#  CUFFT defines and supports the following data types


@value
@register_passable("trivial")
struct Type(Stringable, Writable):
    var _value: Int8
    alias CUFFT_R2C = Self(0x2A)
    alias CUFFT_C2R = Self(0x2C)
    alias CUFFT_C2C = Self(0x29)
    alias CUFFT_D2Z = Self(0x6A)
    alias CUFFT_Z2D = Self(0x6C)
    alias CUFFT_Z2Z = Self(0x69)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUFFT_R2C:
            return writer.write("CUFFT_R2C")
        if self is Self.CUFFT_C2R:
            return writer.write("CUFFT_C2R")
        if self is Self.CUFFT_C2C:
            return writer.write("CUFFT_C2C")
        if self is Self.CUFFT_D2Z:
            return writer.write("CUFFT_D2Z")
        if self is Self.CUFFT_Z2D:
            return writer.write("CUFFT_Z2D")
        if self is Self.CUFFT_Z2Z:
            return writer.write("CUFFT_Z2Z")
        abort("invalid cufftType_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return "cufftType_t(" + String(self) + ")"

    fn __int__(self) -> Int:
        return Int(self._value)


@value
@register_passable("trivial")
struct Compatibility(Stringable, Writable):
    var _value: Int8
    alias CUFFT_COMPATIBILITY_FFTW_PADDING = Self(0)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.CUFFT_COMPATIBILITY_FFTW_PADDING:
            return writer.write("CUFFT_COMPATIBILITY_FFTW_PADDING")
        abort("invalid cufftCompatibility_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return "cufftCompatibility_t(" + String(self) + ")"

    fn __int__(self) -> Int:
        return Int(self._value)


@value
@register_passable("trivial")
struct Property(Stringable, Writable):
    var _value: Int8
    alias NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT = Self(0)
    alias NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS = Self(1)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self is Self.NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT:
            return writer.write("NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT")
        if self is Self.NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS:
            return writer.write(
                "NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS"
            )
        abort("invalid cufftProperty_t entry")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return "cufftProperty_t(" + String(self) + ")"

    fn __int__(self) -> Int:
        return Int(self._value)
